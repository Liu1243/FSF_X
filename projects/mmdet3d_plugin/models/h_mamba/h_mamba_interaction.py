"""
h_mamba_interaction.py

基于希尔伯特曲线的双向 Mamba 特征交互模块（H-Mamba Interaction）。

理论依据（来自 idea.md）：
  - 使用 3D 希尔伯特空间填充曲线将无序实例特征排列为保留局部拓扑的 1D 序列
  - 通过双向 SSM (Bidirectional Mamba) 建模全局上下文：
      Y = SSM_forward(X) + SSM_backward(X)

实现参照：
  Voxel-Mamba/pcdet/models/backbones_3d/voxel_mamba_waymo.py 中的 DSB 块
"""

import math
from functools import partial

import torch
import torch.nn as nn

from .hilbert_utils import build_hilbert_template_3d, get_hilbert_sort_indices

try:
    from mamba_ssm.models.mixer_seq_simple import create_block as mamba_create_block
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    mamba_create_block = None


# ---------------------------------------------------------------------------
# 权重初始化（参照 Voxel-Mamba）
# ---------------------------------------------------------------------------

def _init_weights(module, n_layer, initializer_range=0.02, n_residuals_per_layer=2):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    for name, p in module.named_parameters():
        if name in ["out_proj.weight", "fc2.weight"]:
            nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            with torch.no_grad():
                p /= math.sqrt(n_residuals_per_layer * n_layer)


# ---------------------------------------------------------------------------
# 轻量位置编码 MLP
# ---------------------------------------------------------------------------

def _build_pos_embed_mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    """将 3D 归一化坐标映射到特征空间的小 MLP。"""
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(inplace=True),
        nn.Linear(out_dim, out_dim),
    )


# ---------------------------------------------------------------------------
# 核心模块：HilbertMambaInteraction
# ---------------------------------------------------------------------------

class HilbertMambaInteraction(nn.Module):
    """
    基于希尔伯特曲线的双向 Mamba 特征交互模块（H-Mamba Interaction）。

    将无序实例特征按 3D Hilbert 曲线排列为有拓扑感知的 1D 序列，
    通过正向 + 反向双向 SSM（Mamba）聚合全局上下文后残差输出。

    数学形式（idea.md 公式）：
        h_i = H(x_i, y_i, z_i)          （希尔伯特索引）
        X = sort(F, h_i)                 （序列化）
        Y = SSM_fwd(X) + SSM_bwd(X)      （双向 SSM）
        Output = F + unorder(Y)          （残差还原）

    Args:
        d_model (int): 特征维度（须与上游输出维度一致）。
        hilbert_order (int): 希尔伯特曲线阶数，空间分辨率 = 2^order。
            默认 6（64×64 XY 网格），适合 nuScenes ±51.2m 感知范围。
        num_layers (int): 堆叠的双向 Mamba 块数量，默认 1。
        grid_range (list): [[x_min,x_max], [y_min,y_max], [z_min,z_max]]
            用于将连续坐标量化。默认适用 nuScenes 标准范围。
        ssm_cfg (dict | None): 传给 mamba_ssm.create_block 的额外 SSM 配置。
        norm_epsilon (float): LayerNorm eps，默认 1e-5。
        rms_norm (bool): 是否使用 RMSNorm，默认 False（使用 LayerNorm）。
        residual_in_fp32 (bool): SSM 残差是否强制使用 fp32，默认 True。
        fused_add_norm (bool): 是否使用融合 AddNorm，默认 False。
        pos_embed_dim (int): 位置编码输入维度，默认 3（直接用归一化 xyz）。
        device (str | None): 设备，默认 None（跟随输入）。
        dtype: 数据类型，默认 torch.float32。
    """

    def __init__(
        self,
        d_model: int,
        hilbert_order: int = 6,
        num_layers: int = 1,
        grid_range: list = None,
        ssm_cfg: dict = None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        residual_in_fp32: bool = True,
        fused_add_norm: bool = False,
        pos_embed_dim: int = 3,
        device=None,
        dtype=torch.float32,
    ):
        super().__init__()

        if not MAMBA_AVAILABLE:
            raise ImportError(
                "mamba_ssm 未安装。请先安装：pip install mamba_ssm，"
                "或参照 Voxel-Mamba 目录下的 setup.py 进行本地安装。"
            )

        self.d_model = d_model
        self.hilbert_order = hilbert_order
        self.num_layers = num_layers
        self.norm_epsilon = norm_epsilon

        # 默认 nuScenes 感知范围（±51.2m XY, -5~3m Z）
        if grid_range is None:
            grid_range = [[-51.2, 51.2], [-51.2, 51.2], [-5.0, 3.0]]
        self.grid_range = grid_range

        factory_kwargs = {"device": device, "dtype": dtype}
        norm_cls = partial(nn.LayerNorm, eps=norm_epsilon)

        # ---------- Hilbert 查找表（离线生成，注册为 buffer） ----------
        template = build_hilbert_template_3d(hilbert_order)   # (W*W,)
        self.register_buffer("hilbert_template", template, persistent=False)

        # ---------- 位置编码 MLP ----------
        self.pos_embed = _build_pos_embed_mlp(pos_embed_dim, d_model)

        # ---------- 双向 Mamba 块（每层两个 SSM：正向 + 反向） ----------
        self.mamba_fwd_layers = nn.ModuleList()
        self.mamba_bwd_layers = nn.ModuleList()
        self.norm_fwd_layers  = nn.ModuleList()
        self.norm_bwd_layers  = nn.ModuleList()

        for layer_idx in range(num_layers):
            self.mamba_fwd_layers.append(
                mamba_create_block(
                    d_model=d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=layer_idx * 2,
                    **factory_kwargs,
                )
            )
            self.mamba_bwd_layers.append(
                mamba_create_block(
                    d_model=d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=layer_idx * 2 + 1,
                    **factory_kwargs,
                )
            )
            self.norm_fwd_layers.append(norm_cls(d_model))
            self.norm_bwd_layers.append(norm_cls(d_model))

        # ---------- 输入归一化 ----------
        self.norm_in = norm_cls(d_model)

        # ---------- 参数初始化 ----------
        self.apply(
            partial(_init_weights, n_layer=num_layers * 2)
        )

    # ------------------------------------------------------------------

    def _pos_encode(self, centers: torch.Tensor) -> torch.Tensor:
        """将 3D 中心坐标归一化后输入位置编码 MLP。"""
        x_min, x_max = self.grid_range[0]
        y_min, y_max = self.grid_range[1]
        z_min, z_max = self.grid_range[2]

        norm_xyz = torch.stack([
            (centers[:, 0] - x_min) / (x_max - x_min + 1e-6),
            (centers[:, 1] - y_min) / (y_max - y_min + 1e-6),
            (centers[:, 2] - z_min) / (z_max - z_min + 1e-6),
        ], dim=-1).clamp(0.0, 1.0)   # (N, 3)

        return self.pos_embed(norm_xyz)   # (N, d_model)

    # ------------------------------------------------------------------

    def forward(
        self,
        obj_feats: torch.Tensor,
        obj_centers: torch.Tensor,
        batch_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播。

        Args:
            obj_feats (Tensor): shape=(N, d_model)，实例特征（来自 frustum + fsd 合并）。
            obj_centers (Tensor): shape=(N, 3)，实例 3D 中心坐标 (x, y, z)，LiDAR 系。
            batch_idx (Tensor[long]): shape=(N,)，各实例所属 batch 索引。

        Returns:
            out_feats (Tensor): shape=(N, d_model)，经全局上下文建模后的增强特征（残差相加）。
        """
        batch_size = int(batch_idx.max().item()) + 1

        # 1. 位置编码
        pos_emb = self._pos_encode(obj_centers)          # (N, d_model)
        feats = obj_feats + pos_emb                      # (N, d_model)

        # 2. 输入归一化
        feats = self.norm_in(feats)

        # 3. 计算希尔伯特排序索引
        index_info = get_hilbert_sort_indices(
            centers=obj_centers,
            batch_idx=batch_idx,
            batch_size=batch_size,
            template=self.hilbert_template,
            grid_range=self.grid_range,
            hilbert_order=self.hilbert_order,
        )
        inds_curt_to_next = index_info['inds_curt_to_next']
        inds_next_to_curt = index_info['inds_next_to_curt']

        # 4. 逐层双向 Mamba
        residual_feats = feats
        for layer_idx in range(self.num_layers):
            mamba_fwd = self.mamba_fwd_layers[layer_idx]
            mamba_bwd = self.mamba_bwd_layers[layer_idx]
            norm_fwd  = self.norm_fwd_layers[layer_idx]
            norm_bwd  = self.norm_bwd_layers[layer_idx]

            out_fwd = torch.zeros_like(residual_feats)
            out_bwd = torch.zeros_like(residual_feats)

            # 逐 batch 处理（与 Voxel-Mamba DSB.forward 一致）
            for i in range(batch_size):
                b_mask = (batch_idx == i)
                if b_mask.sum() == 0:
                    continue

                # --- 正向 SSM（按希尔伯特顺序）---
                feat_i = residual_feats[b_mask]                      # (Ni, C)
                feat_sorted = feat_i[inds_curt_to_next[i]][None]    # (1, Ni, C)
                out_fwd_i, _ = mamba_fwd(feat_sorted, None)
                out_fwd[b_mask] = out_fwd_i.squeeze(0)[inds_next_to_curt[i]]

                # --- 反向 SSM（翻转序列，等价于反向扫描）---
                feat_back = feat_sorted.flip(1)                      # (1, Ni, C)
                out_bwd_i, _ = mamba_bwd(feat_back, None)
                out_bwd[b_mask] = out_bwd_i.squeeze(0).flip(0)[inds_next_to_curt[i]]

            # 双向融合 + 归一化
            residual_feats = norm_fwd(out_fwd) + norm_bwd(out_bwd)  # (N, C)

        # 5. 残差连接：将增强特征叠加回原始特征
        out_feats = obj_feats + residual_feats

        return out_feats
