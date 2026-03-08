"""
FSF_HMamba: Fully Sparse Fusion with H-Mamba Cross-Modal Interaction
=====================================================================
在 FSF 基线架构的基础上，将 `combine_frustum_and_fsd` 步骤中的
简单 MLP 映射 + concat 替换为 H-Mamba 选择性 SSM 跨模态交互。

核心改动：
    原始 FSF：
        frustum_feat → MLP → embed_dims
        fsd_feat    → MLP → embed_dims
        concat → obj_feats (2·embed_dims slice)  ← 无跨实例交互

    FSF_HMamba：
        frustum_feat → MLP → embed_dims  ┐
                                          ├─ cat → [M_L+M_C, D] → HMamba → fused_obj_feats
        fsd_feat    → MLP → embed_dims  ┘

    H-Mamba 以 O(m·N) 复杂度完成跨模态实例感知，
    输出 fused_obj_feats 直接送入多阶段精炼头。
"""

import torch
from torch import nn
from mmdet.models import DETECTORS

from .FSF import FSF
from projects.mmdet3d_plugin.models.utils.h_mamba import HMambaInteraction


@DETECTORS.register_module()
class FSF_HMamba(FSF):
    """带 H-Mamba 跨模态实例交互的 FSF 检测器。

    相较于 FSF，仅在 `__init__` 中额外构建 HMambaInteraction 模块，
    并重写 `combine_frustum_and_fsd` 方法，其余完全继承 FSF 的行为。

    额外 Args（通过 hmamba_cfg dict 传入）：
        d_model      (int): H-Mamba 输入/输出维度，应与 embed_dims 一致，默认 256
        d_state      (int): SSM 状态空间维度 N，默认 16
        expand_factor(int): 内部扩展因子，默认 2
        dt_rank      (str|int): Δ 的秩，'auto' 则 = ceil(d_model/16)
        conv_kernel  (int): 局部上下文卷积核大小，默认 4
        use_fast_path(bool): 是否使用 mamba_ssm CUDA 核，默认 True

    示例 cfg（追加到 model dict 中）：
        hmamba_cfg=dict(d_model=1024, d_state=16, expand_factor=2)
    """

    def __init__(self, hmamba_cfg=None, **kwargs):
        super().__init__(**kwargs)

        # ------------------------------------------------------------------ #
        # H-Mamba 交互模块
        # d_model 默认与 embed_dims 对齐（combine 后特征维度 = embed_dims）
        # ------------------------------------------------------------------ #
        if hmamba_cfg is None:
            hmamba_cfg = {}

        # embed_dims 由父类的 mlp_cfg 决定，默认 256
        d_model = hmamba_cfg.get('d_model', self.embed_dims)

        self.h_mamba = HMambaInteraction(
            d_model=d_model,
            d_state=hmamba_cfg.get('d_state', 16),
            expand_factor=hmamba_cfg.get('expand_factor', 2),
            dt_rank=hmamba_cfg.get('dt_rank', 'auto'),
            conv_kernel=hmamba_cfg.get('conv_kernel', 4),
            use_fast_path=hmamba_cfg.get('use_fast_path', True),
        )

    # ---------------------------------------------------------------------- #
    #  重写 combine_frustum_and_fsd
    # ---------------------------------------------------------------------- #

    def combine_frustum_and_fsd(
        self,
        frustum_obj_centers,
        frustum_obj_coors,
        frustum_obj_result,
        frustum_obj_feats,
        frustum_preds_2d,
        fsd_obj_centers,
        fsd_obj_coors,
        fsd_obj_result,
        fsd_obj_feats,
    ):
        """合并两路 query 并通过 H-Mamba 完成跨模态实例交互。

        相较于 FSF.combine_frustum_and_fsd，主要差异：
            1. 两路特征经各自 MLP 投影到 embed_dims 后，
               先按 [frustum | fsd] 顺序拼接成单条 1D 实例序列；
            2. 整条序列送入 HMambaInteraction，完成 O(m·N) 的几何-语义协同；
            3. H-Mamba 输出与原始 MLP 映射结果做残差相加（训练稳定性）；
            4. 最终 fused_obj_feats 直接替代原始 obj_feats 参与后续精炼。

        Args:
            frustum_obj_centers : (M_C, 3) 视锥实例中心坐标
            frustum_obj_coors   : (M_C, 3) 视锥实例索引 (batch, ?, obj_id)
            frustum_obj_result  : dict     视锥初始检测结果
            frustum_obj_feats   : (M_C, C_frustum) 视锥原始特征（MLP 投影前）
            frustum_preds_2d    : (M_C, 9) 视锥 2D 预测信息
            fsd_obj_centers     : (M_L, 3) LiDAR 实例中心坐标
            fsd_obj_coors       : (M_L, 3) LiDAR 实例索引
            fsd_obj_result      : dict     LiDAR 初始检测结果
            fsd_obj_feats       : (M_L, C_lidar) LiDAR 原始特征（MLP 投影前）

        Returns:
            obj_centers   : (M_L+M_C, 3)
            obj_coors     : (M_L+M_C, 3)
            obj_result    : dict
            fused_obj_feats: (M_L+M_C, embed_dims)  经 H-Mamba 融合的特征
            preds_2d      : (M_L+M_C, 9)
        """
        # ---- Step 1：中心坐标与坐标索引合并（与原 FSF 完全相同）---- #
        obj_centers = torch.cat([frustum_obj_centers, fsd_obj_centers], dim=0)

        fsd_obj_coors_re = fsd_obj_coors.clone()
        fsd_obj_coors_re[:, 0] = fsd_obj_coors[:, 1]
        fsd_obj_coors_re[:, 1] = fsd_obj_coors[:, 0]
        fsd_obj_coors_re[:, 2] += self.fsd_begin_idx
        obj_coors = torch.cat([frustum_obj_coors, fsd_obj_coors_re], dim=0)

        # ---- Step 2：合并检测结果（与原 FSF 完全相同）---- #
        obj_result = {}
        for key in frustum_obj_result.keys():
            batch_size = len(frustum_obj_result[key])
            obj_result[key] = []
            for bidx in range(batch_size):
                data = torch.cat(
                    [frustum_obj_result[key][bidx], fsd_obj_result[key][bidx]],
                    dim=0
                )
                obj_result[key].append(data)

        # ---- Step 3：各自 MLP 投影到 embed_dims ---- #
        #   frustum_proj : (M_C, embed_dims)
        #   fsd_proj     : (M_L, embed_dims)
        frustum_proj = self.combine_frustum_feat_mlp(frustum_obj_feats)
        fsd_proj = self.combine_fsd_feat_mlp(fsd_obj_feats)

        # 原始拼接特征（作为残差基线 & 序列构建）
        # shape: (M_L + M_C, embed_dims)
        obj_feats_cat = torch.cat([frustum_proj, fsd_proj], dim=0)

        # ---- Step 4：H-Mamba 跨模态实例交互 ---- #
        # 注意：selective_scan 需要 batch 维度；此处全场景实例扁平存储（batch=1 的伪批次）
        # 如果有多个 batch 的实例混在一起（batch_inds 混合），
        # 先整体扫描（全局跨 batch 交互），再分 batch 精炼。
        # 实践中单 sample 推理（bz=1）时不存在歧义；
        # 训练多 batch 时各 batch 实例序列各自独立更优——见下方 per-batch 处理。

        m = obj_feats_cat.shape[0]

        if m == 0:
            # 边界情况：完全无实例（极少情况）
            fused_obj_feats = obj_feats_cat
        else:
            # 获取每个实例归属的 batch_id（frustum coors[:,0]，fsd coors重排后[:,0]）
            batch_ids = obj_coors[:, 0]  # (M_L+M_C,)
            unique_batches = batch_ids.unique(sorted=True)

            fused_list = []
            for bid in unique_batches:
                bmask = (batch_ids == bid)
                feat_b = obj_feats_cat[bmask]          # (m_b, embed_dims)
                m_b = feat_b.shape[0]

                if m_b == 0:
                    fused_list.append(feat_b)
                    continue

                # H-Mamba 期望输入 (batch=1, seq_len, d_model)
                feat_b_3d = feat_b.unsqueeze(0)         # (1, m_b, embed_dims)
                delta_b = self.h_mamba(feat_b_3d)       # (1, m_b, embed_dims)
                delta_b = delta_b.squeeze(0)            # (m_b, embed_dims)

                # 残差连接：保留原始 MLP 映射的梯度通路
                fused_list.append(feat_b + delta_b)

            fused_obj_feats = torch.cat(fused_list, dim=0)  # (M_L+M_C, embed_dims)

        # ---- Step 5：2D 预测信息合并（与原 FSF 完全相同）---- #
        fsd_preds_2d = frustum_preds_2d.new_zeros(
            (fsd_obj_feats.shape[0], frustum_preds_2d.shape[1])
        )
        preds_2d = torch.cat([frustum_preds_2d, fsd_preds_2d], dim=0)

        return obj_centers, obj_coors, obj_result, fused_obj_feats, preds_2d
