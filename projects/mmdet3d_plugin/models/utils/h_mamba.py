"""
H-Mamba Interaction Module for FSF-X Architecture
=================================================
通过选择性状态空间模型（Selective SSM）替代 O(m²) 自注意力机制，
以线性复杂度 O(m·N) 完成 LiDAR 与 Camera 实例的几何-语义跨模态融合。

数学核心（递推关系）：
    h_k = Ā·h_{k-1} + B̄·x_k
    y_k =  C̄·h_k    + D·x_k

其中 B̄、C̄、Δ 均为输入依赖（input-dependent），赋予模型动态过滤视锥噪声
和补全稀疏 LiDAR 特征的能力。

依赖：
    mamba_ssm >= 1.0.1  （含 selective_scan_fn CUDA 算子）
    安装：pip install mamba-ssm
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    MAMBA_AVAILABLE = True
except ImportError:
    # 降级：纯 PyTorch 顺序扫描（仅用于 CPU 调试，不推荐生产使用）
    selective_scan_fn = None
    MAMBA_AVAILABLE = False


def selective_scan_ref(u, delta, A, B, C, D=None, delta_bias=None,
                       delta_softplus=False, return_last_state=False):
    """纯 PyTorch 参考实现的选择性并行扫描（仅供 mamba_ssm 不可用时降级使用）。

    Args:
        u      : (B, d_inner, L)  输入序列
        delta  : (B, d_inner, L)  步长（已离散化之前）
        A      : (d_inner, d_state)  连续状态矩阵（负数，取 log 后存储）
        B      : (B, d_state, L)   输入依赖 B
        C      : (B, d_state, L)   输入依赖 C
        D      : (d_inner,) 可选残差项
    Returns:
        y      : (B, d_inner, L)
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)

    batch, d_inner, L = u.shape
    d_state = A.shape[1]
    A = A.float()
    B = B.float()
    C = C.float()

    # ZOH 离散化：Ā = exp(Δ·A),  B̄ = (Ā - I) / A * B ≈ Δ·B（近似）
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))  # (B, d_inner, L, d_state)
    deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)   # (B, d_inner, L, d_state)

    # 顺序扫描（非并行，仅供调试）
    last_state = u.new_zeros((batch, d_inner, d_state))
    ys = []
    for i in range(L):
        last_state = deltaA[:, :, i] * last_state + deltaB_u[:, :, i]
        y = torch.einsum('bdn,bn->bd', last_state, C[:, :, i])
        ys.append(y)
    y = torch.stack(ys, dim=2)  # (B, d_inner, L)

    if D is not None:
        y = y + u * D[:, None]

    return y.to(dtype=dtype_in), last_state if return_last_state else y.to(dtype=dtype_in)


class HMambaInteraction(nn.Module):
    """双模态实例序列的 H-Mamba 跨模态交互模块。

    核心思想：
        将 LiDAR 实例特征（精确几何）与 Camera 实例特征（丰富语义）
        拼接为一条 1D 实例序列，利用选择性 SSM 完成 O(m·N) 的跨模态
        信息传递，替代传统的 O(m²) 自注意力机制。

    Args:
        d_model      (int): 输入/输出特征维度（即 embed_dims），默认 256
        d_state      (int): SSM 状态空间维度 N，默认 16
        expand_factor(int): 内部扩展因子（d_inner = expand_factor × d_model），默认 2
        dt_rank      (str|int): Δ 的秩，'auto' 时设为 ceil(d_model/16)
        dt_min       (float): Δ 的最小值（初始化）
        dt_max       (float): Δ 的最大值（初始化）
        dt_init      (str): Δ 初始化策略，'random' 或 'constant'
        dt_scale     (float): Δ 初始化缩放系数
        conv_kernel  (int): 局部上下文卷积核大小，默认 4
        bias         (bool): 是否在线性层使用偏置
        use_fast_path(bool): 是否使用 mamba_ssm CUDA 算子（默认 True）
    """

    def __init__(
        self,
        d_model: int = 256,
        d_state: int = 16,
        expand_factor: int = 2,
        dt_rank: str = 'auto',
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = 'random',
        dt_scale: float = 1.0,
        conv_kernel: int = 4,
        bias: bool = False,
        use_fast_path: bool = True,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand_factor * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == 'auto' else dt_rank
        self.use_fast_path = use_fast_path and MAMBA_AVAILABLE

        # ------------------------------------------------------------------ #
        # 1. 输入投影：d_model → d_inner * 2  (SSM 数据流 x + 门控流 z)
        # ------------------------------------------------------------------ #
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        # ------------------------------------------------------------------ #
        # 2. 局部上下文聚合（深度可分离卷积）
        #    kernel = conv_kernel，padding = conv_kernel - 1（因果）
        # ------------------------------------------------------------------ #
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=conv_kernel,
            padding=conv_kernel - 1,
            groups=self.d_inner,
            bias=True,
        )

        # ------------------------------------------------------------------ #
        # 3. 输入依赖参数投影：x → (Δ_rank, B, C)
        #    Δ: 控制"记忆衰减"速率
        #    B: 控制新信息的写入门控（过滤视锥噪声）
        #    C: 控制隐藏状态的读取门控（几何-语义协同）
        # ------------------------------------------------------------------ #
        self.x_proj = nn.Linear(
            self.d_inner,
            self.dt_rank + self.d_state * 2,
            bias=False,
        )

        # Δ 从 dt_rank 升维到 d_inner（线性层，含 bias 作为 dt_bias）
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Δ 初始化：使 exp(Δ·A) ≈ 适当衰减率
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == 'constant':
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == 'random':
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        # 将 dt_bias 初始化为使 softplus(·) 落在 [dt_min, dt_max] 的区间内
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))   # softplus inverse
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True  # 跳过常规权重初始化

        # ------------------------------------------------------------------ #
        # 4. SSM 连续状态矩阵 A（负数，以 log|A| 形式存储保证稳定性）
        #    初始化为 A_ij = -(i+1)，使特征值均为负实数
        # ------------------------------------------------------------------ #
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        # ------------------------------------------------------------------ #
        # 5. 残差直连系数 D（对应 y_k = C·h_k + D·x_k 中的 D）
        # ------------------------------------------------------------------ #
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        # ------------------------------------------------------------------ #
        # 6. 输出归一化 + 投影
        # ------------------------------------------------------------------ #
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    # ---------------------------------------------------------------------- #
    #  前向传播
    # ---------------------------------------------------------------------- #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对合并后的双模态实例序列执行 H-Mamba 跨模态交互。

        Args:
            x (Tensor): [batch, seq_len, d_model]
                seq_len = M_L + M_C，前 M_L 个为 LiDAR 实例，后 M_C 个为 Camera 实例

        Returns:
            Tensor: [batch, seq_len, d_model]，融合后的双模态实例特征
        """
        B, L, D = x.shape
        assert D == self.d_model, \
            f"输入维度 {D} 与 d_model {self.d_model} 不匹配"

        # -------- 步骤 1：输入投影 → 分离 SSM 流和门控流 -------- #
        xz = self.in_proj(x)                        # (B, L, d_inner*2)
        x_ssm, z = xz.chunk(2, dim=-1)              # 各 (B, L, d_inner)

        # -------- 步骤 2：局部上下文聚合（Conv1d） -------- #
        x_ssm = x_ssm.transpose(1, 2)              # → (B, d_inner, L)
        x_ssm = self.conv1d(x_ssm)[..., :L]        # 截断到原长，因果对齐
        x_ssm = F.silu(x_ssm)                      # 激活

        # -------- 步骤 3：计算输入依赖参数 Δ、B、C -------- #
        # x_ssm: (B, d_inner, L) → 转置为 (B, L, d_inner) 再投影
        x_dbl = x_ssm.transpose(1, 2)              # (B, L, d_inner)
        x_proj_out = self.x_proj(x_dbl)            # (B, L, dt_rank + 2*d_state)

        delta_raw = x_proj_out[..., :self.dt_rank]             # (B, L, dt_rank)
        B_param  = x_proj_out[..., self.dt_rank: self.dt_rank + self.d_state]   # (B, L, d_state)
        C_param  = x_proj_out[..., self.dt_rank + self.d_state:]                # (B, L, d_state)

        # Δ 升维 → (B, L, d_inner)，再转置为 (B, d_inner, L)
        delta = self.dt_proj(delta_raw).transpose(1, 2)        # (B, d_inner, L)

        # B、C 转置为 (B, d_state, L) 以符合 selective_scan_fn 接口
        B_param = B_param.transpose(1, 2)          # (B, d_state, L)
        C_param = C_param.transpose(1, 2)          # (B, d_state, L)

        # 连续矩阵 A：取回负实数值
        A = -torch.exp(self.A_log.float())         # (d_inner, d_state)

        # -------- 步骤 4：选择性状态空间扫描 -------- #
        if self.use_fast_path:
            # 使用 mamba_ssm 的 CUDA 硬件感知并行扫描算子
            # 接口：selective_scan_fn(u, delta, A, B, C, D, z=None,
            #                         delta_bias, delta_softplus, return_last_state)
            y = selective_scan_fn(
                u=x_ssm,                           # (B, d_inner, L)
                delta=delta,                       # (B, d_inner, L)
                A=A,                               # (d_inner, d_state)
                B=B_param,                         # (B, d_state, L)
                C=C_param,                         # (B, d_state, L)
                D=self.D.float(),                  # (d_inner,)
                z=None,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=False,
            )                                      # → (B, d_inner, L)
        else:
            # 降级到纯 PyTorch 参考实现（CPU 调试用）
            y, _ = selective_scan_ref(
                u=x_ssm,
                delta=delta,
                A=A,
                B=B_param,
                C=C_param,
                D=self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=True,
            )

        # -------- 步骤 5：门控融合 + 输出投影 -------- #
        y = y.transpose(1, 2)                      # → (B, L, d_inner)
        y = self.out_norm(y)
        y = y * F.silu(z)                          # Gated output: y ⊙ σ(z)
        out = self.out_proj(y)                     # → (B, L, d_model)

        return out
