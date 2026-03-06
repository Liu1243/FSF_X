"""
hilbert_utils.py

基于查找表的 3D 希尔伯特曲线索引工具。

实现参照：
  Voxel-Mamba/pcdet/models/model_utils/voxel_mamba_utils.py
  中的 get_hilbert_index_3d_mamba_lite，
  针对实例级连续坐标（obj_centers）进行了适配。

使用流程：
1. 调用 build_hilbert_template_3d(order) 生成查找表（通常在模块 __init__ 中完成）。
2. 前向传播时调用 get_hilbert_sort_indices(...) 获取排序索引字典，
   再按索引对特征序列重排后送入 Mamba SSM。
"""

import math
import torch
import numpy as np


# ---------------------------------------------------------------------------
# 离线生成查找表
# ---------------------------------------------------------------------------

def _xy_to_hilbert_d(order: int, x: int, y: int) -> int:
    """将 2D 坐标 (x, y) 映射为 order 阶希尔伯特曲线距离 d（整数）。"""
    n = 1 << order  # 2^order
    d = 0
    s = n >> 1
    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)
        # 旋转
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        s >>= 1
    return d


def build_hilbert_template_2d(order: int) -> torch.Tensor:
    """
    生成 2D 希尔伯特曲线查找表。

    Args:
        order (int): 曲线阶数，空间分辨率为 2^order × 2^order。

    Returns:
        template (Tensor[long]): shape=(2^order * 2^order,)
            template[y * W + x] = hilbert_index
    """
    n = 1 << order
    table = np.zeros(n * n, dtype=np.int64)
    for y in range(n):
        for x in range(n):
            table[y * n + x] = _xy_to_hilbert_d(order, x, y)
    return torch.from_numpy(table)


def build_hilbert_template_3d(order: int) -> torch.Tensor:
    """
    生成 3D 希尔伯特曲线查找表（通过 Z × 2D 平面拼接方式实现）。

    3D 希尔伯特索引定义为：
        hil_index = z_idx * (W * W) + hilbert_2d(x_idx, y_idx)

    这保证了 XY 平面内的空间局部性，同时保留 Z 轴的相对顺序。
    对于 3D 稀疏点云实例场景，此近似方法计算高效，
    且与 Voxel-Mamba 的实践一致。

    Args:
        order (int): 曲线阶数，空间分辨率为 2^order × 2^order 的 XY 平面。

    Returns:
        template (Tensor[long]): shape=(2^order * 2^order,)
            仅包含 XY 平面的 2D 希尔伯特索引，Z 轴偏移在 get_hilbert_sort_indices
            中以 z_idx * plane_size 叠加。
    """
    return build_hilbert_template_2d(order)


# ---------------------------------------------------------------------------
# 前向传播时的在线索引计算
# ---------------------------------------------------------------------------

def get_hilbert_sort_indices(
    centers: torch.Tensor,
    batch_idx: torch.Tensor,
    batch_size: int,
    template: torch.Tensor,
    grid_range: list,
    hilbert_order: int,
) -> dict:
    """
    将实例 3D 中心坐标按希尔伯特曲线距离排序，返回排序索引字典。

    参照 voxel_mamba_utils.get_hilbert_index_3d_mamba_lite，
    针对连续坐标（而非离散体素坐标）适配。

    Args:
        centers (Tensor[float]): shape=(N, 3)，实例 3D 中心坐标 (x, y, z)（LiDAR 系）。
        batch_idx (Tensor[long]): shape=(N,)，各实例所属 batch 索引。
        batch_size (int): batch 大小。
        template (Tensor[long]): shape=(W*W,) 2D 希尔伯特查找表（由 build_hilbert_template_3d 生成）。
        grid_range (list): [[x_min, x_max], [y_min, y_max], [z_min, z_max]]，
            用于将连续坐标量化到 [0, 2^order) 范围。
        hilbert_order (int): 希尔伯特曲线阶数，grid 分辨率 = 2^order。

    Returns:
        index_info (dict):
            'inds_curt_to_next' (dict[int -> Tensor]): 每个 batch_id 对应的排序索引
                （将当前顺序→希尔伯特顺序）
            'inds_next_to_curt' (dict[int -> Tensor]): 逆排列索引
                （将希尔伯特顺序→当前顺序）
    """
    grid_size = 1 << hilbert_order  # 2^order
    plane_size = grid_size * grid_size  # XY 平面大小

    # ---------- 坐标量化 ----------
    x_min, x_max = grid_range[0]
    y_min, y_max = grid_range[1]
    z_min, z_max = grid_range[2]

    # 归一化到 [0, 1)，再乘以 grid_size 得离散格索引
    x_norm = (centers[:, 0] - x_min) / (x_max - x_min + 1e-6)
    y_norm = (centers[:, 1] - y_min) / (y_max - y_min + 1e-6)
    z_norm = (centers[:, 2] - z_min) / (z_max - z_min + 1e-6)

    x_idx = (x_norm.clamp(0.0, 1.0 - 1e-6) * grid_size).long()
    y_idx = (y_norm.clamp(0.0, 1.0 - 1e-6) * grid_size).long()
    z_idx = (z_norm.clamp(0.0, 1.0 - 1e-6) * grid_size).long()

    # ---------- 2D 平面 Hilbert 索引 ----------
    if template.device != centers.device:
        template = template.to(centers.device)

    flat_xy = (y_idx * grid_size + x_idx).long()           # (N,)
    hil_xy = template[flat_xy].long()                       # (N,)  2D 希尔伯特距离

    # 3D 希尔伯特距离 = z_offset + 2D_hilbert
    hil_inds = z_idx * plane_size + hil_xy                  # (N,)

    # ---------- 逐 batch 排序 ----------
    inds_curt_to_next = {}
    inds_next_to_curt = {}

    for i in range(batch_size):
        batch_mask = (batch_idx == i)
        if batch_mask.sum() == 0:
            inds_curt_to_next[i] = torch.zeros(0, dtype=torch.long, device=centers.device)
            inds_next_to_curt[i] = torch.zeros(0, dtype=torch.long, device=centers.device)
            continue
        batch_hil = hil_inds[batch_mask]
        sort_idx = torch.argsort(batch_hil)          # curt → next（希尔伯特升序）
        inv_idx  = torch.argsort(sort_idx)            # next → curt（还原原顺序）
        inds_curt_to_next[i] = sort_idx
        inds_next_to_curt[i] = inv_idx

    return {
        'inds_curt_to_next': inds_curt_to_next,
        'inds_next_to_curt': inds_next_to_curt,
    }
