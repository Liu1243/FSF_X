"""
h_mamba 子包 __init__.py

提供基于希尔伯特曲线的双向 Mamba 特征交互模块。
"""

from .hilbert_utils import (
    build_hilbert_template_2d,
    build_hilbert_template_3d,
    get_hilbert_sort_indices,
)
from .h_mamba_interaction import HilbertMambaInteraction

__all__ = [
    'HilbertMambaInteraction',
    'build_hilbert_template_2d',
    'build_hilbert_template_3d',
    'get_hilbert_sort_indices',
]
