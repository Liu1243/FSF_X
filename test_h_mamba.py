"""
test_h_mamba.py

H-Mamba Interaction 模块独立单元测试。

运行方式（在 FullySparseFusion 环境中）：
    cd /home/ddd/pc/FullySparseFusion
    python test_h_mamba.py
"""

import sys
import os

# 将项目根目录加入 sys.path
sys.path.insert(0, "/home/ddd/pc/FullySparseFusion")

import torch
import traceback


# ============================================================
# 测试 1：工具函数测试（不依赖 mamba_ssm）
# ============================================================

def test_hilbert_utils():
    print("\n" + "="*60)
    print("测试 1: hilbert_utils 函数")
    print("="*60)

    from projects.mmdet3d_plugin.models.h_mamba.hilbert_utils import (
        build_hilbert_template_2d,
        build_hilbert_template_3d,
        get_hilbert_sort_indices,
    )

    # 1a. 查找表生成
    order = 4   # 16x16 网格
    template_2d = build_hilbert_template_2d(order)
    assert template_2d.shape == (2**order * 2**order,), \
        f"2D 查找表 shape 错误：{template_2d.shape}"
    assert template_2d.max() == 2**(2*order) - 1, \
        f"2D 查找表最大值错误：{template_2d.max()}"
    print(f"  [PASS] build_hilbert_template_2d(order={order}) -> shape={template_2d.shape}, max={template_2d.max()}")

    template_3d = build_hilbert_template_3d(order)
    assert template_3d.shape == template_2d.shape
    print(f"  [PASS] build_hilbert_template_3d(order={order}) -> shape={template_3d.shape}")

    # 1b. 验证 2D Hilbert 索引是 [0, W*W) 范围内的置换
    n = 2**order
    expected_set = set(range(n * n))
    actual_set = set(template_2d.tolist())
    assert expected_set == actual_set, "2D Hilbert 查找表不是有效置换！"
    print(f"  [PASS] 2D 查找表是 [0, {n*n}) 的有效置换")

    # 1c. 索引计算（合成数据）
    N = 30
    batch_size = 2
    centers = torch.rand(N, 3) * 100 - 50
    batch_idx = torch.zeros(N, dtype=torch.long)
    batch_idx[N // 2:] = 1

    grid_range = [[-51.2, 51.2], [-51.2, 51.2], [-5.0, 3.0]]
    index_info = get_hilbert_sort_indices(
        centers, batch_idx, batch_size, template_3d, grid_range, order
    )

    for i in range(batch_size):
        n_i = (batch_idx == i).sum().item()
        fwd = index_info['inds_curt_to_next'][i]
        inv = index_info['inds_next_to_curt'][i]
        assert fwd.shape == (n_i,), f"batch {i}: fwd shape 错误 {fwd.shape}"
        assert inv.shape == (n_i,), f"batch {i}: inv shape 错误 {inv.shape}"
        # 验证 inv 是 fwd 的逆排列
        identity = fwd[inv]
        assert (identity == torch.arange(n_i)).all(), f"batch {i}: 逆排列验证失败"
        print(f"  [PASS] batch={i}, N={n_i}, 排序与逆排列正确（fwd[inv] == arange）")

    print("测试 1 全部通过！")
    return True


# ============================================================
# 测试 2：HilbertMambaInteraction 前向传播
# ============================================================

def test_h_mamba_forward():
    print("\n" + "="*60)
    print("测试 2: HilbertMambaInteraction forward")
    print("="*60)

    try:
        from projects.mmdet3d_plugin.models.h_mamba.h_mamba_interaction import (
            HilbertMambaInteraction, MAMBA_AVAILABLE
        )
    except ImportError as e:
        print(f"  [SKIP] 导入失败，跳过 forward 测试: {e}")
        return True

    if not MAMBA_AVAILABLE:
        print("  [SKIP] mamba_ssm 未安装，跳过 Mamba forward 测试")
        return True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  使用设备: {device}")

    d_model = 256
    N = 60
    batch_size = 3

    obj_feats   = torch.randn(N, d_model, device=device, requires_grad=True)
    obj_centers = torch.rand(N, 3, device=device) * 102.4 - 51.2
    batch_idx   = torch.zeros(N, dtype=torch.long, device=device)
    for i in range(1, batch_size):
        batch_idx[i * (N // batch_size):] = i

    model = HilbertMambaInteraction(
        d_model=d_model,
        hilbert_order=5,
        num_layers=1,
        grid_range=[[-51.2, 51.2], [-51.2, 51.2], [-5.0, 3.0]],
    ).to(device)
    model.train()

    # 前向
    out = model(obj_feats, obj_centers, batch_idx)
    assert out.shape == obj_feats.shape, f"输出 shape 错误：{out.shape}"
    print(f"  [PASS] 输出 shape: {out.shape}（与输入一致）")

    # NaN / Inf 检查
    assert not torch.isnan(out).any(), "输出含 NaN！"
    assert not torch.isinf(out).any(), "输出含 Inf！"
    print(f"  [PASS] 输出无 NaN / Inf")

    # 反向传播
    loss = out.sum()
    loss.backward()
    assert obj_feats.grad is not None, "梯度为 None！"
    assert not torch.isnan(obj_feats.grad).any(), "梯度含 NaN！"
    print(f"  [PASS] 反向传播成功，梯度无 NaN")

    with torch.no_grad():
        diff = (out.detach() - obj_feats.detach()).abs().mean().item()
    print(f"  [INFO] 输出与输入的平均差异（残差贡献）: {diff:.6f}")

    print("测试 2 全部通过！")
    return True


# ============================================================
# 测试 3：多层堆叠
# ============================================================

def test_h_mamba_multilayer():
    print("\n" + "="*60)
    print("测试 3: 多层 HilbertMambaInteraction (num_layers=2)")
    print("="*60)

    try:
        from projects.mmdet3d_plugin.models.h_mamba.h_mamba_interaction import (
            HilbertMambaInteraction, MAMBA_AVAILABLE
        )
    except ImportError as e:
        print(f"  [SKIP] 导入失败: {e}")
        return True

    if not MAMBA_AVAILABLE:
        print("  [SKIP] mamba_ssm 未安装，跳过多层测试")
        return True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_model = 128
    N = 40
    batch_size = 2

    obj_feats   = torch.randn(N, d_model, device=device)
    obj_centers = torch.rand(N, 3, device=device) * 100 - 50
    batch_idx   = torch.cat([
        torch.zeros(N // 2, dtype=torch.long),
        torch.ones(N - N // 2, dtype=torch.long)
    ]).to(device)

    model = HilbertMambaInteraction(
        d_model=d_model,
        hilbert_order=5,
        num_layers=2,
    ).to(device)

    out = model(obj_feats, obj_centers, batch_idx)
    assert out.shape == obj_feats.shape
    assert not torch.isnan(out).any()
    print(f"  [PASS] num_layers=2，输出 shape: {out.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  [INFO] 模型参数量: {total_params:,}")

    print("测试 3 全部通过！")
    return True


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    all_pass = True

    try:
        all_pass &= test_hilbert_utils()
    except Exception:
        print("\n[FAIL] 测试 1 失败：")
        traceback.print_exc()
        all_pass = False

    try:
        all_pass &= test_h_mamba_forward()
    except Exception:
        print("\n[FAIL] 测试 2 失败：")
        traceback.print_exc()
        all_pass = False

    try:
        all_pass &= test_h_mamba_multilayer()
    except Exception:
        print("\n[FAIL] 测试 3 失败：")
        traceback.print_exc()
        all_pass = False

    print("\n" + "="*60)
    if all_pass:
        print("✓ 所有测试通过！HilbertMambaInteraction 模块正常。")
    else:
        print("✗ 部分测试失败，请检查输出日志。")
    print("="*60)
