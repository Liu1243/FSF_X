import importlib.util
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


def load_hmamba_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "projects/mmdet3d_plugin/models/utils/h_mamba.py"
    spec = importlib.util.spec_from_file_location("h_mamba_scan_memory_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def materialized_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False):
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)

    A = A.float()
    B = B.float()
    C = C.float()
    delta_a = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    delta_b_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)

    state = u.new_zeros((u.size(0), u.size(1), A.size(1)))
    ys = []
    for i in range(u.size(2)):
        state = delta_a[:, :, i] * state + delta_b_u[:, :, i]
        y = torch.einsum("bdn,bn->bd", state, C[:, :, i])
        if D is not None:
            y = y + u[:, :, i] * D.float()[None, :]
        ys.append(y)
    return torch.stack(ys, dim=2)


class HMambaScanMemoryTest(unittest.TestCase):
    def test_ref_scan_avoids_materialized_state_sequence(self):
        module = load_hmamba_module()
        torch.manual_seed(0)
        u = torch.randn(2, 3, 5)
        delta = torch.randn(2, 3, 5)
        A = -torch.rand(3, 4)
        B = torch.randn(2, 4, 5)
        C = torch.randn(2, 4, 5)
        D = torch.randn(3)
        delta_bias = torch.randn(3)
        expected = materialized_scan(
            u,
            delta,
            A,
            B,
            C,
            D=D,
            delta_bias=delta_bias,
            delta_softplus=True,
        )

        original_einsum = torch.einsum

        def guarded_einsum(equation, *operands):
            if "bdln" in equation:
                raise AssertionError("selective_scan_ref must not materialize [B,d,L,N]")
            return original_einsum(equation, *operands)

        torch.einsum = guarded_einsum
        try:
            actual, last_state = module.selective_scan_ref(
                u,
                delta,
                A,
                B,
                C,
                D=D,
                delta_bias=delta_bias,
                delta_softplus=True,
                return_last_state=True,
            )
        finally:
            torch.einsum = original_einsum

        self.assertEqual(last_state.shape, (2, 3, 4))
        torch.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
