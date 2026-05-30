import importlib.util
import sys
import unittest
from pathlib import Path

import torch


def load_module(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ScatterV2Fp16IndicesTest(unittest.TestCase):
    def _assert_scatter_accepts_float_cast_indices(self, scatter_v2):
        feat = torch.tensor([[1.0], [3.0], [2.0]], dtype=torch.float32)
        coors = torch.tensor([[0, 0], [0, 0], [1, 0]], dtype=torch.float32)
        new_coors = torch.tensor([[0, 0], [1, 0]], dtype=torch.float32)
        unq_inv = torch.tensor([0, 0, 1], dtype=torch.float32)

        out, out_coors, out_inv = scatter_v2(
            feat,
            coors,
            mode="max",
            unq_inv=unq_inv,
            new_coors=new_coors,
        )

        torch.testing.assert_allclose(out, torch.tensor([[3.0], [2.0]]))
        self.assertEqual(out_inv.dtype, torch.long)
        self.assertEqual(out_coors.dtype, torch.long)

    def test_project_scatter_v2_restores_integer_indices(self):
        repo_root = Path(__file__).resolve().parents[1]
        module = load_module(
            "project_sst_ops_for_test",
            repo_root / "projects/mmdet3d_plugin/ops/sst_ops.py",
        )
        self._assert_scatter_accepts_float_cast_indices(module.scatter_v2)

    def test_mmdet3d_scatter_v2_restores_integer_indices(self):
        mmdet3d_root = "/home/ddd/pc/mmdetection3d"
        if mmdet3d_root not in sys.path:
            sys.path.insert(0, mmdet3d_root)
        from mmdet3d.models.voxel_encoders import voxel_encoder as module

        self._assert_scatter_accepts_float_cast_indices(module.scatter_v2)


if __name__ == "__main__":
    unittest.main()
