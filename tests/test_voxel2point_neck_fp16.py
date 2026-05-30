import importlib
import unittest

import torch


def load_voxel2point_neck():
    module = importlib.import_module("projects.mmdet3d_plugin.models.necks.voxel2point_neck")
    return module.Voxel2PointScatterNeck


class Voxel2PointScatterNeckFp16Test(unittest.TestCase):
    def test_fp16_voxel_features_keep_coordinate_math_in_float32(self):
        neck_cls = load_voxel2point_neck()
        neck = neck_cls(
            point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            voxel_size=[0.2, 0.2, 0.2],
            with_xyz=True,
            normalize_local_xyz=False,
        )
        neck.train()

        points = torch.tensor([[48.999, -51.1, -4.9, 0.0, 0.0]], dtype=torch.float32)
        pts_coors = torch.tensor([[0, 0, 0, 500]], dtype=torch.long)
        voxel_feats = torch.ones((1, 4), dtype=torch.float16)
        voxel2point_inds = torch.tensor([0], dtype=torch.long)

        results, pts_mask = neck(points, pts_coors, voxel_feats, voxel2point_inds)

        self.assertTrue(pts_mask.item())
        self.assertEqual(results.dtype, voxel_feats.dtype)
        self.assertLess(abs(float(results[0, -3])), 0.101)


if __name__ == "__main__":
    unittest.main()
