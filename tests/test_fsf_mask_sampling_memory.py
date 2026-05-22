import unittest

import torch


class FSFMaskSamplingMemoryTest(unittest.TestCase):
    def test_points_in_mask_does_not_materialize_all_masks_as_float(self):
        from projects.mmdet3d_plugin.models.detectors.FSF import FSF

        detector = FSF.__new__(FSF)
        points = torch.tensor([[1.0, 1.0, 2.0]], dtype=torch.float32)
        mask_data = torch.zeros((1, 2, 4, 4), dtype=torch.int16)
        lidar2img = torch.eye(4, dtype=torch.float32).unsqueeze(0)

        original_float = torch.Tensor.float

        def fail_on_full_float(tensor):
            if tensor is mask_data:
                raise RuntimeError("materialized full mask_data as float")
            return original_float(tensor)

        try:
            torch.Tensor.float = fail_on_full_float
            obj_ids = detector.points_in_mask(points, mask_data, lidar2img)
        finally:
            torch.Tensor.float = original_float

        self.assertEqual(obj_ids.shape, (1, 1, 2))
        self.assertEqual(obj_ids.dtype, torch.long)


if __name__ == "__main__":
    unittest.main()
