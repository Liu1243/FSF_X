import unittest

import torch


class FSFDecodeStageBboxesDtypeTest(unittest.TestCase):
    def test_decode_stage_bboxes_returns_float_rois_for_pooling_extension(self):
        from projects.mmdet3d_plugin.models.detectors.FSF import FSF

        class FloatBBoxCoder:
            def decode(self, reg_pred, centers):
                return torch.ones((centers.size(0), reg_pred.size(-1) - 1), dtype=torch.float32)

        detector = FSF.__new__(FSF)
        detector.bbox_coder = FloatBBoxCoder()
        obj_centers = torch.zeros((3, 3), dtype=torch.float16)
        bz_coors = torch.tensor([0, 0, 1], dtype=torch.long)
        reg_preds = [
            torch.zeros((2, 11), dtype=torch.float16),
            torch.zeros((1, 11), dtype=torch.float16),
        ]

        decoded = detector.decode_stage_bboxes(obj_centers, bz_coors, reg_preds)

        self.assertEqual(decoded.shape, (3, 11))
        self.assertEqual(decoded.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
