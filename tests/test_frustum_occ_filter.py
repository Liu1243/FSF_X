import importlib.util
import pathlib
import unittest

import torch


MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "projects"
    / "mmdet3d_plugin"
    / "models"
    / "utils"
    / "frustum_occ_filter.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("frustum_occ_filter", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FrustumOccFilterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def test_compute_losses_returns_multitask_keys(self):
        filt = self.module.FrustumOccFilter(
            occ_mlp_cfg=dict(in_channels=4, hidden_dims=[8]),
            refine_mlp_cfg=dict(in_channels=5, hidden_dims=[8]),
            completion_head_cfg=dict(in_channels=4, hidden_dims=[8]),
        )
        losses = filt.compute_losses(
            coarse_occ_prob=torch.full((4,), 0.5),
            refine_occ_prob=torch.full((4,), 0.5),
            occ_gt=torch.tensor([1.0, 0.0, 1.0, 0.0]),
            pred_offsets=torch.zeros(2, 3),
            center_gt=torch.zeros(2, 3),
            valid_center_mask=torch.tensor([True, False]),
            pred_size_residuals=torch.zeros(2, 3),
            size_gt=torch.zeros(2, 3),
            pred_visibility=torch.zeros(2),
            visibility_gt=torch.zeros(2),
            valid_completion_mask=torch.tensor([True, False]),
        )
        self.assertIn("loss_occ_coarse", losses)
        self.assertIn("loss_occ_refine", losses)
        self.assertIn("loss_amodal_center", losses)
        self.assertIn("loss_amodal_size", losses)
        self.assertIn("loss_visibility", losses)

    def test_instance_fallback_keeps_one_point_per_instance(self):
        keep = self.module.ensure_minimum_points_per_instance(
            instance_ids=torch.tensor([0, 0, 1, 1]),
            scores=torch.tensor([0.1, 0.2, 0.3, 0.4]),
            mask=torch.tensor([False, False, False, True]),
        )
        self.assertTrue(torch.equal(keep, torch.tensor([False, True, False, True])))

    def test_instance_fallback_can_keep_topk_points_per_instance(self):
        keep = self.module.ensure_minimum_points_per_instance(
            instance_ids=torch.tensor([0, 0, 0, 1, 1, 1]),
            scores=torch.tensor([0.1, 0.5, 0.3, 0.6, 0.2, 0.4]),
            mask=torch.tensor([False, False, False, True, False, False]),
            min_points=2,
        )
        self.assertTrue(torch.equal(keep, torch.tensor([False, True, True, True, False, True])))

    def test_filter_uses_configured_min_points_per_instance(self):
        filt = self.module.FrustumOccFilter(
            occ_mlp_cfg=dict(in_channels=4, hidden_dims=[8]),
            refine_mlp_cfg=dict(in_channels=5, hidden_dims=[8]),
            completion_head_cfg=dict(in_channels=4, hidden_dims=[8]),
            min_points_per_instance=2,
        )
        self.assertEqual(filt.min_points_per_instance, 2)

    def test_instance_fallback_uses_class_specific_min_points(self):
        keep = self.module.ensure_minimum_points_per_instance(
            instance_ids=torch.tensor([0, 0, 0, 1, 1, 1]),
            scores=torch.tensor([0.1, 0.5, 0.3, 0.6, 0.2, 0.4]),
            mask=torch.tensor([False, False, False, False, False, False]),
            min_points=1,
            class_ids=torch.tensor([5, 5, 5, 0, 0, 0]),
            class_min_points={5: 2},
        )
        self.assertTrue(torch.equal(keep, torch.tensor([False, True, True, True, False, False])))

    def test_bev_orientation_cues_follow_instance_major_axis(self):
        cues = self.module.compute_bev_orientation_cues(
            points=torch.tensor([
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [1.0, -1.0, 0.0],
                [1.0, 1.0, 0.0],
            ]),
            instance_ids=torch.tensor([0, 0, 1, 1]),
            valid_mask=torch.tensor([True, True, True, True]),
        )
        self.assertTrue(torch.allclose(cues[0], torch.tensor([1.0, 0.0]), atol=1e-4))
        self.assertTrue(torch.allclose(cues[1], torch.tensor([0.0, 1.0]), atol=1e-4))

    def test_forward_returns_completion_outputs(self):
        filt = self.module.FrustumOccFilter(
            occ_mlp_cfg=dict(in_channels=4, hidden_dims=[8]),
            refine_mlp_cfg=dict(in_channels=5, hidden_dims=[8]),
            completion_head_cfg=dict(in_channels=4, hidden_dims=[8]),
        )
        outputs = filt.forward(
            pts_feat=torch.randn(4, 4),
            points=torch.randn(4, 3),
            sir_coors=torch.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 0, 1]]),
            obs_centers=torch.randn(2, 3),
            batch_idx=torch.tensor([0, 0, 0, 0]),
            gt_bboxes_3d_list=None,
        )
        self.assertEqual(len(outputs), 5)
        self.assertEqual(outputs[3]["descriptor"].shape, (2, 10))


if __name__ == "__main__":
    unittest.main()
