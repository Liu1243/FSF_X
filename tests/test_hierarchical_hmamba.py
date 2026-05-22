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
    / "hierarchical_hmamba.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("hierarchical_hmamba", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HierarchicalHMambaTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def test_hilbert_serialization_preserves_batch_membership(self):
        centers = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.2, 0.3],
                [5.0, 5.0, 0.0],
                [5.1, 5.2, 0.2],
            ],
            dtype=torch.float32,
        )
        batch_ids = torch.tensor([0, 0, 1, 1], dtype=torch.long)

        serial = self.module.serialize_hilbert(centers, batch_ids, num_rotations=2)

        self.assertEqual(serial["orders"].shape, (2, 4))
        self.assertEqual(serial["inverse"].shape, (2, 4))

        for order in serial["orders"]:
            ordered_batches = batch_ids.index_select(0, order)
            self.assertTrue(torch.equal(ordered_batches[:2], torch.tensor([0, 0])))
            self.assertTrue(torch.equal(ordered_batches[2:], torch.tensor([1, 1])))

    def test_foreground_selector_keeps_high_score_tokens_per_batch(self):
        selector = self.module.ForegroundTokenSelector(
            d_model=8,
            keep_ratio=0.5,
            min_tokens=1,
            min_per_modality=1,
        )
        features = torch.randn(6, 8)
        batch_ids = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
        modality_ids = torch.tensor([0, 0, 1, 0, 1, 1], dtype=torch.long)
        cls_logits = torch.tensor(
            [
                [8.0, -8.0],
                [6.0, -6.0],
                [1.0, -1.0],
                [7.0, -7.0],
                [5.0, -5.0],
                [0.5, -0.5],
            ],
            dtype=torch.float32,
        )
        preds_2d = torch.tensor(
            [
                [0.95, 0.0, 0.0],
                [0.80, 0.0, 0.0],
                [0.10, 0.0, 0.0],
                [0.90, 0.0, 0.0],
                [0.75, 0.0, 0.0],
                [0.15, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )

        output = selector(features, batch_ids, modality_ids, cls_logits, preds_2d)

        self.assertEqual(output["mask"].dtype, torch.bool)
        self.assertEqual(output["mask"].shape, (6,))
        self.assertGreaterEqual(int(output["mask"][:3].sum().item()), 2)
        self.assertGreaterEqual(int(output["mask"][3:].sum().item()), 2)
        self.assertTrue(output["mask"][0].item())
        self.assertTrue(output["mask"][3].item())

    def test_foreground_selector_preserves_configured_class_group(self):
        selector = self.module.ForegroundTokenSelector(
            d_model=8,
            keep_ratio=0.25,
            min_tokens=1,
            min_per_modality=0,
            class_groups=[[2]],
            class_group_min_tokens=[1],
        )
        features = torch.zeros(4, 8)
        batch_ids = torch.zeros(4, dtype=torch.long)
        modality_ids = torch.zeros(4, dtype=torch.long)
        cls_logits = torch.tensor(
            [
                [8.0, -8.0, -8.0],
                [7.0, -7.0, -7.0],
                [-6.0, -6.0, -6.0],
                [-6.0, -6.0, 0.5],
            ],
            dtype=torch.float32,
        )

        output = selector(features, batch_ids, modality_ids, cls_logits=cls_logits)

        self.assertTrue(output["mask"][3].item())
        self.assertEqual(int(output["mask"].sum().item()), 1)

    def test_class_group_quota_counts_tokens_kept_by_modality_floor(self):
        selector = self.module.ForegroundTokenSelector(
            d_model=8,
            keep_ratio=0.5,
            min_tokens=1,
            min_per_modality=1,
            class_groups=[[2]],
            class_group_min_tokens=[1],
        )
        features = torch.zeros(4, 8)
        batch_ids = torch.zeros(4, dtype=torch.long)
        modality_ids = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        cls_logits = torch.tensor(
            [
                [8.0, -8.0, -8.0],
                [7.0, -7.0, -7.0],
                [-6.0, -6.0, -6.0],
                [-6.0, -6.0, 0.5],
            ],
            dtype=torch.float32,
        )

        output = selector(features, batch_ids, modality_ids, cls_logits=cls_logits)

        self.assertTrue(output["mask"][0].item())
        self.assertTrue(output["mask"][3].item())
        self.assertEqual(int(output["mask"].sum().item()), 2)

    def test_reliability_gate_outputs_valid_range(self):
        gate = self.module.ReliabilityGatedFusion(d_model=8, descriptor_dim=6)
        residual = torch.randn(4, 8)
        interacted = torch.randn(4, 8)
        descriptors = torch.randn(4, 6)

        fused, aux = gate(residual, interacted, descriptors)

        self.assertEqual(fused.shape, residual.shape)
        self.assertEqual(aux["gate"].shape, (4, 1))
        self.assertEqual(aux["log_var"].shape, (4, 1))
        self.assertTrue(torch.all(aux["gate"] >= 0.0).item())
        self.assertTrue(torch.all(aux["gate"] <= 1.0).item())

    def test_extended_reliability_descriptors_include_geometry_cues(self):
        model = self.module.HierarchicalHMambaInteraction(
            d_model=8,
            d_state=4,
            expand_factor=1,
            use_fast_path=False,
            use_extended_reliability=True,
        )
        centers = torch.tensor(
            [
                [3.0, 4.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        modality_ids = torch.tensor([0, 1], dtype=torch.long)
        cls_logits = torch.tensor(
            [
                [4.0, -4.0, -4.0],
                [-2.0, -2.0, 2.0],
            ],
            dtype=torch.float32,
        )
        preds_2d = torch.tensor(
            [
                [0.0, 0.0, 20.0, 10.0, 0.80, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 5.0, 20.0, 0.25, 0.0, 0.0, 0.0, 0.5, 0.0],
            ],
            dtype=torch.float32,
        )

        descriptors = model._reliability_descriptors(
            centers,
            modality_ids,
            cls_logits=cls_logits,
            preds_2d=preds_2d,
        )

        self.assertEqual(descriptors.shape, (2, 9))
        self.assertAlmostEqual(float(descriptors[0, 0].item()), 1.0, places=5)
        self.assertAlmostEqual(float(descriptors[1, 0].item()), 0.0, places=5)
        self.assertAlmostEqual(float(descriptors[0, 4].item()), 0.80, places=5)
        self.assertAlmostEqual(float(descriptors[0, 5].item()), 1.0, places=5)
        self.assertTrue(torch.all(descriptors[:, 6:] >= 0.0).item())
        self.assertTrue(torch.all(descriptors[:, 6:] <= 1.0).item())

    def test_hierarchical_hmamba_passes_class_groups_to_selector(self):
        model = self.module.HierarchicalHMambaInteraction(
            d_model=8,
            d_state=4,
            expand_factor=1,
            use_fast_path=False,
            class_groups=[[2, 4], [5, 6, 8, 9]],
            class_group_min_tokens=[1, 2],
        )

        self.assertEqual(model.selector.class_groups, [(2, 4), (5, 6, 8, 9)])
        self.assertEqual(model.selector.class_group_min_tokens, [1, 2])

    def test_hierarchical_hmamba_restores_original_token_order(self):
        model = self.module.HierarchicalHMambaInteraction(
            d_model=16,
            d_state=8,
            expand_factor=1,
            keep_ratio=0.75,
            num_rotations=2,
            window_size=2,
            use_fast_path=False,
        )
        features = torch.randn(8, 16)
        centers = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.3, 0.0, 0.0],
                [5.0, 5.0, 0.0],
                [5.1, 5.0, 0.0],
                [5.2, 5.0, 0.0],
                [5.3, 5.0, 0.0],
            ],
            dtype=torch.float32,
        )
        batch_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
        modality_ids = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long)
        cls_logits = torch.randn(8, 3)
        preds_2d = torch.randn(8, 3)

        fused, aux = model(
            features,
            centers,
            batch_ids,
            modality_ids,
            cls_logits=cls_logits,
            preds_2d=preds_2d,
        )

        self.assertEqual(fused.shape, features.shape)
        self.assertTrue(torch.equal(aux["restore_indices"].sort().values, torch.arange(8)))
        self.assertEqual(aux["selected_mask"].shape, (8,))
        self.assertEqual(aux["scores"].shape, (8,))

    def test_hierarchical_hmamba_aux_outputs_follow_feature_dtype(self):
        class HalfAuxFuser(torch.nn.Module):
            def forward(self, residual, interacted, descriptors):
                gate = residual.new_full((residual.size(0), 1), 0.5).half()
                log_var = residual.new_zeros((residual.size(0), 1)).half()
                return residual, {"gate": gate, "log_var": log_var}

        model = self.module.HierarchicalHMambaInteraction(
            d_model=8,
            d_state=4,
            expand_factor=1,
            keep_ratio=1.0,
            min_tokens=1,
            num_rotations=1,
            window_size=2,
            use_fast_path=False,
        )
        model.reliability_fuser = HalfAuxFuser()
        features = torch.randn(4, 8, dtype=torch.float32)
        centers = torch.randn(4, 3, dtype=torch.float32)
        batch_ids = torch.zeros(4, dtype=torch.long)
        modality_ids = torch.tensor([0, 1, 0, 1], dtype=torch.long)
        cls_logits = torch.randn(4, 3, dtype=torch.float32)

        fused, aux = model(
            features,
            centers,
            batch_ids,
            modality_ids,
            cls_logits=cls_logits,
        )

        self.assertEqual(fused.dtype, features.dtype)
        self.assertEqual(aux["gate"].dtype, features.dtype)
        self.assertEqual(aux["log_var"].dtype, features.dtype)

    def test_small_object_residual_head_reports_configured_tokens(self):
        model = self.module.HierarchicalHMambaInteraction(
            d_model=8,
            d_state=4,
            expand_factor=1,
            keep_ratio=1.0,
            min_tokens=1,
            num_rotations=1,
            window_size=2,
            use_fast_path=False,
            small_object_class_indices=[2],
        )
        features = torch.randn(4, 8)
        centers = torch.randn(4, 3)
        batch_ids = torch.zeros(4, dtype=torch.long)
        modality_ids = torch.tensor([0, 1, 0, 1], dtype=torch.long)
        cls_logits = torch.tensor(
            [
                [5.0, -5.0, -5.0],
                [-5.0, -5.0, 5.0],
                [4.0, -4.0, -4.0],
                [-4.0, -4.0, 4.0],
            ],
            dtype=torch.float32,
        )

        fused, aux = model(
            features,
            centers,
            batch_ids,
            modality_ids,
            cls_logits=cls_logits,
        )

        self.assertEqual(fused.shape, features.shape)
        self.assertIn("small_object_mask", aux)
        self.assertIn("small_object_pose_residual", aux)
        self.assertIn("small_object_scale_residual", aux)
        self.assertIn("small_object_temperature", aux)
        self.assertTrue(torch.equal(aux["small_object_mask"], torch.tensor([False, True, False, True])))
        self.assertEqual(aux["small_object_pose_residual"].shape, (4, 2))
        self.assertEqual(aux["small_object_scale_residual"].shape, (4, 3))
        self.assertEqual(aux["small_object_temperature"].shape, (4, 1))


if __name__ == "__main__":
    unittest.main()
