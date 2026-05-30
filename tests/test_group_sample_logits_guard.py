import unittest

import torch


class GroupSampleLogitsGuardTest(unittest.TestCase):
    def test_group_sample_accepts_all_positive_raw_logits(self):
        from projects.mmdet3d_plugin.models.detectors.single_stage_fsd import SingleStageFSD

        detector = SingleStageFSD.__new__(SingleStageFSD)
        detector.num_classes = 2
        detector.train_cfg = dict(
            score_thresh=[0.1, 0.1],
            group_names=[["Car"], ["Pedestrian"]],
            class_names=["Car", "Pedestrian"],
            offset_weight="max",
        )
        detector.test_cfg = detector.train_cfg
        detector.cfg = detector.train_cfg
        detector.runtime_info = None
        detector.training = True

        seg_logits = torch.tensor(
            [
                [3.0, 2.0, 1.0],
                [2.0, 3.0, 1.0],
            ]
        )
        offsets = torch.zeros((2, 9), dtype=torch.float32)
        dict_to_sample = dict(
            seg_points=torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                ]
            ),
            seg_logits=seg_logits,
            seg_vote_preds=offsets,
            seg_feats=torch.zeros((2, 4), dtype=torch.float32),
            batch_idx=torch.zeros(2, dtype=torch.long),
            vote_offsets=offsets,
        )

        sampled = detector.group_sample(dict_to_sample, offsets)

        self.assertEqual(len(sampled["center_preds"]), 2)
        self.assertTrue(sampled["fg_mask_list"][0].any())
        self.assertTrue(sampled["fg_mask_list"][1].any())


if __name__ == "__main__":
    unittest.main()
