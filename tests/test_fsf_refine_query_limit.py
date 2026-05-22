import unittest

import torch


class FSFRefineQueryLimitTest(unittest.TestCase):
    def test_limit_refine_queries_keeps_top_scores_per_batch(self):
        from projects.mmdet3d_plugin.models.detectors.FSF import FSF

        detector = FSF.__new__(FSF)
        detector.max_refine_queries = 2

        obj_centers = torch.arange(18, dtype=torch.float32).view(6, 3)
        obj_coors = torch.tensor(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 0, 2],
                [1, 0, 0],
                [1, 0, 1],
                [1, 0, 2],
            ],
            dtype=torch.long,
        )
        obj_feats = torch.arange(24, dtype=torch.float32).view(6, 4)
        preds_2d = torch.arange(12, dtype=torch.float32).view(6, 2)
        obj_result = dict(
            cls_logits=[
                torch.tensor([[0.1], [3.0], [1.0]], dtype=torch.float32),
                torch.tensor([[2.0], [0.2], [4.0]], dtype=torch.float32),
            ],
            reg_preds=[
                torch.arange(12, dtype=torch.float32).view(3, 4),
                torch.arange(12, 24, dtype=torch.float32).view(3, 4),
            ],
        )

        centers, coors, result, feats, kept_preds = detector.limit_refine_queries(
            obj_centers,
            obj_coors,
            obj_result,
            obj_feats,
            preds_2d,
        )

        self.assertTrue(torch.equal(coors[:, 0], torch.tensor([0, 0, 1, 1])))
        self.assertTrue(torch.equal(coors[:, 2], torch.tensor([1, 2, 0, 2])))
        self.assertTrue(torch.equal(centers, obj_centers[[1, 2, 3, 5]]))
        self.assertTrue(torch.equal(feats, obj_feats[[1, 2, 3, 5]]))
        self.assertTrue(torch.equal(kept_preds, preds_2d[[1, 2, 3, 5]]))
        self.assertTrue(torch.equal(result["cls_logits"][0], obj_result["cls_logits"][0][[1, 2]]))
        self.assertTrue(torch.equal(result["cls_logits"][1], obj_result["cls_logits"][1][[0, 2]]))
        self.assertTrue(torch.equal(result["reg_preds"][0], obj_result["reg_preds"][0][[1, 2]]))
        self.assertTrue(torch.equal(result["reg_preds"][1], obj_result["reg_preds"][1][[0, 2]]))


if __name__ == "__main__":
    unittest.main()
