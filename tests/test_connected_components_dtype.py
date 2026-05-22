import importlib
import unittest

import torch


class ConnectedComponentsDtypeTest(unittest.TestCase):
    def test_connected_components_matches_batch_index_dtype(self):
        module = importlib.import_module(
            "projects.mmdet3d_plugin.models.detectors.single_stage_fsd"
        )
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.05, 0.0, 0.0],
                [10.0, 10.0, 0.0],
            ],
            dtype=torch.float32,
        )
        batch_idx = torch.tensor([0, 0, 0], dtype=torch.long)

        cluster_inds = module.find_connected_componets(points, batch_idx, dist=0.2)

        self.assertEqual(cluster_inds.dtype, batch_idx.dtype)
        self.assertEqual(cluster_inds.tolist(), [0, 0, 1])


if __name__ == "__main__":
    unittest.main()
