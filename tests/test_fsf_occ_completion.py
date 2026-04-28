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


class FSFOccCompletionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def test_completion_descriptor_matches_instance_count(self):
        descriptor = self.module.build_completion_descriptor(
            coarse_scores=torch.tensor([0.1, 0.9, 0.2, 0.8]),
            refine_scores=torch.tensor([0.2, 0.8, 0.3, 0.7]),
            instance_ids=torch.tensor([0, 0, 1, 1]),
            pred_size_residuals=torch.tensor([[1.0, 0.0, 0.5], [0.5, 0.1, 0.2]]),
            pred_visibility=torch.tensor([0.3, 0.7]),
            center_offsets=torch.tensor([[0.0, 0.1, 0.0], [0.2, 0.0, 0.0]]),
        )
        self.assertEqual(descriptor.shape, (2, 8))


if __name__ == "__main__":
    unittest.main()
