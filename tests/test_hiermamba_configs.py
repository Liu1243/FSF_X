import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_config(path):
    spec = importlib.util.spec_from_file_location("config_under_test", ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HierMambaConfigTests(unittest.TestCase):

    def test_nuscenes_mini_quotas_only_target_present_difficult_classes(self):
        cfg = load_config("projects/configs/nuScenes/FSF_HierMamba_nuScenes_mini_config.py")
        hier_cfg = cfg.model["hiermamba_cfg"]

        configured = set()
        for group in hier_cfg["class_groups"]:
            configured.update(group)

        self.assertEqual(configured, {5, 6, 8})
        self.assertEqual(set(hier_cfg["small_object_class_indices"]), {5, 6, 8})


if __name__ == "__main__":
    unittest.main()
