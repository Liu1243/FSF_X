import importlib.util
import pathlib
import sys
import types
import unittest


MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "projects"
    / "mmdet3d_plugin"
    / "models"
    / "detectors"
    / "FSF_HierMamba.py"
)


class DetectorPassThroughTests(unittest.TestCase):

    def setUp(self):
        self._old_modules = {}
        for name in [
            "mmdet",
            "mmdet.models",
            "projects",
            "projects.mmdet3d_plugin",
            "projects.mmdet3d_plugin.models",
            "projects.mmdet3d_plugin.models.detectors",
            "projects.mmdet3d_plugin.models.detectors.FSF",
            "projects.mmdet3d_plugin.models.utils",
            "projects.mmdet3d_plugin.models.utils.hierarchical_hmamba",
        ]:
            self._old_modules[name] = sys.modules.get(name)

        class Registry:
            def register_module(self):
                def decorator(cls):
                    return cls
                return decorator

        mmdet_mod = types.ModuleType("mmdet")
        mmdet_models_mod = types.ModuleType("mmdet.models")
        mmdet_models_mod.DETECTORS = Registry()

        projects_mod = types.ModuleType("projects")
        plugin_mod = types.ModuleType("projects.mmdet3d_plugin")
        models_mod = types.ModuleType("projects.mmdet3d_plugin.models")
        detectors_mod = types.ModuleType("projects.mmdet3d_plugin.models.detectors")
        utils_mod = types.ModuleType("projects.mmdet3d_plugin.models.utils")

        fsf_mod = types.ModuleType("projects.mmdet3d_plugin.models.detectors.FSF")

        class FSF:
            def __init__(self, **kwargs):
                self.embed_dims = kwargs.get("embed_dims", 1024)

        fsf_mod.FSF = FSF

        hmamba_mod = types.ModuleType("projects.mmdet3d_plugin.models.utils.hierarchical_hmamba")

        class HierarchicalHMambaInteraction:
            last_kwargs = None

            def __init__(self, **kwargs):
                HierarchicalHMambaInteraction.last_kwargs = kwargs

        hmamba_mod.HierarchicalHMambaInteraction = HierarchicalHMambaInteraction
        self.recorder = HierarchicalHMambaInteraction

        sys.modules.update(
            {
                "mmdet": mmdet_mod,
                "mmdet.models": mmdet_models_mod,
                "projects": projects_mod,
                "projects.mmdet3d_plugin": plugin_mod,
                "projects.mmdet3d_plugin.models": models_mod,
                "projects.mmdet3d_plugin.models.detectors": detectors_mod,
                "projects.mmdet3d_plugin.models.detectors.FSF": fsf_mod,
                "projects.mmdet3d_plugin.models.utils": utils_mod,
                "projects.mmdet3d_plugin.models.utils.hierarchical_hmamba": hmamba_mod,
            }
        )

    def tearDown(self):
        for name, old_module in self._old_modules.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module

    def load_detector_module(self):
        spec = importlib.util.spec_from_file_location(
            "projects.mmdet3d_plugin.models.detectors.FSF_HierMamba",
            MODULE_PATH,
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_hiermamba_config_passes_optimization_options(self):
        module = self.load_detector_module()

        module.FSF_HierMamba(
            hiermamba_cfg=dict(
                d_model=64,
                class_groups=[[2, 4], [5, 6, 8, 9]],
                class_group_min_tokens=[1, 2],
                use_extended_reliability=True,
                small_object_class_indices=[2, 4, 5, 6, 8, 9],
                small_object_residual_scale=0.2,
            )
        )

        kwargs = self.recorder.last_kwargs
        self.assertEqual(kwargs["class_groups"], [[2, 4], [5, 6, 8, 9]])
        self.assertEqual(kwargs["class_group_min_tokens"], [1, 2])
        self.assertTrue(kwargs["use_extended_reliability"])
        self.assertEqual(kwargs["small_object_class_indices"], [2, 4, 5, 6, 8, 9])
        self.assertEqual(kwargs["small_object_residual_scale"], 0.2)


if __name__ == "__main__":
    unittest.main()
