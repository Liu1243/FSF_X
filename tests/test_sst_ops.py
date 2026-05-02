import importlib.util
import pathlib
import sys
import types
import unittest

import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "projects" / "mmdet3d_plugin" / "ops" / "sst_ops.py"


def install_stub(name, module):
    old = sys.modules.get(name)
    sys.modules[name] = module
    return old


def load_module():
    installed = {}
    if "mmdet3d" not in sys.modules:
        mmdet3d = types.ModuleType("mmdet3d")
        ops = types.ModuleType("mmdet3d.ops")
        ops.spconv = types.SimpleNamespace()
        installed["mmdet3d"] = install_stub("mmdet3d", mmdet3d)
        installed["mmdet3d.ops"] = install_stub("mmdet3d.ops", ops)
    if "mmcv" not in sys.modules:
        mmcv = types.ModuleType("mmcv")
        cnn = types.ModuleType("mmcv.cnn")
        cnn.build_norm_layer = lambda *args, **kwargs: None
        installed["mmcv"] = install_stub("mmcv", mmcv)
        installed["mmcv.cnn"] = install_stub("mmcv.cnn", cnn)
    if "torch_scatter" not in sys.modules and importlib.util.find_spec("torch_scatter") is None:
        scatter = types.ModuleType("torch_scatter")
        scatter.scatter = None
        scatter.scatter_max = None
        installed["torch_scatter"] = install_stub("torch_scatter", scatter)
    if "ingroup_indices" not in sys.modules:
        ingroup_indices = types.ModuleType("ingroup_indices")
        ingroup_indices.forward = lambda group_inds, out_inds: out_inds.zero_()
        installed["ingroup_indices"] = install_stub("ingroup_indices", ingroup_indices)

    spec = importlib.util.spec_from_file_location("sst_ops", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for name, old in installed.items():
        if old is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = old
    return module


class ScatterV2Tests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def test_scatter_v2_chunks_wide_feature_mean(self):
        feat = torch.arange(20, dtype=torch.float32).view(4, 5)
        coors = torch.tensor(
            [
                [0, 0],
                [0, 0],
                [0, 1],
                [0, 1],
            ],
            dtype=torch.long,
        )
        calls = []

        def guarded_scatter(src, index, dim=0, out=None, dim_size=None, reduce=None):
            self.assertEqual(dim, 0)
            self.assertEqual(dim_size, 2)
            self.assertLessEqual(src.size(1), 2)
            calls.append(src.size(1))
            output = src.new_zeros((dim_size, src.size(1)))
            output.index_add_(0, index, src)
            if reduce == "mean":
                counts = src.new_zeros((dim_size, 1))
                counts.index_add_(0, index, torch.ones((src.size(0), 1), dtype=src.dtype))
                output = output / counts.clamp(min=1.0)
            return output

        old_scatter = self.module.torch_scatter.scatter
        self.module.torch_scatter.scatter = guarded_scatter
        try:
            out, new_coors = self.module.scatter_v2(
                feat,
                coors,
                mode="avg",
                return_inv=False,
                feature_chunk_size=2,
            )
        finally:
            self.module.torch_scatter.scatter = old_scatter

        expected = torch.stack([feat[:2].mean(dim=0), feat[2:].mean(dim=0)], dim=0)
        self.assertTrue(torch.equal(new_coors, torch.tensor([[0, 0], [0, 1]], dtype=torch.long)))
        self.assertTrue(torch.allclose(out, expected))
        self.assertEqual(calls, [2, 2, 1])


if __name__ == "__main__":
    unittest.main()
