import unittest

import torch


class Fp16DepthClampTest(unittest.TestCase):
    def test_half_depth_clamp_does_not_overflow(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required to reproduce fp16 scalar overflow")

        depth = torch.tensor([1.0], dtype=torch.float16, device="cuda")

        try:
            torch.clamp(depth, min=1e-5, max=1e5)
        except RuntimeError as exc:
            self.assertIn("without overflow", str(exc))
        else:
            self.fail("baseline clamp should overflow with max=1e5 for float16")

        max_depth = torch.finfo(depth.dtype).max
        clamped = torch.clamp(depth, min=1e-5, max=max_depth)

        self.assertEqual(clamped.dtype, torch.float16)
        self.assertEqual(float(clamped.item()), 1.0)


if __name__ == "__main__":
    unittest.main()
