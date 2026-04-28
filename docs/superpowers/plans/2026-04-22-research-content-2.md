# Research Content 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade `FSF_Occ` from a point-filtering prototype into a sparse occupancy-guided instance completion module with coarse-to-fine occupancy, amodal completion targets, and detector feature reuse.

**Architecture:** Keep the existing FSF frustum pipeline intact and concentrate the new logic inside `FrustumOccFilter` plus a narrow integration change in `FSF_Occ`. The detector should remain fully sparse: all new reasoning happens on frustum points and instance-pooled sparse features, then feeds compact completion descriptors back into the frustum detection head.

**Tech Stack:** Python, PyTorch, MMDetection3D-style modules, unit tests via `pytest`, validation in WSL with `conda activate FSF`.

---

### Task 1: Add failing unit tests for `FrustumOccFilter`

**Files:**
- Create: `Y:/home/ddd/pc/FullySparseFusion/tests/test_frustum_occ_filter.py`
- Test: `Y:/home/ddd/pc/FullySparseFusion/tests/test_frustum_occ_filter.py`

- [ ] **Step 1: Write the failing test**

```python
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


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_frustum_occ_filter.py -v`
Expected: FAIL because `refine_mlp_cfg`, `completion_head_cfg`, new loss keys, and `ensure_minimum_points_per_instance` do not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
def ensure_minimum_points_per_instance(instance_ids, scores, mask):
    kept = mask.clone()
    for instance_id in instance_ids.unique():
        instance_mask = instance_ids == instance_id
        if kept[instance_mask].any():
            continue
        local_scores = scores[instance_mask]
        best_local = local_scores.argmax()
        global_indices = instance_mask.nonzero(as_tuple=False).squeeze(-1)
        kept[global_indices[best_local]] = True
    return kept
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_frustum_occ_filter.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_frustum_occ_filter.py projects/mmdet3d_plugin/models/utils/frustum_occ_filter.py
git commit -m "test: cover frustum occupancy multitask behavior"
```

### Task 2: Add failing integration tests for `FSF_Occ` completion descriptors

**Files:**
- Create: `Y:/home/ddd/pc/FullySparseFusion/tests/test_fsf_occ_completion.py`
- Test: `Y:/home/ddd/pc/FullySparseFusion/tests/test_fsf_occ_completion.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fsf_occ_completion.py -v`
Expected: FAIL because `build_completion_descriptor` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
def build_completion_descriptor(
    coarse_scores,
    refine_scores,
    instance_ids,
    pred_size_residuals,
    pred_visibility,
    center_offsets,
):
    rows = []
    for instance_id in instance_ids.unique():
        mask = instance_ids == instance_id
        rows.append(torch.cat([
            coarse_scores[mask].mean().reshape(1),
            coarse_scores[mask].max().reshape(1),
            refine_scores[mask].mean().reshape(1),
            refine_scores[mask].max().reshape(1),
            pred_size_residuals[int(instance_id)],
            pred_visibility[int(instance_id)].reshape(1),
            center_offsets[int(instance_id)].norm().reshape(1),
        ]))
    return torch.stack(rows, dim=0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_fsf_occ_completion.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_fsf_occ_completion.py projects/mmdet3d_plugin/models/utils/frustum_occ_filter.py
git commit -m "test: cover fsf occupancy completion descriptor"
```

### Task 3: Implement coarse-to-fine occupancy and completion heads

**Files:**
- Modify: `Y:/home/ddd/pc/FullySparseFusion/projects/mmdet3d_plugin/models/utils/frustum_occ_filter.py`
- Test: `Y:/home/ddd/pc/FullySparseFusion/tests/test_frustum_occ_filter.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_frustum_occ_filter.py::FrustumOccFilterTests::test_forward_returns_completion_outputs -v`
Expected: FAIL because `forward` still returns only three outputs.

- [ ] **Step 3: Write minimal implementation**

```python
coarse_occ_prob = self.occ_mlp(pts_feat)
refine_input = torch.cat([pts_feat, coarse_occ_prob.unsqueeze(-1)], dim=-1)
refine_occ_prob = self.refine_mlp(refine_input)
valid_mask = ensure_minimum_points_per_instance(
    instance_ids=sir_coors[:, -1],
    scores=refine_occ_prob,
    mask=refine_occ_prob > self.occ_thr,
)
pred_offsets, pred_sizes, pred_visibility, agg_coors = self.completion_head(
    pts_feat, sir_coors, valid_mask
)
return valid_mask, obs_centers + pred_offsets, occ_losses, completion_outputs, refine_occ_prob
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_frustum_occ_filter.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add projects/mmdet3d_plugin/models/utils/frustum_occ_filter.py tests/test_frustum_occ_filter.py
git commit -m "feat: add coarse-to-fine frustum occupancy completion"
```

### Task 4: Integrate completion descriptors into `FSF_Occ`

**Files:**
- Modify: `Y:/home/ddd/pc/FullySparseFusion/projects/mmdet3d_plugin/models/detectors/FSF_Occ.py`
- Modify: `Y:/home/ddd/pc/FullySparseFusion/projects/configs/nuScenes/FSF_Occ_nuScenes_mini_config.py`
- Test: `Y:/home/ddd/pc/FullySparseFusion/tests/test_fsf_occ_completion.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fsf_occ_completion.py -v`
Expected: FAIL until `FSF_Occ` consumes the new descriptor shape and config exposes matching `in_channel`.

- [ ] **Step 3: Write minimal implementation**

```python
completion_descriptor = build_completion_descriptor(
    coarse_scores=coarse_occ_prob,
    refine_scores=refine_occ_prob,
    instance_ids=sir_coors[:, -1],
    pred_size_residuals=completion_outputs["size_residuals"],
    pred_visibility=completion_outputs["visibility"],
    center_offsets=completion_outputs["center_offsets"],
)
obj_feat = torch.cat([lidar_img_feat, completion_descriptor], dim=-1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_fsf_occ_completion.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add projects/mmdet3d_plugin/models/detectors/FSF_Occ.py projects/configs/nuScenes/FSF_Occ_nuScenes_mini_config.py tests/test_fsf_occ_completion.py
git commit -m "feat: reuse occupancy completion descriptors in fsf occ"
```

### Task 5: Validate in WSL `FSF` environment

**Files:**
- Test: `Y:/home/ddd/pc/FullySparseFusion/tests/test_frustum_occ_filter.py`
- Test: `Y:/home/ddd/pc/FullySparseFusion/tests/test_fsf_occ_completion.py`
- Test: `Y:/home/ddd/pc/FullySparseFusion/tests/test_hierarchical_hmamba.py`

- [ ] **Step 1: Write the failing test**

```python
# No new code. Validation task.
```

- [ ] **Step 2: Run test to verify it fails**

Run: `wsl bash -lc "source ~/.bashrc && conda activate FSF && cd /mnt/y/home/ddd/pc/FullySparseFusion && pytest tests/test_frustum_occ_filter.py tests/test_fsf_occ_completion.py tests/test_hierarchical_hmamba.py -v"`
Expected: If implementation is incomplete, at least one test fails with missing symbols or shape mismatches.

- [ ] **Step 3: Write minimal implementation**

```python
# Apply the fixes from Tasks 3 and 4 until the suite is green.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `wsl bash -lc "source ~/.bashrc && conda activate FSF && cd /mnt/y/home/ddd/pc/FullySparseFusion && pytest tests/test_frustum_occ_filter.py tests/test_fsf_occ_completion.py tests/test_hierarchical_hmamba.py -v"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_frustum_occ_filter.py tests/test_fsf_occ_completion.py projects/mmdet3d_plugin/models/utils/frustum_occ_filter.py projects/mmdet3d_plugin/models/detectors/FSF_Occ.py projects/configs/nuScenes/FSF_Occ_nuScenes_mini_config.py
git commit -m "feat: implement research content 2 occupancy completion pipeline"
```
