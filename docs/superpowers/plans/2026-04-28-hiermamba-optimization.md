# HierMamba Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve FSF_HierMamba long-tail and small-object handling with category-aware token retention, richer reliability descriptors, and a lightweight residual correction head.

**Architecture:** Extend the existing `ForegroundTokenSelector` rather than replacing it. Keep `HierarchicalHMambaInteraction` API backward compatible and make all new behavior opt-in through `hiermamba_cfg`.

**Tech Stack:** PyTorch modules under `projects/mmdet3d_plugin/models/utils`, config plumbing in `FSF_HierMamba.py`, unit tests with `pytest`/`unittest`.

---

### Task 1: Category-Aware Token Retention

**Files:**
- Modify: `tests/test_hierarchical_hmamba.py`
- Modify: `projects/mmdet3d_plugin/models/utils/hierarchical_hmamba.py`

- [ ] **Step 1: Write the failing test**

Add a test where global top-k would drop a low-score long-tail class, but `class_group_min_tokens` preserves it.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_hierarchical_hmamba.py::HierarchicalHMambaTests::test_foreground_selector_preserves_configured_class_group -q`

- [ ] **Step 3: Implement minimal selector support**

Add optional `class_groups` and `class_group_min_tokens` constructor args. During per-batch selection, mark the top group-scoring tokens for each configured group before filling the remaining global top-k slots.

- [ ] **Step 4: Run test to verify it passes**

Run the same pytest command and then the full HierMamba test file.

### Task 2: Reliability Descriptor Extension

**Files:**
- Modify: `tests/test_hierarchical_hmamba.py`
- Modify: `projects/mmdet3d_plugin/models/utils/hierarchical_hmamba.py`

- [ ] **Step 1: Write the failing test**

Assert that `_reliability_descriptors` returns 9 dimensions when extended descriptors are enabled and includes valid normalized center-distance, entropy, image score, 2D validity, range confidence, aspect cue, and projection consistency terms.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_hierarchical_hmamba.py::HierarchicalHMambaTests::test_extended_reliability_descriptors_include_geometry_cues -q`

- [ ] **Step 3: Implement descriptor support**

Add `use_extended_reliability` and choose descriptor dimension in `HierarchicalHMambaInteraction.__init__`; compute extra cues from centers, class logits, and available 2D prediction columns.

- [ ] **Step 4: Run test to verify it passes**

Run the single test and full HierMamba tests.

### Task 3: Small-Object Residual Correction

**Files:**
- Modify: `tests/test_hierarchical_hmamba.py`
- Modify: `projects/mmdet3d_plugin/models/utils/hierarchical_hmamba.py`
- Modify: `projects/mmdet3d_plugin/models/detectors/FSF_HierMamba.py`
- Modify: `projects/configs/nuScenes/FSF_HierMamba_nuScenes_mini_config.py`
- Modify: `projects/configs/nuScenes/FSF_HierMamba_nuScenes_config.py`
- Modify: `projects/configs/Argoverse2/FSF_HierMamba_AV2_config.py`

- [ ] **Step 1: Write the failing test**

Assert that enabling `small_object_class_indices` creates correction outputs in `aux`, only activates masks for configured classes, and preserves feature shape.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_hierarchical_hmamba.py::HierarchicalHMambaTests::test_small_object_residual_head_reports_configured_tokens -q`

- [ ] **Step 3: Implement minimal correction head**

Add a lightweight residual MLP initialized to zero. Apply a gated feature correction for selected tokens whose predicted class is configured as small/difficult, and expose `small_object_mask`, `small_object_pose_residual`, `small_object_scale_residual`, and `small_object_temperature` in `aux`.

- [ ] **Step 4: Wire config**

Pass new settings from `FSF_HierMamba` and enable them in HierMamba configs for classes `[2, 4, 5, 6, 8, 9]`.

- [ ] **Step 5: Run verification**

Run: `python -m pytest tests/test_hierarchical_hmamba.py -q`
