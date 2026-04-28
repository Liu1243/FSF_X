# Research Content 2 Design

**Topic:** Sparse occupancy-guided instance geometry completion and amodal center modeling for FSF

## Context

The current repository already contains a prototype implementation in [FSF_Occ.py](/Y:/home/ddd/pc/FullySparseFusion/projects/mmdet3d_plugin/models/detectors/FSF_Occ.py) and [frustum_occ_filter.py](/Y:/home/ddd/pc/FullySparseFusion/projects/mmdet3d_plugin/models/utils/frustum_occ_filter.py). That prototype improves the frustum branch with:

- per-point occupancy filtering
- amodal center offset regression
- extra occupancy and center losses

This is still narrower than the expanded requirements in `idea.md`. The missing pieces are:

- coarse-to-fine sparse occupancy completion instead of one-shot filtering
- instance-level completion descriptors beyond center offset
- weak pseudo-target construction that is explicit and testable
- real reuse of completion outputs by the detection branch

## Goal

Upgrade the current `FSF_Occ` path from a proof-of-concept into a research-content-2 implementation that covers:

- weak occupancy pseudo labels
- coarse-to-fine sparse occupancy prediction
- amodal center + shape/visibility completion targets
- completion-aware frustum features for detection
- deterministic unit tests for the new logic

## Non-Goals

This iteration does not expand into research content 3. In particular, it will not:

- modify OTA cost formulation
- change assigner behavior
- add new dataset conversion pipelines
- add full panoptic occupancy evaluation

Interfaces may be prepared for later use, but no training-time coupling with OTA is introduced now.

## Design

### 1. Upgrade `FrustumOccFilter` into an instance completion module

`FrustumOccFilter` remains the integration point because it already sits between frustum grouping and SIR aggregation. It will be extended with three internal stages:

- coarse occupancy head
  predicts a low-cost foreground prior for every frustum point
- refine occupancy head
  predicts a second-stage occupancy confidence conditioned on point features and coarse scores
- instance completion head
  aggregates valid point features per frustum instance and predicts:
  - amodal center offset `(dx, dy, dz)`
  - log-scale residual `(dl, dw, dh)`
  - visibility scalar
  - occupancy statistics embedding for reuse by the detector

This stays fully sparse because all operations are defined on frustum points and per-instance pooled features.

### 2. Explicit pseudo-target construction

The current GT generation only checks whether a point falls inside a GT box. That is useful, but too implicit. The upgraded module will make pseudo targets explicit:

- occupancy pseudo target
  positive if a point lies in a GT box; optional soft weighting based on distance to box center
- center target
  nearest matched GT center in BEV under a distance threshold
- size target
  GT box dimensions in log-space residual relative to observed instance extent
- visibility target
  ratio between in-box visible points and expected occupied points derived from matched frustum points

These targets are weakly supervised rather than dataset-native occupancy labels, which matches the research framing in `idea.md`.

### 3. Coarse-to-fine filtering policy

Instead of directly thresholding one occupancy probability:

- coarse scores suppress obvious empty/background points
- refine scores decide the final valid mask
- if a whole instance collapses after filtering, the logic falls back to the top-scoring points for that instance rather than keeping the entire raw set

This reduces noise while avoiding empty-instance failures during SIR pooling.

### 4. Completion-aware reuse by the frustum detector

The existing prototype only changes filtering and centers. The upgraded detector will also reuse completion results in feature space.

`FSF_Occ.frustum_forward` will concatenate a compact completion descriptor to `lidar_feat` before the frustum head:

- pooled occupancy confidence statistics
- predicted size residual
- predicted visibility
- center shift magnitude

This makes the occupancy/completion branch a real auxiliary detector component rather than an isolated side loss.

### 5. Configuration changes

`projects/configs/nuScenes/FSF_Occ_nuScenes_mini_config.py` will be updated to expose:

- coarse and refine occupancy head widths
- completion head widths
- thresholds and fallback behavior
- loss weights for occupancy, center, size, and visibility

Defaults should remain conservative so the config still behaves like the prototype when extra losses are lightly weighted.

## File Impact

Primary files:

- [FSF_Occ.py](/Y:/home/ddd/pc/FullySparseFusion/projects/mmdet3d_plugin/models/detectors/FSF_Occ.py)
- [frustum_occ_filter.py](/Y:/home/ddd/pc/FullySparseFusion/projects/mmdet3d_plugin/models/utils/frustum_occ_filter.py)
- [FSF_Occ_nuScenes_mini_config.py](/Y:/home/ddd/pc/FullySparseFusion/projects/configs/nuScenes/FSF_Occ_nuScenes_mini_config.py)

New tests:

- `tests/test_frustum_occ_filter.py`
- `tests/test_fsf_occ_completion.py`

## Testing Strategy

The repo currently only has unit tests for H-Mamba, so research content 2 should add isolated tests that do not depend on full MMDetection runtime.

Required unit coverage:

- occupancy pseudo-target generation from synthetic points and GT boxes
- coarse-to-fine fallback when all points are filtered
- per-instance completion target generation
- loss dictionary includes all expected keys
- completion descriptors match instance count and concatenate cleanly

Environment validation will run in WSL with `conda activate FSF`, as requested by the user.

## Risks and Mitigations

- Risk: batch/instance ordering mismatch between pooled outputs and observed centers.
  Mitigation: all pooled tensors must be aligned by a shared instance key and covered by tests.

- Risk: aggressive filtering destroys frustum recall.
  Mitigation: add per-instance fallback and keep thresholds configurable.

- Risk: new feature dimensions break the frustum detection head config.
  Mitigation: explicitly update config in_channels and add a shape test for `frustum_forward`.

## Acceptance Criteria

The implementation is considered complete when:

- `FSF_Occ` supports coarse-to-fine occupancy and multi-task completion outputs
- frustum features consume completion descriptors
- the mini config instantiates the new branch without manual patching
- new unit tests pass in the WSL `FSF` conda environment
