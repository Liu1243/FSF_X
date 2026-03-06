# Fully Sparse Fusion (FSF) Code Analysis

This document provides a source code analysis of the Fully Sparse Fusion (FSF) framework, focusing on the core components mentioned in the paper: Sparse Occupancy Guided Amodal Completion (SOG-AC) and Optimal Transport Assignment (OTA).

## 1. Overall Architecture
FSF is a multi-modal fully sparse 3D object detection framework. It processes 3D points and 2D image features using sparse structures to avoid dense operations, which leads to high efficiency and performance.
The core implementation resides in `projects/mmdet3d_plugin`, primarily structured into `models` (for networks) and `core/bbox/assigners` (for label assignment).

## 2. Sparse Occupancy Guided Amodal Completion (SOG-AC)
The SOG-AC methodology focuses on predicting the complete 3D object shapes from sparse partial observations.
In the codebase, this is driven by the cluster-based heads, such as `SparseClusterHeadV2` and `FrustumClusterHead`.
- **Sparse Feature Processing**: The features are inherently sparse representations (e.g., center, size, rotation, velocity). `SparseClusterHeadV2` processes these via shared MLPs for each task. It is robust to sparsity and implicitly performs amodal completion by learning to regress complete 3D bounding boxes (`dim`, `center`, `rot`) from clustered sparse features. 
- **Frustum Clustering**: `FrustumClusterHead` extends this by working jointly with camera-based frustum groupings to provide additional geometric constraints, guiding the sparse features for completion.

## 3. Optimal Transport Assignment (OTA)
The assignment of predictions (clusters) to ground truths in a sparse regime is crucial. The pipeline utilizes custom assigners in `projects/mmdet3d_plugin/core/bbox/assigners/`:
- **DistAssigner (`dist_assigner.py`)**: This assigner calculates the BEV (Bird's-Eye View) distance matrix between predicted cluster centers and ground truth centers (`torch.cdist(pd_xy, gt_centers)`). It matches predictions to the nearest ground truth within a specified `max_dist` threshold per class. This distance-based bipartite matching embodies the principle of Optimal Transport Assignment (OTA) presented in the paper, simplified into a nearest-neighbor thresholded OT problem.
- **FrustumAssigner (`frustum_assigner.py`)**: To facilitate multi-modal matching, this assigner projects 3D cluster points or bounding boxes onto the 2D image plane (`prj_lidar_bbox3d_on_img`). It merges 3D assignment results with 2D image assignment results. If a 3D cluster is not assigned in the 3D space but successfully maps to a 2D ground truth bounding box on the image, the `FrustumAssigner` will replace the negative 3D mask with the positive 2D mask (`merge_3d_2d_assign_result`). This builds the bridge between LiDAR and Camera modalities during target assignment.

## 4. Summary
FSF manages to fuse multi-modal data entirely in a sparse representation. It relies heavily on:
1. **Cluster Heads (SOG-AC)** (`sparse_cluster_head_v2.py`, `frustum_cluster_head.py`) to regress bounding boxes from sparse points.
2. **Advanced Assigners (OTA)** (`dist_assigner.py`, `frustum_assigner.py`) to seamlessly assign labels across 3D geometries and 2D camera planes using minimal transport distances.
