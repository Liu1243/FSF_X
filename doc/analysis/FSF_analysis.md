# Fully Sparse Fusion (FSF) 代码分析

本文档提供了 Fully Sparse Fusion (FSF) 框架的源代码分析，重点关注论文中提到的核心组件：稀疏占据引导的非模态补全 (Sparse Occupancy Guided Amodal Completion, SOG-AC) 和最优传输分配 (Optimal Transport Assignment, OTA)。

## 1. 整体架构
FSF 是一个多模态全稀疏 3D 目标检测框架。它使用稀疏结构处理 3D 点云和 2D 图像特征，以避免密集的计算操作，从而带来高度的效率和性能。
核心实现位于 `projects/mmdet3d_plugin`，主要分为 `models` (用于网络) 和 `core/bbox/assigners` (用于标签分配)。

## 2. 稀疏占据引导的非模态补全 (SOG-AC)
SOG-AC 方法侧重于从稀疏的局部观测中预测完整的 3D 物体形状。
在代码库中，这是由基于聚类的检测头驱动的，例如 `SparseClusterHeadV2` 和 `FrustumClusterHead`。
- **稀疏特征处理**: 特征天生就是稀疏表示 (例如：中心、尺寸、旋转、速度)。`SparseClusterHeadV2` 通过每个任务共享的多层感知机 (MLP) 来处理这些特征。它对稀疏性具有鲁棒性，并通过学习从聚类的稀疏特征中回归出完整的 3D 边界框 (`dim`, `center`, `rot`) ，从而隐式地执行非模态补全。
- **视锥聚类 (Frustum Clustering)**: `FrustumClusterHead` 在此基础上进行了扩展，它与基于相机的视锥分组联合工作，以提供额外的几何约束，从而指导稀疏特征进行补全。

## 3. 最优传输分配 (OTA)
在稀疏体制下将预测 (聚类) 分配给真实标注 (ground truths) 是至关重要的。该流水线利用 `projects/mmdet3d_plugin/core/bbox/assigners/` 中的自定义分配器：
- **DistAssigner (`dist_assigner.py`)**: 该分配器计算预测聚类中心和真实标注中心之间的 BEV (鸟瞰图) 距离矩阵 (`torch.cdist(pd_xy, gt_centers)`)。它将预测与每个类别指定 `max_dist` 阈值内最近的真实标注进行匹配。这种基于距离的二分匹配体现了论文中提出的最优传输分配 (OTA) 原理，将其简化为具有阈值限制的最近邻最优传输问题。
- **FrustumAssigner (`frustum_assigner.py`)**: 为了促进多模态匹配，该分配器将 3D 聚类点或边界框投影到 2D 图像平面上 (`prj_lidar_bbox3d_on_img`)。它合并 3D 分配结果和 2D 图像分配结果。如果在 3D 空间中未分配 3D 聚类，但成功映射到了图像上的 2D 真实边界框，则 `FrustumAssigner` 会用正样本的 2D 掩码替换掉负样本的 3D 掩码 (`merge_3d_2d_assign_result`)。这在目标分配期间建立了 LiDAR 和相机模态之间的桥梁。

## 4. 总结
FSF 成功地在完全稀疏特征的表示方式下融合了多模态数据。它在很大程度上依赖于：
1. **聚类检测头 (SOG-AC)** (`sparse_cluster_head_v2.py`, `frustum_cluster_head.py`)，用于从稀疏点中预测回归边界框。
2. **高级分配器 (OTA)** (`dist_assigner.py`, `frustum_assigner.py`)，使用最小的传输距离，在 3D 几何结构和 2D 相机平面之间无缝地分配标签。
