# SparseOcc 代码与论文分析

本文档综合了 `SparseOcc` 论文（arxiv 2312.17118）以及 `tmp/SparseOcc` 下的源码实现，对 SparseOcc 的整体架构与核心组件进行了深入分析。

## 1. 整体架构 (Overall Architecture)

SparseOcc 提出了一个完全稀疏的 3D 全景占据（Panoptic Occupancy）预测框架。与以往依赖构建密集 3D 体素（Dense 3D Volume）进行占据预测的方法不同，SparseOcc 指出场景中超过 90% 的体素都是空闲的（Free）。因此，它通过稀疏查询（Sparse Queries）来重建 3D 稀疏表示，这极大降低了计算负担。

从代码入口 `models/sparseocc.py` 中的 `SparseOcc` 类可以看出，模型主要分为以下几个阶段：
1. **图像特征提取**: 利用 2D Backbone（如 ResNet）和 Neck 提取多视图图像特征（`extract_img_feat`）。
2. **Sparse Voxel Decoder (稀疏体素解码器)**: 从视觉输入中重建稀疏的 3D 几何结构。这是由 `models/sparseocc_transformer.py` 和 `models/sparse_voxel_decoder.py` 协同完成的。
3. **Mask Transformer (掩码 Transformer)**: 基于前面的 3D 稀疏表示，使用稀疏掩码查询（Mask Queries）来预测语义和实例的覆盖范围。

## 2. 核心模块分析

### 2.1 Sparse Voxel Decoder (稀疏体素解码器)

论文与代码对应指出，Sparse Voxel Decoder 采取了由粗到细（Coarse-to-fine）的层次结构：
- **对应代码**: `models/sparse_voxel_decoder.py` 中的 `SparseVoxelDecoder` 及其内部的 `SparseVoxelDecoderLayer`。
- **机制与流程**: 
  1. 解码器具有多层（如3层）结构。每一层输入一组稀疏的体素查询（Voxel Queries），包括它们的三维坐标和通道特征向量。
  2. 代码中首先进行 Self-Attention 来聚合全局信息（仅在前两层激活），这得益于在稀疏形式下的计算优势。
  3. 然后，通过 `SparseBEVSampling` 和 `AdaptiveMixing`，根据当前查询的 3D 坐标生成 3D 采样点（Offset），投影至 2D 多视角图像并在多尺度图像特征图上进行双线性采样和特征融合（Cross-Attention 机制的轻量级实现）。
  4. **稀疏化 (Sparsification)**: 每一层结束前，预测每个体素的占据分数（Score），并据此剔除被判断为空（Free）的体素。代码中采用的是计算 `non_free_prob = 1 - softmax(seg_pred)[..., -1]`，并取分数最高的 `top-k` 个体素传入下一层。论文提到使用 `top-k` 取代直接阈值截断可以固定样本长度，更利于训练效率。之后，保留的体素会被进行空间上溯（Upsample 2x，即一分为八），增加分辨率供下一层迭代。
- **时序建模 (Temporal Modeling)**: SparseOcc 直接使用 3D 采样点反投影到历史多帧图像上提取特征来实现时序融合，因为模型本身是稀疏的，这比基于密集网格或 3D 卷积进行特征对齐更高效。

### 2.2 Mask Transformer

这部分受到了 Mask2Former 思想启发，将最终的占据预测解耦为了掩码查询和内容向量：
- **对应代码**: `models/sparseocc_transformer.py` 中的 `MaskFormerOccDecoder` 和 `MaskFormerOccDecoderLayer`，以及输出侧的 `sparseocc_head.py`。
- **运行流程**: 
  1. 使用一组固定数量的实例查询（Queries）与稀疏体素特征（来自于 Decoder）进行交互。
  2. 其架构包含了 Multi-Head Self Attention (MHSA)、Mask-guided Sparse Sampling (掩码引导稀疏采样) 以及 Adaptive Mixing。
  3. **掩码引导稀疏采样**: 区别于标准交叉注意力机制（过于耗时），网络从上一层预测的掩码（Mask）内随机选取 3D 体素点作为采样点，反向投影到 2D 图像进行特征采样。代码中体现在 `MaskFormerSampling` 的 `make_sample_points_from_mask` 方法上。
- **分类与预测**: 
  - `sparseocc_head.py` 中的网络通过 MLP 将 Query Embedding 映射为分类结果（`class_pred`）和掩码权重（`mask_pred`）。掩码预测空间被严格限制在前面 Sparse Voxel Decoder 输出的非空（Non-free）稀疏 3D 体素上，而不是整个密集的 3D 空间中。掩码会与类别信息合并生成语义分割或实例分割（Panoptic）输出。

## 3. 损失函数与监督策略 (Supervision)

代码 `models/sparseocc_head.py` 的 `loss` 函数和论文内容相呼应：
- **Sparse Voxel Decoder**：对稀疏层级的重建结果用 Lovasz Softmax Loss, Semantic/Geometric Scaled Loss, 以及 CE SSC Loss 进行监督。这里因为不计算已被剔除的空间，能够节省大量训练开销。对于不平衡的类别，使用了频率倒数作为权重（`NUSC_CLASS_FREQ`）。
- **Mask Transformer**：使用基于二分图匹配（Hungarian Matching）的 `loss_mask2former`，包括 Focal Loss (用于分类)、DICE Loss 和 BCE Loss (用于掩码)。

## 4. 评价指标：RayIoU

除模型结构外，论文重点提出了一种新的基于射线追踪的评估指标 **RayIoU**。
以往的体素级 mIoU 会对于物体表面厚度有不一致的惩罚（预测表面厚会导致较差的 mIoU）。SparseOcc 的全稀疏架构总是预测薄的表面。RayIoU 通过从空间中发射射线寻找碰撞的占据点来进行比对，可以对不管是“厚表面”还是“薄表面”给出更公平合理的空间三维重建评估。

## 总结

SparseOcc 最大的工程与模型亮点是将“由粗到细的 3D 几何提取”和“掩码机制（Mask2Former）”结合，且通过严格的在每阶段进行 `top-k` 稀疏截断，极大程度上解决了 3D Occupancy 密集图算力消耗问题，实现了高性能下的完全稀疏的 3D 占据预测。
