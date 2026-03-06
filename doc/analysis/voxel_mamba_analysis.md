# Voxel Mamba 源码解析

本文档旨在对位于 `Voxel-Mamba/pcdet/models/backbones_3d/voxel_mamba_waymo.py` 及其相关工具文件中的 **Voxel Mamba** 核心源码进行解读，侧重分析其模型网络架构与创新机制。

## 1. 整体架构概述

Voxel Mamba 被设计为在 OpenPCDet 框架下运行的 3D Backbone (类 `Voxel_Mamba_Waymo`)，旨在避免三维点云处理中局部感受野运算（如 KNN、Ball Query 引起的 Grouping 分组操作）带来的高额计算开销，实现真正意义上的 **Group-Free**。

它的核心思想是：不依赖局部的窗口或半径搜索，而是将全场景内所有非空稀疏 Voxel 取出，按照特定的空间曲线映射规则（Hilbert Curve）排布成一长串连续的一维序列数据，并借助拥有线性复杂度且擅长长序列建模的 State Space Models (Mamba) 进行全局特征的连续扫描与提取。

## 2. 连续化（Serialization）与 Hilbert Curve 机制

传统的 3D 卷积主要依靠对周围固定邻域内体素的寻找与计算，一旦转化为 Mamba 能够处理的一维序列，必须保证这种映射变换能够最大限度地**保留原始 3D 空间上的局部相邻关系**。Voxel Mamba 采用的是三维 **Hilbert 曲线（Hilbert Curve）** 映射机制。

相关代码主要在 `pcdet/models/model_utils/voxel_mamba_utils.py` 中的 `get_hilbert_index_3d_mamba_lite` 函数：

```python
def get_hilbert_index_3d_mamba_lite(template, coors, batch_size, z_dim, hilbert_spatial_size, shift=(0, 0, 0), debug=True):
    # 根据 Voxel 坐标和网格尺寸，将其拍平为 1D 的绝对空间索引 flat_coors
    flat_coors = (z * hil_size_y * hil_size_x + y * hil_size_x + x).long()
    
    # 根据提前生成并加载的 template（Hilbert 映射模板）查询该绝对位置在 Hilbert 连线中的顺序值
    hil_inds = template[flat_coors].long()

    inds_curt_to_next = {}
    inds_next_to_curt = {}
    for i in range(batch_size):
        batch_mask = coors[:, 0] == i
        
        # 1. argsort 得到 Voxel 数据转化为一维序列的索引 (Sparse To Seq)
        inds_curt_to_next[i] = torch.argsort(hil_inds[batch_mask])
        
        # 2. 再次 argsort 得到 一维序列还原回 Voxel 数据的索引 (Seq To Sparse)
        inds_next_to_curt[i] = torch.argsort(inds_curt_to_next[i])

    # ... 返回映射关系 ...
```

通过这一巧妙设计，无序的稀疏 `coors` （坐标）可以快速转换为连续的一维特征张量，从而可以喂入 Mamba 的处理层，处理完毕之后也能被绝对精准无误差地还原到各自原本的空间位置。

## 3. 核心计算模块：DSB (Dual-scale State Space Models Block)

Voxel Mamba 网络的主干包含多个堆叠的级联模块 `DSB`。该模块包含了空间下采样、位置重编码嵌入、Mamba 变体网络推断以及残差回传等机制。以下按执行数据流向依次解读：

### a. 双分支结构提取与下采样
在传入体素数据后，首先利用 `DownSp` 层使用稀疏卷积获得两个分支（高分辨率保留 `x_s1` 和下采样获得粗粒度大范围信息 `x_s2`）。`x_s1` 和 `x_s2` 会各自通过 Hilbert 曲线函数获得独立的排序索引。

### b. Relative Position Embedding (相对特征编码嵌入)
在输入 Mamba 前，为消除空间三维结构抹平为一维时潜在的方向感缺失，DSB 生成了精巧的 Position Embedding：
```python
# 计算各个 Voxel 在 3D 网格中的标准化坐标，以及在固定网格间隔(12)处的相对偏移量等，赋予 9 个维度的特征。
pos_embed_coords_s2[:, 0] = coords_s2[:, 1] / x_s2.spatial_shape[0]  # Z 维度归一化
# 对 Y 和 X 维度除了使用相对大小，还结合了正弦波周期特性的近似取余处理等操作
pos_embed_coords_s2[:, 1:3] = (coords_s2[:, 2:] // 12) / (x_s2.spatial_shape[1]//12 + 1)
# ... 并经过两层 Linear 获得 pos_embed
```
附加 Positional Embedding 后的特征更加具备对点云稀疏位置的辨识度。

### c. 前向与反向的双向 Mamba (Bidirectional Scanning)
因为传统 Mamba（用于NLP领域）是一个**单向**因果推理模型（Causal Model），但在计算机视觉和点云目标检测中，每个体素都应该能够汇聚来自“前面”与“后面”体素的信息（非因果的）。因此，Voxel Mamba 在处理双分支时执行了相反的操作扫掠，模拟了类似 Vision Mamba (Vim) 的机制：
- **前向扫描 (低分辨率粗粒分支 x_s2)**：
  根据 `inds_curt_to_next_s2` 按正序拉成序列 `feat_m2`，直接投入第一个 Mamba 层 (`mamba_layer1`)进行推断，然后借助逆向映射 `inds_next_to_curt_s2` 还原回 3D 卷积的索引排序。
- **反向扫描 (高分辨率细粒分支 x_s1)**：
  将排好序的序列 `feat_m1` 在输入 Mamba 前通过 `.flip(1)` 将维度上的序列数据强制翻转以得到逆序序列。送入第二个 Mamba 层 (`mamba_layer2`) 后，将输出的序列再一次 `.flip(0)` 翻滚矫正，再通过逆向映射还原。

### d. 反卷积（上采样）与残差融合
获取处理好的两种特征表达后，`DSB` 后半部分使用 Inverse Convolution 即 `post_act_block(conv_type='inverseconv')` 将较低分辨率提取到的粗尺度、更全局视野的语义特征图，放大（恢复）分辨率回匹配 `x_s1` 甚至更高原始传入的初始分辨率尺寸 `features[0]` 上，最后通过相加：
```python
x = replace_feature(x, x.features + up_x.features + features[0].features)
```
汇集完成了从序列、全局推延模型反馈到本地细颗粒层级点云的残差跳跃连接。

## 4. 总结与优势

1. **摆脱 O(N^2) 定律：** 当场景体积增大，点云密度剧增时，不需要遍历查找邻居，复杂度从平方项降低到 `O(N)`。
2. **兼备全局视野与局部紧密联系：** Hilbert 曲线保证了一维空间上的邻居大抵等于三维上的邻居；借由 Mamba 的 State Selection 特性可以长距离记忆和传播。
3. **架构的即插即用：** 很好地融入到以 SPConv 为核心逻辑的点云检测代码框架中（通过提取特征 `features`，记录位置索引 `indices`，执行自定义算法）。它展现了 SSM (State Space Models) 原理可以很优雅地去替补和升级大模型 3D 主干网络。
