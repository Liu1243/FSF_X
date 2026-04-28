Fully Sparse 3D Occupancy Prediction

1.  [1 Introduction](https://arxiv.org/html/2312.17118v5#S1 "In Fully Sparse 3D Occupancy Prediction") 2.  [2 Related Work](https://arxiv.org/html/2312.17118v5#S2 "In Fully Sparse 3D Occupancy Prediction") 1.  [Camera-based 3D Occupancy Prediction.](https://arxiv.org/html/2312.17118v5#S2.SS0.SSS0.Px1 "In 2 Related Work ‣ Fully Sparse 3D Occupancy Prediction") 2.  [Sparse Architectures for 3D Vision.](https://arxiv.org/html/2312.17118v5#S2.SS0.SSS0.Px2 "In 2 Related Work ‣ Fully Sparse 3D Occupancy Prediction") 3.  [End-to-end 3D Reconstruction from Posed Images.](https://arxiv.org/html/2312.17118v5#S2.SS0.SSS0.Px3 "In 2 Related Work ‣ Fully Sparse 3D Occupancy Prediction") 4.  [Mask Transformer.](https://arxiv.org/html/2312.17118v5#S2.SS0.SSS0.Px4 "In 2 Related Work ‣ Fully Sparse 3D Occupancy Prediction") 3.  [3 SparseOcc](https://arxiv.org/html/2312.17118v5#S3 "In Fully Sparse 3D Occupancy Prediction") 1.  [3.1 Sparse Voxel Decoder](https://arxiv.org/html/2312.17118v5#S3.SS1 "In 3 SparseOcc ‣ Fully Sparse 3D Occupancy Prediction") 1.  [Overall architecture.](https://arxiv.org/html/2312.17118v5#S3.SS1.SSS0.Px1 "In 3.1 Sparse Voxel Decoder ‣ 3 SparseOcc ‣ Fully Sparse 3D Occupancy Prediction") 2.  [Detailed design.](https://arxiv.org/html/2312.17118v5#S3.SS1.SSS0.Px2 "In 3.1 Sparse Voxel Decoder ‣ 3 SparseOcc ‣ Fully Sparse 3D Occupancy Prediction") 3.  [Temporal modeling.](https://arxiv.org/html/2312.17118v5#S3.SS1.SSS0.Px3 "In 3.1 Sparse Voxel Decoder ‣ 3 SparseOcc ‣ Fully Sparse 3D Occupancy Prediction") 4.  [Supervision.](https://arxiv.org/html/2312.17118v5#S3.SS1.SSS0.Px4 "In 3.1 Sparse Voxel Decoder ‣ 3 SparseOcc ‣ Fully Sparse 3D Occupancy Prediction") 2.  [3.2 Mask Transformer](https://arxiv.org/html/2312.17118v5#S3.SS2 "In 3 SparseOcc ‣ Fully Sparse 3D Occupancy Prediction") 1.  [Mask-guided sparse sampling.](https://arxiv.org/html/2312.17118v5#S3.SS2.SSS0.Px1 "In 3.2 Mask Transformer ‣ 3 SparseOcc ‣ Fully Sparse 3D Occupancy Prediction") 2.  [Prediction.](https://arxiv.org/html/2312.17118v5#S3.SS2.SSS0.Px2 "In 3.2 Mask Transformer ‣ 3 SparseOcc ‣ Fully Sparse 3D Occupancy Prediction") 3.  [Supervision.](https://arxiv.org/html/2312.17118v5#S3.SS2.SSS0.Px3 "In 3.2 Mask Transformer ‣ 3 SparseOcc ‣ Fully Sparse 3D Occupancy Prediction") 4.  [Loss Functions.](https://arxiv.org/html/2312.17118v5#S3.SS2.SSS0.Px4 "In 3.2 Mask Transformer ‣ 3 SparseOcc ‣ Fully Sparse 3D Occupancy Prediction") 4.  [4 Ray-level mIoU](https://arxiv.org/html/2312.17118v5#S4 "In Fully Sparse 3D Occupancy Prediction") 1.  [4.1 Revisiting the Voxel-level mIoU](https://arxiv.org/html/2312.17118v5#S4.SS1 "In 4 Ray-level mIoU ‣ Fully Sparse 3D Occupancy Prediction") 2.  [4.2 Mean IoU by Ray Casting](https://arxiv.org/html/2312.17118v5#S4.SS2 "In 4 Ray-level mIoU ‣ Fully Sparse 3D Occupancy Prediction") 5.  [5 Experiments](https://arxiv.org/html/2312.17118v5#S5 "In Fully Sparse 3D Occupancy Prediction") 1.  [5.1 Implementation Details](https://arxiv.org/html/2312.17118v5#S5.SS1 "In 5 Experiments ‣ Fully Sparse 3D Occupancy Prediction") 2.  [5.2 Main Results](https://arxiv.org/html/2312.17118v5#S5.SS2 "In 5 Experiments ‣ Fully Sparse 3D Occupancy Prediction") 3.  [5.3 Ablations](https://arxiv.org/html/2312.17118v5#S5.SS3 "In 5 Experiments ‣ Fully Sparse 3D Occupancy Prediction") 1.  [Sparse voxel decoder vs. dense voxel decoder.](https://arxiv.org/html/2312.17118v5#S5.SS3.SSS0.Px1 "In 5.3 Ablations ‣ 5 Experiments ‣ Fully Sparse 3D Occupancy Prediction") 2.  [Mask Transformer.](https://arxiv.org/html/2312.17118v5#S5.SS3.SSS0.Px2 "In 5.3 Ablations ‣ 5 Experiments ‣ Fully Sparse 3D Occupancy Prediction") 3.  [Is a limited set of voxels sufficient to cover the scene?](https://arxiv.org/html/2312.17118v5#S5.SS3.SSS0.Px3 "In 5.3 Ablations ‣ 5 Experiments ‣ Fully Sparse 3D Occupancy Prediction") 4.  [Temporal modeling.](https://arxiv.org/html/2312.17118v5#S5.SS3.SSS0.Px4 "In 5.3 Ablations ‣ 5 Experiments ‣ Fully Sparse 3D Occupancy Prediction") 4.  [5.4 More Studies](https://arxiv.org/html/2312.17118v5#S5.SS4 "In 5 Experiments ‣ Fully Sparse 3D Occupancy Prediction") 1.  [The effect of training with visible masks.](https://arxiv.org/html/2312.17118v5#S5.SS4.SSS0.Px1 "In 5.4 More Studies ‣ 5 Experiments ‣ Fully Sparse 3D Occupancy Prediction") 2.  [Panoptic occupancy.](https://arxiv.org/html/2312.17118v5#S5.SS4.SSS0.Px2 "In 5.4 More Studies ‣ 5 Experiments ‣ Fully Sparse 3D Occupancy Prediction") 5.  [5.5 Limitations](https://arxiv.org/html/2312.17118v5#S5.SS5 "In 5 Experiments ‣ Fully Sparse 3D Occupancy Prediction") 1.  [Accumulative errors.](https://arxiv.org/html/2312.17118v5#S5.SS5.SSS0.Px1 "In 5.5 Limitations ‣ 5 Experiments ‣ Fully Sparse 3D Occupancy Prediction") 6.  [6 Conclusion](https://arxiv.org/html/2312.17118v5#S6 "In Fully Sparse 3D Occupancy Prediction")

# Fully Sparse 3D Occupancy Prediction

Haisong Liu1,2∗, Yang Chen1∗, Haiguang Wang1, Zetong Yang2, Tianyu Li2, Jia Zeng2, Li Chen2, Hongyang Li2, Limin Wang1,2,✉ 1Nanjing University    2Shanghai AI Lab [https://github.com/MCG-NJU/SparseOcc](https://github.com/MCG-NJU/SparseOcc)

###### Abstract

Occupancy prediction plays a pivotal role in autonomous driving. Previous methods typically construct dense 3D volumes, neglecting the inherent sparsity of the scene and suffering from high computational costs. To bridge the gap, we introduce a novel fully sparse occupancy network, termed SparseOcc. SparseOcc initially reconstructs a sparse 3D representation from camera-only inputs and subsequently predicts semantic/instance occupancy from the 3D sparse representation by sparse queries. A mask-guided sparse sampling is designed to enable sparse queries to interact with 2D features in a fully sparse manner, thereby circumventing costly dense features or global attention. Additionally, we design a thoughtful ray-based evaluation metric, namely RayIoU, to solve the inconsistency penalty along the depth axis raised in traditional voxel-level mIoU criteria. SparseOcc demonstrates its effectiveness by achieving a RayIoU of 34.0, while maintaining a real-time inference speed of 17.3 FPS, with 7 history frames inputs. By incorporating more preceding frames to 15, SparseOcc continuously improves its performance to 35.1 RayIoU without bells and whistles.

00footnotetext: \*: Equal contribution.00footnotetext: ✉: Corresponding author.

## 1 Introduction

Vision-centric 3D occupancy prediction \[[1](https://arxiv.org/html/2312.17118v5#bib.bib1)\] focuses on partitioning 3D scenes into structured grids from visual images. Each grid is assigned a label indicating if it is occupied or not. This task offers more geometric details than 3D object detection and produces an alternative representation to LiDAR-based perception \[[63](https://arxiv.org/html/2312.17118v5#bib.bib63), [23](https://arxiv.org/html/2312.17118v5#bib.bib23), [60](https://arxiv.org/html/2312.17118v5#bib.bib60), [61](https://arxiv.org/html/2312.17118v5#bib.bib61), [62](https://arxiv.org/html/2312.17118v5#bib.bib62), [31](https://arxiv.org/html/2312.17118v5#bib.bib31), [32](https://arxiv.org/html/2312.17118v5#bib.bib32)\].

![Refer to caption](x1.png)

(a)

![Refer to caption](x2.png)

(b)

Figure 1: (a) SparseOcc reconstructs a sparse 3D representation from camera-only inputs by a sparse voxel decoder, and then estimates the mask and label of each segment via a set of sparse queries. (b) Performance comparison on the validation split of Occ3D-nuScenes. FPS is measured on a Tesla A100 with the PyTorch fp32 backend.

Existing methods \[[27](https://arxiv.org/html/2312.17118v5#bib.bib27), [16](https://arxiv.org/html/2312.17118v5#bib.bib16), [57](https://arxiv.org/html/2312.17118v5#bib.bib57), [45](https://arxiv.org/html/2312.17118v5#bib.bib45), [28](https://arxiv.org/html/2312.17118v5#bib.bib28)\] typically construct dense 3D features yet suffer from computational overhead (*e.g*., $2\sim 3$ FPS on the Tesla A100 GPU). However, dense representations are not necessary for occupancy prediction. We statistic the geometry sparsity and find that more than 90% of the voxels are empty. This manifests a large room in occupancy prediction acceleration by exploiting the sparsity. Some works \[[26](https://arxiv.org/html/2312.17118v5#bib.bib26), [19](https://arxiv.org/html/2312.17118v5#bib.bib19)\] explore the sparsity of 3D scenes, but they still rely on sparse-to-dense modules for dense predictions. This inspires us to seek a pure sparse occupancy network without any dense design.

In this paper, we propose SparseOcc, the first fully sparse occupancy network. As depicted in Fig. [1](https://arxiv.org/html/2312.17118v5#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Fully Sparse 3D Occupancy Prediction") (a), SparseOcc includes two steps. First, it leverages a sparse voxel decoder to reconstruct the sparse geometry of a scene in a coarse-to-fine manner. This only models non-free regions, saving computational costs significantly. Second, we design a mask transformer with sparse semantic/instance queries to predict masks and labels of segments from the sparse space. The mask transformer not only improves performance on semantic occupancy but also paves the way for panoptic occupancy. A mask-guided sparse sampling is designed to achieve sparse cross-attention in the mask transformer. As such, our SparseOcc fully exploits the sparse property and gets rid of any dense design like dense 3D features, sparse-to-dense modules, and global attention.

Besides, we notice flaws in popular voxel-level mean Intersection-over-Union (mIoU) metrics for occupancy evaluation and further design a ray-level evaluation, RayIoU, as the solution. The mIoU criterion is an ill-posed formulation given the ambiguous labeling of unscanned voxels. Previous methods\[[50](https://arxiv.org/html/2312.17118v5#bib.bib50)\] relieve this issue by only evaluating observed areas but raise extra issues in inconsistency penalty along depths. Instead, RayIoU addresses the two aforementioned issues simultaneously. It evaluates predicted 3D occupancy volume by retrieving depth and category predictions of designated rays. To be specific, RayIoU casts query rays into predicted 3D volumes and decides true positive predictions as the ray with the correct distance and class of its first touched occupied voxel grid. This formulates a more fair and reasonable criterion.

Thanks to the sparsity design, SparseOcc achieves 34.0 RayIoU on Occ3D-nuScenes \[[50](https://arxiv.org/html/2312.17118v5#bib.bib50)\], while maintaining a real-time inference speed of 17.3 FPS (Tesla A100, PyTorch fp32 backend), with 7 history frames inputs. By incorporating more preceding frames to 15, SparseOcc continuously improves its performance to 35.1 RayIoU, achieving state-of-the-art performance without bells and whistles. The comparison between SparseOcc with previous methods in terms of performance and efficiency is shown in Fig. [1](https://arxiv.org/html/2312.17118v5#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Fully Sparse 3D Occupancy Prediction") (b).

We summarize our contributions as follows:

1.  1.

We propose SparseOcc, the first fully sparse occupancy network without any time-consuming dense designs. It achieves 34.0 RayIoU on Occ3D-nuScenes benchmark with an real-time inference speed of 17.3 FPS.

2.  2.

We present RayIoU, a ray-wise criterion for occupancy evaluation. By querying rays to 3D volume, it solves the ambiguous penalty issue for unscanned free voxels and the inconsistent depth penalty issue in the mIoU metric.

## 2 Related Work

#### Camera-based 3D Occupancy Prediction.

The occupancy network is originally proposed by Mescheder *et al*. \[[37](https://arxiv.org/html/2312.17118v5#bib.bib37), [42](https://arxiv.org/html/2312.17118v5#bib.bib42)\], focusing on continuous object representations in 3D space. Recent variations \[[1](https://arxiv.org/html/2312.17118v5#bib.bib1), [4](https://arxiv.org/html/2312.17118v5#bib.bib4), [45](https://arxiv.org/html/2312.17118v5#bib.bib45), [50](https://arxiv.org/html/2312.17118v5#bib.bib50), [54](https://arxiv.org/html/2312.17118v5#bib.bib54), [56](https://arxiv.org/html/2312.17118v5#bib.bib56), [11](https://arxiv.org/html/2312.17118v5#bib.bib11), [58](https://arxiv.org/html/2312.17118v5#bib.bib58)\] mostly draw inspiration from Bird’s Eye View (BEV) perception \[[25](https://arxiv.org/html/2312.17118v5#bib.bib25), [24](https://arxiv.org/html/2312.17118v5#bib.bib24), [27](https://arxiv.org/html/2312.17118v5#bib.bib27), [16](https://arxiv.org/html/2312.17118v5#bib.bib16), [15](https://arxiv.org/html/2312.17118v5#bib.bib15), [18](https://arxiv.org/html/2312.17118v5#bib.bib18), [17](https://arxiv.org/html/2312.17118v5#bib.bib17), [55](https://arxiv.org/html/2312.17118v5#bib.bib55), [34](https://arxiv.org/html/2312.17118v5#bib.bib34), [35](https://arxiv.org/html/2312.17118v5#bib.bib35), [30](https://arxiv.org/html/2312.17118v5#bib.bib30), [33](https://arxiv.org/html/2312.17118v5#bib.bib33), [53](https://arxiv.org/html/2312.17118v5#bib.bib53), [59](https://arxiv.org/html/2312.17118v5#bib.bib59)\] and predicts voxel-level semantic information from image inputs. For instance, MonoScene \[[4](https://arxiv.org/html/2312.17118v5#bib.bib4)\] estimates occupancy through a 2D and a 3D UNet \[[43](https://arxiv.org/html/2312.17118v5#bib.bib43)\] connected by a sight projection module. SurroundOcc \[[57](https://arxiv.org/html/2312.17118v5#bib.bib57)\] proposes a coarse-to-fine architecture. However, the large number of voxel queries is computationally heavy. TPVFormer \[[19](https://arxiv.org/html/2312.17118v5#bib.bib19)\] proposes tri-perspective view representations to supplement vertical structural information, but this inevitably leads to information loss. VoxFormer \[[26](https://arxiv.org/html/2312.17118v5#bib.bib26)\] initializes sparse queries based on monocular depth prediction. Nevertheless, VoxFormer is not fully sparse as it still requires a sparse-to-dense MAE \[[13](https://arxiv.org/html/2312.17118v5#bib.bib13)\] module to complete the scene. Some methods emerged in the CVPR 2023 occupancy challenge \[[28](https://arxiv.org/html/2312.17118v5#bib.bib28), [40](https://arxiv.org/html/2312.17118v5#bib.bib40), [9](https://arxiv.org/html/2312.17118v5#bib.bib9)\], but none of them exploits a fully sparse design. In this paper, we make the first step to explore the fully sparse architecture for 3D occupancy prediction from camera-only inputs.

#### Sparse Architectures for 3D Vision.

Sparse architectures find widespread adoption in LiDAR-based reconstruction \[[48](https://arxiv.org/html/2312.17118v5#bib.bib48)\] and perception \[[7](https://arxiv.org/html/2312.17118v5#bib.bib7), [63](https://arxiv.org/html/2312.17118v5#bib.bib63), [60](https://arxiv.org/html/2312.17118v5#bib.bib60), [61](https://arxiv.org/html/2312.17118v5#bib.bib61)\], leveraging the inherent sparsity of point clouds. However, when it comes to vision-to-3D tasks, a direct adaptation is not feasible due to the absence of point cloud inputs. A prior work, SparseBEV \[[33](https://arxiv.org/html/2312.17118v5#bib.bib33)\], proposes a fully sparse architecture for camera-based 3D object detection. Nevertheless, directly adapting this approach is non-trivial because 3D object detection focuses on a sparse set of objects, whereas 3D occupancy requires dense predictions for each voxel. Consequently, designing a fully sparse architecture for 3D occupancy prediction remains a challenging task.

#### End-to-end 3D Reconstruction from Posed Images.

As a related task to 3D occupancy prediction, 3D reconstruction recovers the 3D geometry from multiple posed images. Recent methods focus on more compact and efficient end-to-end 3D reconstruction pipelines \[[39](https://arxiv.org/html/2312.17118v5#bib.bib39), [47](https://arxiv.org/html/2312.17118v5#bib.bib47), [2](https://arxiv.org/html/2312.17118v5#bib.bib2), [46](https://arxiv.org/html/2312.17118v5#bib.bib46), [10](https://arxiv.org/html/2312.17118v5#bib.bib10)\]. Atlas \[[39](https://arxiv.org/html/2312.17118v5#bib.bib39)\] extracts features from multi-view input images and maps them to 3D space to construct the truncated signed distance function \[[8](https://arxiv.org/html/2312.17118v5#bib.bib8)\]. NeuralRecon \[[47](https://arxiv.org/html/2312.17118v5#bib.bib47)\] directly reconstructs local surfaces as sparse TSDF volumes and uses a GRU-based TSDF fusion module to fuse features from previous fragments. VoRTX \[[46](https://arxiv.org/html/2312.17118v5#bib.bib46)\] utilizes transformers to address occlusion issues in multi-view images.

#### Mask Transformer.

Recently, unified segmentation models have been widely studied to handle semantic and instance segmentation concurrently. Cheng *et al*. first propose MaskFormer \[[6](https://arxiv.org/html/2312.17118v5#bib.bib6)\] for unified segmentation in terms of model architecture, loss functions, and training strategies. Mask2Former \[[5](https://arxiv.org/html/2312.17118v5#bib.bib5)\] then introduces masked attention, with restricted receptive fields on instance masks, for better performance. Later on, Mask3D \[[44](https://arxiv.org/html/2312.17118v5#bib.bib44)\] successfully extends the mask transformer for point cloud segmentation with state-of-the-art performance. OpenMask3D \[[49](https://arxiv.org/html/2312.17118v5#bib.bib49)\] further achieves the open-vocabulary 3D instance segmentation task and proposes a model for zero-shot 3D segmentation.

![Refer to caption](x3.png)

Figure 2: SparseOcc is a fully sparse architecture since it neither relies on dense 3D feature, nor has sparse-to-dense and global attention operations. The sparse voxel decoder reconstructs the sparse geometry of the scene, consisting of $K$ voxels ($K\ll W\times H\times D$). The mask transformer then uses $N$ sparse queries to predict the mask and label of each segment. SparseOcc can be easily extended to panoptic occupancy by replacing the semantic queries with instance queries.

## 3 SparseOcc

SparseOcc is a vision-centric occupancy model that only requires camera inputs. As shown in Fig. [2](https://arxiv.org/html/2312.17118v5#S2.F2 "Figure 2 ‣ Mask Transformer. ‣ 2 Related Work ‣ Fully Sparse 3D Occupancy Prediction"), SparseOcc has three modules: an image encoder consisting of an image backbone and FPN \[[29](https://arxiv.org/html/2312.17118v5#bib.bib29)\] to extract 2D features from multi-view images; a sparse voxel decoder (Sec. [3.1](https://arxiv.org/html/2312.17118v5#S3.SS1 "3.1 Sparse Voxel Decoder ‣ 3 SparseOcc ‣ Fully Sparse 3D Occupancy Prediction")) to predict sparse class-agnostic 3D occupancy with correlated embeddings from the image features; a mask transformer decoder (Sec [3.2](https://arxiv.org/html/2312.17118v5#S3.SS2 "3.2 Mask Transformer ‣ 3 SparseOcc ‣ Fully Sparse 3D Occupancy Prediction")) to distinguish semantics and instances in the sparse 3D space.

### 3.1 Sparse Voxel Decoder

Since 3D occupancy ground truth \[[50](https://arxiv.org/html/2312.17118v5#bib.bib50), [45](https://arxiv.org/html/2312.17118v5#bib.bib45), [57](https://arxiv.org/html/2312.17118v5#bib.bib57), [54](https://arxiv.org/html/2312.17118v5#bib.bib54)\] is a dense volume with dimensions $W\times H\times D$ (*e.g*., 200$\times$200$\times$16), existing methods typically build a dense 3D feature of shape $W\times H\times D\times C$, but suffer from computational overhead. In this paper, we argue that such dense representation is not necessary for occupancy prediction. As in our statistics, we find that over 90% of the voxels in the scene are free. This motivates us to explore a sparse 3D representation that only models the non-free areas of the scene, thereby saving computational resources.

#### Overall architecture.

Our designed sparse voxel decoder is shown in Fig. [3](https://arxiv.org/html/2312.17118v5#S3.F3 "Figure 3 ‣ Overall architecture. ‣ 3.1 Sparse Voxel Decoder ‣ 3 SparseOcc ‣ Fully Sparse 3D Occupancy Prediction"). In general, it follows a coarse-to-fine structure but only models the non-free regions. The decoder starts from a set of coarse voxel queries equally distributed in the 3D space (*e.g*., 25$\times$25). In each layer, we first upsample each voxel by 2$\times$, *e.g*., a voxel with size $d$ will be upsampled into 8 voxels with size $\frac{d}{2}$. Next, we estimate an occupancy score for each voxel and conduct pruning to remove useless voxel grids. Here we have two approaches for pruning: one is based on a threshold (*e.g*., only keeps score $>$ 0.5); the other is by top-$k$ selection. In our implementation, we simply keep voxels with top-$k$ occupancy scores for training efficiency. $k$ is a dataset-related parameter, obtained by counting the maximum number of non-free voxels in each sample at different resolutions. The voxel tokens after pruning will serve as the input for the next layer.

![Refer to caption](x4.png)

Figure 3: The sparse voxel decoder employs a coarse-to-fine pipeline with three layers. Within each layer, we utilize a transformer-like architecture for 3D-2D interaction. At the end of every layer, the voxel resolution is upsampled by a factor of 2$\times$, and probabilities of voxel occupancy are estimated.

#### Detailed design.

Within each layer, we use a transformer-like \[[52](https://arxiv.org/html/2312.17118v5#bib.bib52)\] architecture to handle voxel queries. The concrete architecture is inspired by SparseBEV \[[33](https://arxiv.org/html/2312.17118v5#bib.bib33)\], a detection method using a sparse scheme. To be specific, in layer $l$ with $K_{l-1}$ voxel queries described by 3D locations and a $C$-dim content vector, we first use self-attention to aggregate local and global features for those query voxels. Then, a linear layer is used to generate 3D sampling offsets $\{(\Delta x_{i},\Delta y_{i},\Delta z_{i})\}$ for each voxel query from the associated content vector. These sampling offsets are utilized to transform voxel queries to obtain reference points in global coordinates. We finally project those sampled reference points to multi-view image space for integrating image features by adaptive mixing \[[12](https://arxiv.org/html/2312.17118v5#bib.bib12), [51](https://arxiv.org/html/2312.17118v5#bib.bib51), [20](https://arxiv.org/html/2312.17118v5#bib.bib20)\]. In summary, our approach differs from SparseBEV by shifting the query formulation from pillars to 3D voxels. Other components such as self attention, adaptive sampling and mixing are directly borrowed.

#### Temporal modeling.

Previous dense occupancy methods \[[27](https://arxiv.org/html/2312.17118v5#bib.bib27), [16](https://arxiv.org/html/2312.17118v5#bib.bib16)\] typically warp the history BEV/3D feature to the current timestamp, and use deformable attention \[[64](https://arxiv.org/html/2312.17118v5#bib.bib64)\] or 3D convolutions to fuse temporal information. However, this approach is not directly applicable in our case due to the sparse nature of our 3D features. To handle this, we leverage the flexibility of the aforementioned global sampled reference points by warping them to previous timestamps to sample history multi-view image features. The sampled multi-frame features are stacked and aggregated by adaptive mixing so as for temporal modeling.

#### Supervision.

We compute loss for the sparsified voxels from each layer. We use binary cross entropy (BCE) loss as the supervision, given that we are reconstructing a class-agnostic sparse occupancy space. Only the kept sparse voxels are supervised, while the discarded regions during pruning in earlier stages are ignored.

Moreover, due to the severe class imbalance, the model can be easily dominated by categories with a large proportion, such as the ground, thereby ignoring other important elements in the scene, such as cars, people, etc. Therefore, voxels belonging to different classes are assigned with different loss weights. For example, voxels belonging to class $c$ are assigned with a loss weight of:

$$ \displaystyle w_{c}=\frac{\sum_{i=1}^{C}M_{i}}{M_{c}}, $$

where $M_{i}$ is the number of voxels belonging to the $i$-th class in ground truth.

### 3.2 Mask Transformer

Our mask transformer is inspired by Mask2Former \[[5](https://arxiv.org/html/2312.17118v5#bib.bib5)\], which uses $N$ sparse semantic/instance queries decoupled by binary mask queries $\mathbf{Q}_{m}\in\[0,1\]^{N\times K}$ and content vectors $\mathbf{Q}_{c}\in\mathbb{R}^{N\times C}$. The mask transformer consists of three steps: multi-head self attention (MHSA), mask-guided sparse sampling, and adaptive mixing. MHSA is used for the interaction between different queries as the common practice. Mask-guided sparse sampling and adaptive mixing are responsible for the interaction between queries and 2D image features.

#### Mask-guided sparse sampling.

A simple baseline of mask transformer is to use the masked cross-attention module in Mask2Former. However, it attends to all positions of the key, with unbearable computations. Here, we design a simple alternative. We first randomly select a set of 3D points within the mask predicted by the previous ($l-1$)-th Transformer decoder layer. Then, we project those 3D points to multi-view images and extract their features by bilinear interpolation. Besides, our sparse sampling mechanism makes the temporal modeling easier by simply warping the sampling points (as done in the sparse voxel decoder).

#### Prediction.

For class prediction, we apply a linear classifier with a sigmoid activation based on the query embeddings $\mathbf{Q}_{c}$. For mask prediction, the query embeddings are converted to mask embeddings by an MLP. The mask embeddings $\mathbf{M}\in\mathbb{R}^{Q\times C}$ have the same shape as query embeddings $\mathbf{Q}_{c}$ and are dot-producted with the sparse voxel embeddings $\mathbf{V}\in\mathbb{R}^{K\times C}$ to produce mask predictions. Thus, the prediction space of our mask transformer is constrained to the sparsified 3D space from the sparse voxel decoder, rather than the full 3D scene. The mask predictions will serve as the mask queries $\mathbf{Q}_{m}$ for the next transformer layer.

#### Supervision.

The reconstruction result from the sparse voxel decoder may not be reliable, as it may overlook or inaccurately detect certain elements. Thus, supervising the mask transformer presents certain challenges since its predictions are confined within this unreliable space. In cases of missed detection, where some ground truth segments are absent in the predicted sparse occupancy, we opt to discard these segments to prevent confusion. As for inaccurately detected elements, we simply categorize them as an additional “no object” category.

#### Loss Functions.

Following MaskFormer \[[6](https://arxiv.org/html/2312.17118v5#bib.bib6)\], we match the ground truth with the predictions using Hungarian matching. Focal loss $L_{focal}$ is used for classification, while a combination of DICE loss \[[38](https://arxiv.org/html/2312.17118v5#bib.bib38)\] $L_{dice}$ and BCE mask loss $L_{mask}$ is used for mask prediction. Thus, the total loss of SparseOcc is composed of four parts:

$$ L=L_{focal}+L_{mask}+L_{dice}+L_{occ}, $$

where $L_{occ}$ is the loss of sparse voxel decoder.

## 4 Ray-level mIoU

### 4.1 Revisiting the Voxel-level mIoU

![Refer to caption](x5.png)

(a)

Figure 4: Visualization of the discrepancy between qualitative and quantitative results. We observe that training existing dense occupancy methods (*e.g*. BEVFormer) with a visible mask results in a thick surface, leading to an unreasonably inflated improvement in the current mIoU metrics. In contrast, our new RayIoU metrics provide a more accurate reflection of model performance.

The Occ3D dataset \[[50](https://arxiv.org/html/2312.17118v5#bib.bib50)\], along with its proposed evaluation metrics, are widely recognized as benchmarks in this field. The ground truth occupancy is reconstructed from LiDAR point clouds, and the mean Intersection over Union (mIoU) at the voxel level is employed to assess performance. Due to factors such as distance and occlusion, the accumulated point clouds are not perfect. Some areas unscanned by LiDAR are marked as free, resulting in fragmented instances. This raises the problem of label inconsistency. To solve this problem, Occ3D uses a binary visible mask that indicates whether a voxel is observed in the current camera view. Only the observed voxels contribute to evaluation.

However, we found that solely calculating mIoU on the observed voxel positions remains vulnerable and can be hacked by predicting a thicker surface. Dense methods (*e.g*., BEVFormer \[[27](https://arxiv.org/html/2312.17118v5#bib.bib27)\]) can easily achieve this by training with the visible mask. During training, the area behind the surface lacks supervision, causing the model to fill it with duplicated predictions, resulting in a thicker surface. As an example, consider BEVFormer, which generates a thick and noisy surface when trained with the visible mask (see Fig. [4](https://arxiv.org/html/2312.17118v5#S4.F4 "Figure 4 ‣ 4.1 Revisiting the Voxel-level mIoU ‣ 4 Ray-level mIoU ‣ Fully Sparse 3D Occupancy Prediction")). Despite this, its performance exhibits an unreasonably inflated improvement (+5$\sim$15 mIoU) under the current evaluation protocol.

The misalignment between qualitative and quantitative results is caused by the inconsistent penalty along the depth direction. A toy example in Fig. [5](https://arxiv.org/html/2312.17118v5#S4.F5 "Figure 5 ‣ 4.1 Revisiting the Voxel-level mIoU ‣ 4 Ray-level mIoU ‣ Fully Sparse 3D Occupancy Prediction") reveals several issues with the current metrics:

1.  1.

If the model fills all areas behind the surface, it inconsistently penalizes depth predictions. The model can obtain a higher IoU by filling all areas behind the surface and predicting a closer depth. This thick surface issue is very common in dense models trained with visible masks or 2D supervision.

2.  2.

If the predicted occupancy represents a thin surface, the penalty becomes overly strict. Even a deviation of just one voxel results in an IoU of zero.

3.  3.

The visible mask only considers the visible area at the current moment, reducing occupancy prediction to a depth estimation task and overlooking the scene completion ability.

![Refer to caption](x6.png)

Figure 5: Illustration of inconsistent depth penalties caused by current metrics. Consider a scenario where we have a wall in front of us, with a ground-truth distance of $d$ and a thickness of $d_{v}$. When the prediction has a thickness of $d_{p}\gg d_{v}$, we encounter an inconsistent penalty along depth. Specifically, if the predicted wall is $d_{v}$ farther than the ground truth (total distance $d+d_{v}$), its IoU will be zero. Conversely, if the predicted wall is $d_{v}$ closer than the ground truth (total distance $d-d_{v}$), the IoU remains at 0.5. This occurs because all voxels behind the surface are filled with duplicated predictions. Similarly, when the predicted depth is $d-2d_{v}$, the resulting IoU is $\frac{1}{3}$, and so forth.

### 4.2 Mean IoU by Ray Casting

![Refer to caption](extracted/5741767/asserts/ray_cover/eval_1f_lidar_0985_gt.jpg)

(a)

![Refer to caption](extracted/5741767/asserts/ray_cover/eval_1f_equal_0985_gt.jpg)

(b)

![Refer to caption](extracted/5741767/asserts/ray_cover/eval_8f_equal_0985_gt.jpg)

(c)

Figure 6: Covered area of RayIoU. (a) The raw LiDAR ray samples are unbalanced at different distances. (b) We resample the rays to balance the weight on distance. (c) To investigate the performance of scene completion, we propose evaluating occupancy in the visible area on a wide time span, by casting rays on visited waypoints.

To address the above issues, we propose a new evaluation metric: Ray-level mIoU (RayIoU for short). In RayIoU, the set elements are query rays rather than voxels. We emulate LiDAR behavior by projecting query rays into the predicted 3D occupancy volume. For each query ray, we compute the distance it travels before intersecting any surface and retrieve the corresponding class label. We then apply the same procedure to the ground-truth occupancy to obtain the ground-truth depth and class label. In case a ray does not intersect with any voxel present in the ground truth, it will be excluded from the evaluation process.

As shown in Fig. [6](https://arxiv.org/html/2312.17118v5#S4.F6 "Figure 6 ‣ 4.2 Mean IoU by Ray Casting ‣ 4 Ray-level mIoU ‣ Fully Sparse 3D Occupancy Prediction") (a), the raw LiDAR rays in a real dataset tend to be unbalanced from near to far. Thus, we resample the rays to achieve a balanced distribution across different distances (Fig. [6](https://arxiv.org/html/2312.17118v5#S4.F6 "Figure 6 ‣ 4.2 Mean IoU by Ray Casting ‣ 4 Ray-level mIoU ‣ Fully Sparse 3D Occupancy Prediction") (b)). In the near field, we modify the ray channels to achieve equal-distant spacing when projected onto the ground plane. In the far field, we increase the angular resolution of the ray channels to ensure a more uniform data density across varying ranges. Moreover, our query ray can originate from the LiDAR position at the current, past, or future moments of the ego path. Temporal casting (Fig. [6](https://arxiv.org/html/2312.17118v5#S4.F6 "Figure 6 ‣ 4.2 Mean IoU by Ray Casting ‣ 4 Ray-level mIoU ‣ Fully Sparse 3D Occupancy Prediction") (c)) allows us to evaluate scene completion performance while maintaining a well-posed task.

A query ray is classified as a true positive (TP) if the class labels coincide and the L1 error between the ground-truth depth and the predicted depth is less than a certain threshold (*e.g*., 2m). Let $C$ be the number of classes, then RayIoU is calculated as follows:

$$ \text{RayIoU}=\frac{1}{C}\displaystyle\sum_{c=1}^{C}\frac{\text{TP}_{c}}{\text {TP}_{c}+\text{FP}_{c}+\text{FN}_{c}}, $$

where $\text{TP}_{c}$, $\text{FP}_{c}$ and $\text{FN}_{c}$ correspond to the number of true positive, false positive, and false negative predictions for class $c_{i}$.

RayIoU addresses all three of the aforementioned problems:

1.  1.

Since the query ray calculates the distance to the first voxel it touches, the model cannot obtain a higher IoU by predicting a thicker surface.

2.  2.

RayIoU determines true positives based on a distance threshold, which mitigates the overly strict nature of voxel-level mIoU.

3.  3.

The query ray can originate from any position in the scene. This flexibility allows RayIoU to consider the model’s scene completion ability, preventing the reduction of occupancy estimation to mere depth prediction.

## 5 Experiments

We evaluate our model on the Occ3D-nuScenes \[[50](https://arxiv.org/html/2312.17118v5#bib.bib50)\] dataset. Occ3D-nuScenes is based on the nuScenes \[[3](https://arxiv.org/html/2312.17118v5#bib.bib3)\] dataset, which consists of large-scale multimodal data collected from 6 surround-view cameras, 1 lidar and 5 radars. The dataset has 1000 videos in total and is split into 700/150/150 videos for training/validation/testing. Each video has roughly 20s duration and the key samples are annotated every 0.5s.

We use the proposed RayIoU to evaluate the semantic segmentation performance. The query rays originate from 8 LiDAR positions of the ego path. We calculate RayIoU under three distance thresholds: 1, 2 and 4 meters. The final ranking metric is averaged over these distance thresholds.

Table 1: 3D occupancy prediction performance on Occ3D-nuScenes \[[50](https://arxiv.org/html/2312.17118v5#bib.bib50)\]. We use RayIoU to compare our SparseOcc with other methods. “8f” and “16f” mean fusing temporal information from 8 or 16 frames. SparseOcc outperforms all existing methods under a weaker setting.

| Method | Backbone | Input Size | Epoch | RayIoU | RayIoU1m, 2m, 4m | mIoU | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BEVFormer (4f) [27] | R101 | 1600$\times$900 | 24 | 32.4 | 26.1 | 32.9 | 38.0 | 39.2 | 3.0 |
| RenderOcc [40] | Swin-B | 1408$\times$512 | 12 | 19.5 | 13.4 | 19.6 | 25.5 | 24.4 | - |
| SimpleOcc [11] | R101 | 672$\times$336 | 12 | 22.5 | 17.0 | 22.7 | 27.9 | 31.8 | 9.7 |
| BEVDet-Occ (2f) [15] | R50 | 704$\times$256 | 90 | 29.6 | 23.6 | 30.0 | 35.1 | 36.1 | 2.6 |
| BEVDet-Occ-Long (8f) | R50 | 704$\times$384 | 90 | 32.6 | 26.6 | 33.1 | 38.2 | 39.3 | 0.8 |
| FB-Occ (16f) [28] | R50 | 704$\times$256 | 90 | 33.5 | 26.7 | 34.1 | 39.7 | 39.1 | 10.3 |
| SparseOcc (8f) | R50 | 704$\times$256 | 24 | 34.0 | 28.0 | 34.7 | 39.4 | 30.1 | 17.3 |
| SparseOcc (16f) | R50 | 704$\times$256 | 24 | 35.1 | 29.1 | 35.8 | 40.3 | 30.6 | 12.5 |
| SparseOcc (16f) | R50 | 704$\times$256 | 48 | 36.1 | 30.2 | 36.8 | 41.2 | 30.9 | 12.5 |

### 5.1 Implementation Details

We implement our model using PyTorch \[[41](https://arxiv.org/html/2312.17118v5#bib.bib41)\]. Following previous methods, we adopt ResNet-50 \[[14](https://arxiv.org/html/2312.17118v5#bib.bib14)\] as the image backbone. The mask transformer consists of 3 layers with shared weights across different layers. In our main experiments, we employ semantic queries where each query corresponds to a semantic class, rather than an instance. The ray casting module in RayIoU is implemented based on the codebase of \[[21](https://arxiv.org/html/2312.17118v5#bib.bib21)\].

During training, we use the AdamW \[[36](https://arxiv.org/html/2312.17118v5#bib.bib36)\] optimizer with a global batch size of 8. The initial learning rate is set to $2\times 10^{-4}$ and is decayed with cosine annealing policy. For all experiments, we train our models for 24 epochs. FPS is measured on a Tesla A100 GPU with the PyTorch fp32 backend.

### 5.2 Main Results

In Tab. [1](https://arxiv.org/html/2312.17118v5#S5.T1 "Table 1 ‣ 5 Experiments ‣ Fully Sparse 3D Occupancy Prediction") and Fig. [1](https://arxiv.org/html/2312.17118v5#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Fully Sparse 3D Occupancy Prediction") (b), we compare SparseOcc with previous state-of-the-art methods on the validation split of Occ3D-nuScenes. Despite under a weaker setting (ResNet-50 \[[14](https://arxiv.org/html/2312.17118v5#bib.bib14)\], 8 history frames, and input image resolution of 704 $\times$ 256), SparseOcc significantly outperforms previous methods including FB-Occ, the winner of CVPR 2023 occupancy challenge, with many complicated designs including forward-backward view transformation, depth net, joint depth and semantic pre-training, and so on. SparseOcc achieves better results (+1.6 RayIoU) while being much faster and simpler than FB-Occ, which demonstrates the superiority of our solution.

We further provide qualitative results in Fig. [7](https://arxiv.org/html/2312.17118v5#S5.F7 "Figure 7 ‣ 5.2 Main Results ‣ 5 Experiments ‣ Fully Sparse 3D Occupancy Prediction"). Both BEVDet-Occ and FB-Occ are dense methods and make many redundant predictions behind the surface. In contrast, SparseOcc discards over 90% of voxels while still effectively modeling the geometry of the scene and capturing fine-grained details.

![Refer to caption](x7.png)

(a)

Figure 7: Visualized comparison of semantic occupancy prediction. Despite discarding over 90% of voxels, our SparseOcc effectively models the geometry of the scene and captures fine-grained details (*e.g*., the yellow-marked traffic cone in the bottom row).

### 5.3 Ablations

In this section, we conduct ablations on the validation split of Occ3D-nuScenes to confirm the effectiveness of each module. By default, we use the single frame version of SparseOcc as the baseline. The choice for our model is made bold.

Table 2: Sparse voxel decoder vs. dense voxel decoder. Our sparse voxel decoder achieves nearly 4$\times$ faster inference speed than the dense counterparts.

| Voxel Decoder | RayIoU | RayIoU1m | RayIoU2m | RayIoU4m | FPS |
| --- | --- | --- | --- | --- | --- |
| Dense coarse-to-fine | 29.9 | 24.0 | 30.4 | 35.4 | 6.3 |
| Dense patch-based | 25.8 | 20.4 | 26.0 | 30.9 | 7.8 |
| Sparse coarse-to-fine | 29.9 | 23.9 | 30.5 | 35.2 | 24.0 |

#### Sparse voxel decoder vs. dense voxel decoder.

In Tab. [2](https://arxiv.org/html/2312.17118v5#S5.T2 "Table 2 ‣ 5.3 Ablations ‣ 5 Experiments ‣ Fully Sparse 3D Occupancy Prediction"), we compare our sparse voxel decoder to the dense counterparts. Here, we implement two baselines, and both of them output a dense feature map with shape as 200$\times$200$\times$16$\times$$C$. The first baseline is a coarse-to-fine architecture without pruning empty voxels. In this baseline, we also replace self-attention with 3D convolution and use 3D deconvolution to upsample predictions. The other baseline is a patch-based architecture by dividing the 3D space into a small number of patches as PETRv2 \[[35](https://arxiv.org/html/2312.17118v5#bib.bib35)\] for BEV segmentation. We use 25$\times$25$\times$2 = 1250 queries and each one of them corresponds to a specific patch of shape 8$\times$8$\times$8. A stack of deconvolution layers are used to lift the coarse queries to a full-resolution 3D volume.

As we can see from the table, the dense coarse-to-fine baseline achieves a good performance of 29.9 RayIoU but with a slow inference speed of 6.3 FPS. The patch-based one is slightly faster with 7.8 FPS inference speed but with a severe performance drop by 4.1 RayIoU. Instead, our sparse voxel decoder produces sparse 3D features in the shape of $K\times C$ (where $K$ = 32000 $\ll$ 200$\times$200$\times$16), achieving an inference speed that is nearly 4$\times$ faster than the counterparts without compromising performance. This demonstrates the necessity and effectiveness of our sparse design.

Table 3: Ablation of mask transformer (MT) and the cross attention module in MT. Mask-guided sparse sampling is stronger and faster than the dense cross attention.

| MT | Cross Attention | RayIoU | RayIoU1m | RayIoU2m | RayIoU4m | FPS |
| --- | --- | --- | --- | --- | --- | --- |
| - | - | 27.0 | 20.3 | 27.5 | 33.1 | 29.0 |
| $\surd$ | Dense cross attention | 28.7 | 22.9 | 29.3 | 33.8 | 16.2 |
| $\surd$ | Sparse sampling | 25.8 | 20.5 | 26.2 | 30.8 | 24.0 |
| $\surd$ | + Mask-guided | 29.2 | 23.4 | 29.8 | 34.5 | 24.0 |

#### Mask Transformer.

In Tab. [3](https://arxiv.org/html/2312.17118v5#S5.T3 "Table 3 ‣ Sparse voxel decoder vs. dense voxel decoder. ‣ 5.3 Ablations ‣ 5 Experiments ‣ Fully Sparse 3D Occupancy Prediction"), we ablate the effectiveness of the mask transformer. The first row is a simple per-voxel baseline which directly predicts semantics from the sparse voxel decoder using a stack of MLPs. Introducing mask transformer with vanilla cross attention (as it is the common practice in MaskFormer and Mask3D) gives a performance boost of 1.7 RayIoU, but inevitably slows down the inference speed as it attends to all locations in an image. Therefore, to speed up the dense cross-attention pipeline, we adopt a sparse sampling mechanism which brings a 50% reduction in inference time. By further introducing the predicted masks to guide the generation of sampling points, we finally achieve 29.2 RayIoU with 24 FPS.

![Refer to caption](x8.png)

(a)

![Refer to caption](x9.png)

(b)

![Refer to caption](x10.png)

(c)

Figure 8: Ablations on voxel sparsity and temporal modeling. (a) The optimal performance occurs when $k$ is set to 32000 (5% sparsity). (b) Top-$k$ can also be substituted with thresholding, *e.g*., voxels scoring less than a certain threshold will be pruned. (c) The performance continues to increase with the number of frames, but it starts to saturate after 12 frames.

#### Is a limited set of voxels sufficient to cover the scene?

In this study, we delve deeper into the impact of voxel sparsity on final performance. To investigate this, we systematically ablate the value of $k$ in Fig. [8](https://arxiv.org/html/2312.17118v5#S5.F8 "Figure 8 ‣ Mask Transformer. ‣ 5.3 Ablations ‣ 5 Experiments ‣ Fully Sparse 3D Occupancy Prediction") (a). Starting from a modest value of 16k, we observe that the optimal performance occurs when $k$ is set to 32k $\sim$ 48k, which is only 5% $\sim$ 7.5% of the total number of dense voxels (200$\times$200$\times$16 = 640000). Surprisingly, further increasing $k$ does not yield any performance improvements; instead, it introduces noise. Thus, our findings suggest that a $\sim$5% sparsity level is sufficient. Keep increasing the density will reduce both accuracy and speed.

Pruning by top-$k$ is simple and effective, but it is related to specific dataset. In real world, we can substitute top-$k$ with a thresholding method. Voxels scoring less than a given threshold (*e.g*., 0.7) will be pruned. Thresholding achieves similar performance to top-$k$ (see Fig. [8](https://arxiv.org/html/2312.17118v5#S5.F8 "Figure 8 ‣ Mask Transformer. ‣ 5.3 Ablations ‣ 5 Experiments ‣ Fully Sparse 3D Occupancy Prediction") (b)), and has the ability to generalize to different scenes.

#### Temporal modeling.

In Fig. [8](https://arxiv.org/html/2312.17118v5#S5.F8 "Figure 8 ‣ Mask Transformer. ‣ 5.3 Ablations ‣ 5 Experiments ‣ Fully Sparse 3D Occupancy Prediction") (c), we validate the effectiveness of temporal fusion. We can see that the temporal modeling of SparseOcc is very effective, with performance steadily increasing as the number of frames increases. The performance peaks at 12 frames and then saturates. However, the inference speed drops rapidly as the sampling points need to interact with every frame.

### 5.4 More Studies

Table 4: To verify the effect of the visible mask, wo provide per-class RayIoU of BEVFormer and FB-Occ. $\dagger$ uses the visible mask during training. We find that training with the visible mask hurts the performance of background classes such as drivable surface, terrian and sidewalk.

|  |  |  | Per-class RayIoU |  |
| --- | --- | --- | --- | --- |
| Method | mIoU | RayIoU | others | barrier | bicycle | bus | car | cons. veh. | motor. | pedes. | tfc. cone | trailer | truck | drv. surf. | other flat | sidewalk | terrain | manmade | vegetation |
| BEVFormer | 23.7 | 33.7 | 5.0 | 42.2 | 18.2 | 55.2 | 57.1 | 22.7 | 21.3 | 31.0 | 27.1 | 30.7 | 49.4 | 58.4 | 30.4 | 29.4 | 31.7 | 36.3 | 26.5 |
| BEVFormer $\dagger$ | 39.2 | 32.4 | 6.4 | 44.8 | 24.0 | 55.2 | 56.7 | 21.0 | 29.8 | 33.5 | 26.8 | 27.9 | 49.5 | 45.8 | 18.7 | 22.4 | 18.5 | 39.1 | 29.8 |
| FB-Occ | 27.9 | 35.6 | 10.5 | 44.8 | 25.6 | 55.6 | 51.7 | 22.6 | 27.2 | 34.3 | 30.3 | 23.7 | 44.1 | 65.5 | 33.3 | 31.4 | 32.5 | 39.6 | 33.3 |
| FB-Occ $\dagger$ | 39.1 | 33.5 | 5.0 | 44.9 | 26.2 | 59.7 | 55.1 | 27.9 | 29.1 | 34.3 | 29.6 | 29.1 | 50.5 | 44.4 | 22.4 | 21.5 | 19.5 | 39.3 | 31.1 |

![Refer to caption](x11.png)

Figure 9: Why does the performance of background classes, such as drivable surfaces, degrade when using the visible mask during training? We provide a visualization of the drivable surface as predicted by FB-Occ. Here, “FB w/ mask” and “FB wo/ mask” denote training with and without the visible mask, respectively. We observe that “FB w/ mask” tends to predict a higher and thicker road surface, resulting in significant depth errors along a ray. In contrast, “FB wo/ mask” predicts a road surface that is both accurate and consistent.

#### The effect of training with visible masks.

Interestingly, we observed a peculiar phenomenon. Under the traditional voxel-level mIoU metric, dense methods can significantly benefit from disregarding the non-visible voxels during training. These non-visible voxels are indicated by a binary visible mask provided by the Occ3D-nuScenes dataset. However, we find that this strategy actually impairs performance under our new RayIoU metric. For instance, we train two variants of BEVFormer: one uses the visible mask during training, and the other does not. As shown in Tab. [4](https://arxiv.org/html/2312.17118v5#S5.T4 "Table 4 ‣ 5.4 More Studies ‣ 5 Experiments ‣ Fully Sparse 3D Occupancy Prediction"), the former scores 15 points higher than the latter on the voxel-based mIoU, but it scores 1 point lower on RayIoU. This phenomenon is also observed on FB-Occ.

To further explore this, we present the per-class RayIoU in Tab. [4](https://arxiv.org/html/2312.17118v5#S5.T4 "Table 4 ‣ 5.4 More Studies ‣ 5 Experiments ‣ Fully Sparse 3D Occupancy Prediction"). The table reveals that training with the visible mask enhances performance for most foreground classes such as bus, bicycle, and truck. However, it negatively impacts background classes like drivable surface and terrain.

This observation raises a further question: Why does the performance of the background category degrade? To address this, we offer a visual comparison of the depth errors and height maps of the predicted drivable surface from FB-Occ in Fig. [9](https://arxiv.org/html/2312.17118v5#S5.F9 "Figure 9 ‣ 5.4 More Studies ‣ 5 Experiments ‣ Fully Sparse 3D Occupancy Prediction"), both with and without the use of visible mask during training. The figure illustrates that training with visible masks results in a thicker and higher ground prediction, leading to substantial depth errors in distant areas. Conversely, models trained without the visible mask predict depth with greater accuracy.

From these observations, we derive some valuable insights: ignoring non-visible voxels during training benefits foreground classes by resolving the issue of ambiguous labeling of unscanned voxels. However, it also compromises the accuracy of depth estimation, as models tend to predict a thicker and closer surface. We hope that our findings will benefit future research.

#### Panoptic occupancy.

We then show that SparseOcc can be easily extended for panoptic occupancy prediction, a task derived from panoptic segmentation that segments images to not only semantically meaningful regions but also to detect and distinguish individual instances. Compared to panoptic segmentation, panoptic occupancy prediction requires the model to have geometric awareness in order to construct the 3D scene for segmentation. By additionally introducing instance queries to the mask transformer, we seamlessly achieve the first fully sparse panoptic occupancy prediction framework using camera-only inputs.

Firstly, we utilize the ground-truth bounding boxes from the 3D object detection task to generate the panoptic occupancy ground truth. Specifically, we define eight instance categories (including car, truck, construction vehicle, bus, trailer, motorcycle, bicycle, pedestrian) and ten staff categories (including terrain, manmade, vegetation, etc). Each instance segment is identified by grouping the voxels inside the bounding box based on an existing semantic occupancy benchmark, such as Occ3D-nuScenes.

We then design RayPQ based on the well-known panoptic quality (PQ) \[[22](https://arxiv.org/html/2312.17118v5#bib.bib22)\] metric, which is defined as the multiplication of *segmentation quality* (SQ) and *recognition quality* (RQ):

$$ {\text{PQ}}=\underbrace{\frac{\sum_{(p,g)\in\mathit{TP}}\text{IoU}(p,g)}{ \vphantom{\frac{1}{2}}|\mathit{TP}|}}_{\text{segmentation quality (SQ)}}\times \underbrace{\frac{|\mathit{TP}|}{|\mathit{TP}|+\frac{1}{2}|\mathit{FP}|+\frac{ 1}{2}|\mathit{FN}|}}_{\text{recognition quality (RQ) }}\,, $$

where the definition of true positive (TP) is the same as that in RayIoU. The threshold of IoU between prediction $p$ and ground-truth $g$ is set to 0.5.

Table 5: Panoptic occupancy prediction performance on Occ3D-nuScenes.

| Method | Backbone | Input Size | Epoch | RayPQ | RayPQ1m | RayPQ2m | RayPQ4m |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SparseOcc | R50 | 704$\times$256 | 24 | 14.1 | 10.2 | 14.5 | 17.6 |

![Refer to caption](x12.png)

(a)

Figure 10: Panoptic occupancy prediction. Different instances are distinguished by colors. Our model can capture fine-grained objects and road structures simultaneously.

In Tab. [5](https://arxiv.org/html/2312.17118v5#S5.T5 "Table 5 ‣ Panoptic occupancy. ‣ 5.4 More Studies ‣ 5 Experiments ‣ Fully Sparse 3D Occupancy Prediction"), we report the performance of SparseOcc on panoptic occupancy benchmark. Similar to RayIoU, we calculate RayPQ under three distance thresholds: 1, 2 and 4 meters. SparseOcc achieves an averaged RayPQ of 14.1. The visualizations are presented in Fig. [10](https://arxiv.org/html/2312.17118v5#S5.F10 "Figure 10 ‣ Panoptic occupancy. ‣ 5.4 More Studies ‣ 5 Experiments ‣ Fully Sparse 3D Occupancy Prediction").

### 5.5 Limitations

#### Accumulative errors.

In order to implement a fully sparse architecture, we discard a large number of empty voxels in the early stages. However, empty voxels that are mistakenly discarded cannot be recovered in subsequent stages. Moreover, the prediction of the mask transformer is constrained within a space predicted by the sparse voxel decoder. Some ground-truth instances do not appear in this unreliable space, leading to inadequate training of the mask transformer.

## 6 Conclusion

In this paper, we proposed a fully sparse occupancy network, named SparseOcc, which neither relies on dense 3D feature, nor has sparse-to-dense and global attention operations. We also created RayIoU, a ray-level metric for occupancy evaluation, eliminating the inconsistency flaws of previous metric. Experiments show that SparseOcc achieves the state-of-the-art performance on the Occ3D-nuScenes dataset for both speed and accuracy. We hope this exciting result will attract more attention to the fully sparse 3D occupancy paradigm.

## Acknowledgements

We thank the anonymous reviewers for their suggestions that make this work better. This work is supported by the National Key R$\&$D Program of China (No. 2022ZD0160900), the National Natural Science Foundation of China (No. 62076119, No. 61921006), the Fundamental Research Funds for the Central Universities (No. 020214380119), and the Collaborative Innovation Center of Novel Software Technology and Industrialization.

## References

-   \[1\] Tesla AI Day. [https://www.youtube.com/watch?v=j0z4FweCy4M](https://www.youtube.com/watch?v=j0z4FweCy4M) (2021) -   \[2\] Bozic, A., Palafox, P., Thies, J., Dai, A., Nießner, M.: Transformerfusion: Monocular rgb scene reconstruction using transformers. In: NeurIPS (2021) -   \[3\] Caesar, H., Bankiti, V., Lang, A.H., Vora, S., Liong, V.E., Xu, Q., Krishnan, A., Pan, Y., Baldan, G., Beijbom, O.: nuscenes: A multimodal dataset for autonomous driving. In: CVPR (2020) -   \[4\] Cao, A.Q., de Charette, R.: Monoscene: Monocular 3d semantic scene completion. In: CVPR (2022) -   \[5\] Cheng, B., Misra, I., Schwing, A.G., Kirillov, A., Girdhar, R.: Masked-attention mask transformer for universal image segmentation. In: CVPR (2022) -   \[6\] Cheng, B., Schwing, A., Kirillov, A.: Per-pixel classification is not all you need for semantic segmentation. In: NeurIPS (2021) -   \[7\] Choy, C., Gwak, J., Savarese, S.: 4d spatio-temporal convnets: Minkowski convolutional neural networks. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 3075–3084 (2019) -   \[8\] Curless, B., Levoy, M.: A volumetric method for building complex models from range images. In: SIGGRAPH (1996) -   \[9\] Ding, Y., Huang, L., Zhong, J.: Multi-scale occ: 4th place solution for Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition 2023 3d occupancy prediction challenge. arXiv preprint arXiv:2306.11414 (2023) -   \[10\] Feng, Z., Yang, L., Guo, P., Li, B.: Cvrecon: Rethinking 3d geometric feature learning for neural reconstruction. In: ICCV (2023) -   \[11\] Gan, W., Mo, N., Xu, H., Yokoya, N.: A comprehensive framework for 3d occupancy estimation in autonomous driving. IEEE Transactions on Intelligent Vehicles pp. 1–19 (2024) -   \[12\] Gao, Z., Wang, L., Han, B., Guo, S.: Adamixer: A fast-converging query-based object detector. In: CVPR (2022) -   \[13\] He, K., Chen, X., Xie, S., Li, Y., Dollár, P., Girshick, R.: Masked autoencoders are scalable vision learners. In: CVPR (2022) -   \[14\] He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: CVPR (2016) -   \[15\] Huang, J., Huang, G.: Bevdet4d: Exploit temporal cues in multi-camera 3d object detection. arXiv preprint arXiv:2203.17054 (2022) -   \[16\] Huang, J., Huang, G., Zhu, Z., Du, D.: Bevdet: High-performance multi-camera 3d object detection in bird-eye-view. arXiv preprint arXiv:2112.11790 (2021) -   \[17\] Huang, L., Li, Z., Sima, C., Wang, W., Wang, J., Qiao, Y., Li, H.: Leveraging vision-centric multi-modal expertise for 3d object detection. In: NeurIPS (2024) -   \[18\] Huang, L., Wang, H., Zeng, J., Zhang, S., Cao, L., Ji, R., Yan, J., Li, H.: Geometric-aware pretraining for vision-centric 3d object detection. arXiv preprint arXiv:2304.03105 (2023) -   \[19\] Huang, Y., Zheng, W., Zhang, Y., Zhou, J., Lu, J.: Tri-perspective view for vision-based 3d semantic occupancy prediction. In: CVPR (2023) -   \[20\] Jia, X., De Brabandere, B., Tuytelaars, T., Gool, L.V.: Dynamic filter networks. In: NeurIPS (2016) -   \[21\] Khurana, T., Hu, P., Held, D., Ramanan, D.: Point cloud forecasting as a proxy for 4d occupancy forecasting. In: CVPR (2023) -   \[22\] Kirillov, A., He, K., Girshick, R., Rother, C., Dollár, P.: Panoptic segmentation. In: CVPR (2019) -   \[23\] Lang, A.H., Vora, S., Caesar, H., Zhou, L., Yang, J., Beijbom, O.: Pointpillars: Fast encoders for object detection from point clouds. In: CVPR (2019) -   \[24\] Li, H., Li, Y., Wang, H., Zeng, J., Cai, P., Xu, H., Lin, D., Yan, J., Xu, F., Xiong, L., Wang, J., Zhu, F., Yan, K., Xu, C., Wang, T., Mu, B., Ren, S., Peng, Z., Qiao, Y.: Open-sourced data ecosystem in autonomous driving: the present and future. arXiv preprint arXiv:2312.03408 (2023) -   \[25\] Li, H., Sima, C., Dai, J., Wang, W., Lu, L., Wang, H., Zeng, J., Li, Z., Yang, J., Deng, H., et al.: Delving into the devils of bird’s-eye-view perception: A review, evaluation and recipe. IEEE TPAMI (2023) -   \[26\] Li, Y., Yu, Z., Choy, C., Xiao, C., Alvarez, J.M., Fidler, S., Feng, C., Anandkumar, A.: Voxformer: Sparse voxel transformer for camera-based 3d semantic scene completion. In: CVPR (2023) -   \[27\] Li, Z., Wang, W., Li, H., Xie, E., Sima, C., Lu, T., Qiao, Y., Dai, J.: Bevformer: Learning bird’s-eye-view representation from multi-camera images via spatiotemporal transformers. In: ECCV (2022) -   \[28\] Li, Z., Yu, Z., Austin, D., Fang, M., Lan, S., Kautz, J., Alvarez, J.M.: Fb-occ: 3d occupancy prediction based on forward-backward view transformation. arXiv preprint arXiv:2307.01492 (2023) -   \[29\] Lin, T.Y., Dollár, P., Girshick, R., He, K., Hariharan, B., Belongie, S.: Feature pyramid networks for object detection. In: CVPR (2017) -   \[30\] Lin, X., Lin, T., Pei, Z., Huang, L., Su, Z.: Sparse4d: Multi-view 3d object detection with sparse spatial-temporal fusion. arXiv preprint arXiv:2211.10581 (2022) -   \[31\] Liu, H., Lu, T., Xu, Y., Liu, J., Li, W., Chen, L.: Camliflow: Bidirectional camera-lidar fusion for joint optical flow and scene flow estimation. In: CVPR (2022) -   \[32\] Liu, H., Lu, T., Xu, Y., Liu, J., Wang, L.: Learning optical flow and scene flow with bidirectional camera-lidar fusion. arXiv preprint arXiv:2303.12017 (2023) -   \[33\] Liu, H., Teng, Y., Lu, T., Wang, H., Wang, L.: Sparsebev: High-performance sparse 3d object detection from multi-camera videos. In: ICCV (2023) -   \[34\] Liu, Y., Wang, T., Zhang, X., Sun, J.: PETR: position embedding transformation for multi-view 3d object detection. In: ECCV (2022) -   \[35\] Liu, Y., Yan, J., Jia, F., Li, S., Gao, Q., Wang, T., Zhang, X., Sun, J.: Petrv2: A unified framework for 3d perception from multi-camera images. arXiv preprint arXiv:2206.01256 (2022) -   \[36\] Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. In: ICLR (2019) -   \[37\] Mescheder, L., Oechsle, M., Niemeyer, M., Nowozin, S., Geiger, A.: Occupancy networks: Learning 3d reconstruction in function space. In: CVPR (2019) -   \[38\] Milletari, F., Navab, N., Ahmadi, S.A.: V-net: Fully convolutional neural networks for volumetric medical image segmentation. In: 3DV (2016) -   \[39\] Murez, Z., Van As, T., Bartolozzi, J., Sinha, A., Badrinarayanan, V., Rabinovich, A.: Atlas: End-to-end 3d scene reconstruction from posed images. In: ECCV (2020) -   \[40\] Pan, M., Liu, J., Zhang, R., Huang, P., Li, X., Liu, L., Zhang, S.: Renderocc: Vision-centric 3d occupancy prediction with 2d rendering supervision. arXiv preprint arXiv:2309.09502 (2023) -   \[41\] Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., et al.: Pytorch: An imperative style, high-performance deep learning library. In: NeurIPS (2019) -   \[42\] Peng, S., Niemeyer, M., Mescheder, L., Pollefeys, M., Geiger, A.: Convolutional occupancy networks. In: ECCV (2020) -   \[43\] Ronneberger, O., Fischer, P., Brox, T.: U-net: Convolutional networks for biomedical image segmentation. In: MICCAI (2015) -   \[44\] Schult, J., Engelmann, F., Hermans, A., Litany, O., Tang, S., Leibe, B.: Mask3d: Mask transformer for 3d semantic instance segmentation. In: ICRA (2023) -   \[45\] Sima, C., Tong, W., Wang, T., Chen, L., Wu, S., Deng, H., Gu, Y., Lu, L., Luo, P., Lin, D., Li, H.: Scene as occupancy. In: ICCV (2023) -   \[46\] Stier, N., Rich, A., Sen, P., Höllerer, T.: Vortx: Volumetric 3d reconstruction with transformers for voxelwise view selection and fusion. In: 3DV (2021) -   \[47\] Sun, J., Xie, Y., Chen, L., Zhou, X., Bao, H.: Neuralrecon: Real-time coherent 3d reconstruction from monocular video. In: CVPR (2021) -   \[48\] Takikawa, T., Litalien, J., Yin, K., Kreis, K., Loop, C., Nowrouzezahrai, D., Jacobson, A., McGuire, M., Fidler, S.: Neural geometric level of detail: Real-time rendering with implicit 3d shapes. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 11358–11367 (2021) -   \[49\] Takmaz, A., Fedele, E., Sumner, R.W., Pollefeys, M., Tombari, F., Engelmann, F.: Openmask3d: Open-vocabulary 3d instance segmentation. arXiv preprint arXiv:2306.13631 (2023) -   \[50\] Tian, X., Jiang, T., Yun, L., Wang, Y., Wang, Y., Zhao, H.: Occ3d: A large-scale 3d occupancy prediction benchmark for autonomous driving. In: NeurIPS Datasets and Benchmarks (2023) -   \[51\] Tolstikhin, I.O., Houlsby, N., Kolesnikov, A., Beyer, L., Zhai, X., Unterthiner, T., Yung, J., Steiner, A., Keysers, D., Uszkoreit, J., et al.: Mlp-mixer: An all-mlp architecture for vision. In: NeurIPS (2021) -   \[52\] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł., Polosukhin, I.: Attention is all you need. In: NeurIPS (2017) -   \[53\] Wang, S., Liu, Y., Wang, T., Li, Y., Zhang, X.: Exploring object-centric temporal modeling for efficient multi-view 3d object detection. arXiv preprint arXiv:2303.11926 (2023) -   \[54\] Wang, X., Zhu, Z., Xu, W., Zhang, Y., Wei, Y., Chi, X., Ye, Y., Du, D., Lu, J., Wang, X.: Openoccupancy: A large scale benchmark for surrounding semantic occupancy perception. arXiv preprint arXiv:2303.03991 (2023) -   \[55\] Wang, Y., Guizilini, V.C., Zhang, T., Wang, Y., Zhao, H., Solomon, J.: Detr3d: 3d object detection from multi-view images via 3d-to-2d queries. In: CoRL (2022) -   \[56\] Wang, Y., Chen, Y., Liao, X., Fan, L., Zhang, Z.: Panoocc: Unified occupancy representation for camera-based 3d panoptic segmentation. arXiv preprint arXiv:2306.10013 (2023) -   \[57\] Wei, Y., Zhao, L., Zheng, W., Zhu, Z., Zhou, J., Lu, J.: Surroundocc: Multi-camera 3d occupancy prediction for autonomous driving. In: ICCV (2023) -   \[58\] Yang, Z., Chen, L., Sun, Y., Li, H.: Visual point cloud forecasting enables scalable autonomous driving. In: CVPR (2024) -   \[59\] Yang, Z., Jiang, L., Sun, Y., Schiele, B., Jia, J.: A unified query-based paradigm for point cloud understanding. In: ICCV (2022) -   \[60\] Yang, Z., Sun, Y., Liu, S., Jia, J.: 3dssd: Point-based 3d single stage object detector. In: CVPR (2020) -   \[61\] Yang, Z., Sun, Y., Liu, S., Shen, X., Jia, J.: Std: Sparse-to-dense 3d object detector for point cloud. In: ICCV (2019) -   \[62\] Yang, Z., Zhou, Y., Chen, Z., Ngiam, J.: 3d-man: 3d multi-frame attention network for object detection. In: ICCV (2021) -   \[63\] Yin, T., Zhou, X., Krahenbuhl, P.: Center-based 3d object detection and tracking. In: CVPR (2021) -   \[64\] Zhu, X., Su, W., Lu, L., Li, B., Wang, X., Dai, J.: Deformable detr: Deformable transformers for end-to-end object detection. arXiv preprint arXiv:2010.04159 (2020)