𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs   // Restore the saved color scheme preference, or // enact the browser preference if "automatic", // without expecting DOM load to have completed. // // Also restore any saved readingmode and ToC display preferences. function initializeReadingPreferences() { let saved\_theme = localStorage.getItem("ar5iv\_theme") || "automatic"; if (saved\_theme === "automatic") { if (window.matchMedia("(prefers-color-scheme: dark)").matches) { saved\_theme = "dark"; } } if (saved\_theme == "dark") { document.documentElement.setAttribute("data-theme", "dark"); } else { document.documentElement.setAttribute("data-theme", "light"); } const tocDisplay = localStorage.getItem('arxiv\_html\_paper\_toc\_display'); if (tocDisplay) { document.documentElement.setAttribute("data-toc-display", tocDisplay); } const readingMode = localStorage.getItem('arxiv\_html\_paper\_reading\_mode'); if (readingMode) { document.documentElement.setAttribute("data-reading-mode", readingMode); } } // Run as soon as JS starts, to minimize repainting initializeReadingPreferences();

##### Report GitHub Issue

×

Title:

Content selection saved. Describe the issue below:

Description:

Submit without GitHub Submit in GitHub

[![arXiv logo](/static/browse/0.3.4/images/arxiv-logo-one-color-white.svg) Back to arXiv](/)

[Why HTML?](https://info.arxiv.org/about/accessible_HTML.html) [Report Issue](# "Report an Issue") [Back to Abstract](/abs/2603.09737v1 "Back to abstract page") [Download PDF](/pdf/2603.09737v1 "Download PDF")[](javascript:toggleNavTOC\(\); "Toggle navigation")[](javascript:toggleReadingMode\(\); "Disable reading mode, show header and footer")

1.  [Abstract](#abstract1 "In 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs") 2.  [I Introduction](#S1 "In 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs") 3.  [II Related Work](#S2 "In 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs") 1.  [II-A Semantic Occupancy Prediction](#S2.SS1 "In II Related Work ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs") 2.  [II-B Missing-View BEV Perception](#S2.SS2 "In II Related Work ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs") 3.  [II-C Masked Visual Modeling](#S2.SS3 "In II Related Work ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs") 4.  [III Method](#S3 "In 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs") 1.  [III-A Overall Architecture](#S3.SS1 "In III Method ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs") 2.  [III-B Multi-view Masked Reconstruction (MMR)](#S3.SS2 "In III Method ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs") 1.  [III-B1 Perspective Relationship Modeling](#S3.SS2.SSS1 "In III-B Multi-view Masked Reconstruction (MMR) ‣ III Method ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs") 2.  [III-B2 Overlap-based Feature Aggregation](#S3.SS2.SSS2 "In III-B Multi-view Masked Reconstruction (MMR) ‣ III Method ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs") 3.  [III-B3 Masked Reconstruction Mechanism](#S3.SS2.SSS3 "In III-B Multi-view Masked Reconstruction (MMR) ‣ III Method ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs") 4.  [III-B4 MMR Loss Function](#S3.SS2.SSS4 "In III-B Multi-view Masked Reconstruction (MMR) ‣ III Method ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs") 3.  [III-C Feature Memory Module (FMM)](#S3.SS3 "In III Method ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs") 1.  [III-C1 Single-Proto Strategy](#S3.SS3.SSS1 "In III-C Feature Memory Module (FMM) ‣ III Method ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs") 2.  [III-C2 Multi-Proto Strategy](#S3.SS3.SSS2 "In III-C Feature Memory Module (FMM) ‣ III Method ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs") 3.  [III-C3 Memory-Enhanced Feature](#S3.SS3.SSS3 "In III-C Feature Memory Module (FMM) ‣ III Method ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs") 5.  [IV Experiments](#S4 "In 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs") 1.  [IV-A Datasets and Metrics](#S4.SS1 "In IV Experiments ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs") 2.  [IV-B Implementation Details](#S4.SS2 "In IV Experiments ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs") 3.  [IV-C Robustness to Single-View Failures](#S4.SS3 "In IV Experiments ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs") 4.  [IV-D Scaling Behavior under Multi-View Dropout](#S4.SS4 "In IV Experiments ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs") 5.  [IV-E Ablation Study of the Model Components](#S4.SS5 "In IV Experiments ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs") 6.  [IV-F Qualitative Results](#S4.SS6 "In IV Experiments ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs") 7.  [IV-G Model Efficiency](#S4.SS7 "In IV Experiments ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs") 6.  [V Conclusion and Future Work](#S5 "In 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs") 7.  [References](#bib "In 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs")

[License: arXiv.org perpetual non-exclusive license](https://info.arxiv.org/help/license/index.html#licenses-available)

arXiv:2603.09737v1 \[cs.CV\] 10 Mar 2026

# $M^{2}$-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs

Kaixin Lin1, Kunyu Peng2,3,∗, Di Wen2, Yufan Chen2, Ruiping Liu2, and Kailun Yang1,∗ This work was supported in part by the National Natural Science Foundation of China (Grant No. 62473139), in part by the Hunan Provincial Research and Development Project (Grant No. 2025QK3019), in part by the State Key Laboratory of Autonomous Intelligent Unmanned Systems (the opening project number ZZKF2025-2-10), and in part by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) - SFB 1574 - 471687386.1The authors are with the School of Artificial Intelligence and Robotics and the National Engineering Research Center of Robot Visual Perception and Control Technology, Hunan University, China (email: kailun.yang@hnu.edu.cn).2The authors are with the Institute for Anthropomatics and Robotics, Karlsruhe Institute of Technology, Germany (email: kunyu.peng@kit.edu).3The author is also with INSAIT, Sofia University “St. Kliment Ohridski”, Bulgaria.\*Corresponding authors: Kailun Yang and Kunyu Peng.

###### Abstract

Semantic occupancy prediction enables dense 3D geometric and semantic understanding for autonomous driving. However, existing camera-based approaches implicitly assume complete surround-view observations, an assumption that rarely holds in real-world deployment due to occlusion, hardware malfunction, or communication failures. We study semantic occupancy prediction under incomplete multi-camera inputs and introduce $M^{2}$-Occ, a framework designed to preserve geometric structure and semantic coherence when views are missing. $M^{2}$-Occ addresses two complementary challenges. First, a Multi-view Masked Reconstruction (MMR) module leverages the spatial overlap among neighboring cameras to recover missing-view representations directly in the feature space. Second, a Feature Memory Module (FMM) introduces a learnable memory bank that stores class-level semantic prototypes. By retrieving and integrating these global priors, the FMM refines ambiguous voxel features, ensuring semantic consistency even when observational evidence is incomplete. We introduce a systematic missing-view evaluation protocol on the nuScenes-based SurroundOcc benchmark, encompassing both deterministic single-view failures and stochastic multi-view dropout scenarios. Under the safety-critical missing back-view setting, $M^{2}$-Occ improves the IoU by 4.93%. As the number of missing cameras increases, the robustness gap further widens; for instance, under the setting with five missing views, our method boosts the IoU by 5.01%. These gains are achieved without compromising full-view performance. The source code will be publicly released at [https://github.com/qixi7up/M2-Occ](https://github.com/qixi7up/M2-Occ).

## I Introduction

Autonomous vehicles require a fine-grained understanding of 3D environments for safe navigation decisions \[[6](#bib.bib25 "Are we ready for autonomous driving? The KITTI vision benchmark suite"), [2](#bib.bib8 "nuScenes: A multimodal dataset for autonomous driving")\]. Recently, 3D semantic occupancy prediction has emerged as a key task, providing a voxel-level representation of free space and semantic obstacles around the ego-vehicle  \[[3](#bib.bib14 "MonoScene: Monocular 3D semantic scene completion"), [27](#bib.bib26 "OpenOccupancy: A large scale benchmark for surrounding semantic occupancy perception")\]. While LiDAR-based methods provide accurate depth information, camera-based solutions have garnered increasing attention due to dense information and effectiveness \[[11](#bib.bib12 "VoxFormer: Sparse voxel transformer for camera-based 3D semantic scene completion"), [12](#bib.bib27 "BEVFormer: Learning bird’s-eye-view representation from multi-camera images via spatiotemporal transformers")\]. Semantic Occupancy Prediction extends Bird’s-Eye-View (BEV) perception to dense 3D voxel space, thereby distinguishing arbitrarily shaped obstacles and detailed scene semantics, *e.g.*, SurroundOcc \[[30](#bib.bib1 "SurroundOcc: Multi-camera 3D occupancy prediction for autonomous driving")\] and TPVFormer \[[10](#bib.bib2 "Tri-perspective view for vision-based 3D semantic occupancy prediction")\].

![Refer to caption](2603.09737v1/x1.png)

Figure 1: Existing methods rely on complete camera inputs and suffer from geometric gaps when a sensor fails, *e.g.*, missing FRONT-view. Our $M^{2}$-Occ maintains perceptual integrity by hallucinating missing features from adjacent overlaps and stabilizing semantics via global memory, achieving superior performance, especially when multiple views are missing.

Existing multi-camera paradigms for semantic occupancy prediction typically fuse features from multiple views, *e.g.*, six cameras in nuScenes \[[2](#bib.bib8 "nuScenes: A multimodal dataset for autonomous driving")\], to build a unified 3D scene representation. These approaches generally operate under the implicit assumption of ideal sensing conditions, where all cameras are synchronized, calibrated, and fully functional.

In practice, however, real-world scenarios frequently violate this assumption. Camera sensors are inherently vulnerable to complete failure, *e.g.*, lens occlusion, hardware malfunction or communication dropouts. Such missing-view conditions introduce incomplete spatial coverage and inconsistent cross-view correspondences, which can significantly degrade the stability and reliability of occupancy prediction systems. In our preliminary experiments, we observed that even classic models like SurroundOcc \[[30](#bib.bib1 "SurroundOcc: Multi-camera 3D occupancy prediction for autonomous driving")\] suffer a drastic performance drop when a single critical view is lost. This vulnerability poses a serious safety risk for autonomous systems. As illustrated in Fig. [1](#S1.F1 "Figure 1 ‣ I Introduction ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs"), such failures create significant geometric gaps in the perceived environment, compromising the vehicle’s situational awareness. We adopt SurroundOcc as our baseline because it is a representative and widely adopted multi-camera occupancy prediction model, achieving strong performance under full-view conditions while being clearly vulnerable to missing inputs—making it an ideal testbed for evaluating robustness enhancements. This vulnerability poses a severe safety threat.

To tackle this challenge, we draw inspiration from the human ability to infer unseen regions from context and memory. We propose $M^{2}$-Occ, a generic framework that addresses the recovery of missing sensory information. Our approach is built on two pillars: feature-level reconstruction from neighboring views and semantic regularization through global memory.

First, in the typical sensor configurations, adjacent cameras have overlapping Fields of View (FoV). For instance, a front-left camera partially overlaps the front camera’s blind spot. Leveraging this redundancy, we introduce a Multi-View Masked Reconstruction (MMR) module as a “soft repair” mechanism. Unlike generative methods \[[28](#bib.bib35 "DriveDreamer: towards real-world-driven world models for autonomous driving")\] that hallucinate raw pixels, MMR operates in the feature space. It utilizes a transformer-based decoder to aggregate contextual information from neighboring unmasked views to reconstruct the lost features. Secondly, relying solely on visual reconstruction can yield noisy or ambiguous results. To provide high-level semantic guidance, we introduce a Feature Memory Module (FMM) that learns global semantic prototypes using two complementary strategies: Single-Proto and Multi-Proto. The Single-Proto strategy maintains one global centroid per semantic class to capture its core characteristics, promoting stability and robustness under incomplete observations. In contrast, the Multi-Proto strategy learns multiple sub-prototypes for each class to model intra-class variance (e.g., different vehicle types or orientations) and dynamically retrieves them based on feature similarity, enabling finer-grained semantic refinement. This memory bank serves as a prior knowledge base, allowing the model to refine the reconstructed voxels based on learned class-specific attributes, ensuring that a “car” object still retains the characteristic features of a car, even if its visual features are partially corrupted.

To verify the effectiveness of our approach, we provide a systematic set of analyses for missing-view occupancy prediction. We simulate realistic failure patterns, including specific single-view losses (*e.g.*, front or rear camera failure due to physical damage) and stochastic multi-view dropouts. This protocol allows us to quantify the perceptual boundaries of occupancy models and identify the specific vulnerability of existing methods when facing blind spots caused by sensor malfunctions. Extensive empirical results on the nuScenes dataset demonstrate that $M^{2}$-Occ significantly enhances robustness without sacrificing standard performance. Specifically, under the safety-critical “missing back view” setting, our method improves the IoU by $4.93\%$ compared to the baseline, effectively recovering geometry in the rear blind spot. Furthermore, in extreme scenarios where up to $5$ cameras are disabled, our framework exhibits remarkable resilience, maintaining an IoU of $18.36\%$ while the baseline collapses to $13.35\%$, proving its capability to preserve essential structural information under catastrophic sensor failure. Our main contributions are summarized as follows:

-   •

We conduct a systematic study on semantic occupancy prediction under incomplete multi-camera inputs. The results show that even with only one missing view, representative state-of-the-art models such as SurroundOcc suffer severe performance degradation. This finding highlights the urgent need to build robust perception systems for real-world deployment.

-   •

We propose $M^{2}$-Occ, a novel framework that enhances robustness against camera failures through two key innovations: the Multi-view Masked Reconstruction (MMR) module that recovers missing-view features by leveraging spatial overlaps between adjacent cameras, and the Feature Memory Module (FMM) that refines voxel representations using learnable semantic prototypes.

-   •

Extensive experiments conducted on the nuScenes and SurroundOcc datasets demonstrate that our method significantly outperforms existing classic approaches in terms of robustness, and can effectively recover model performance in missing-view scenarios.

## II Related Work

### II-A Semantic Occupancy Prediction

Semantic Occupancy Prediction (SOP) estimates a dense 3D voxel grid, assigning occupancy and semantic labels to both visible and occluded regions \[[20](#bib.bib29 "Semantic scene completion from a single depth image"), [1](#bib.bib28 "SemanticKITTI: A dataset for semantic scene understanding of LiDAR sequences")\]. This representation enables holistic scene understanding beyond 2D projections or 3D bounding boxes for autonomous driving. Early SOP methods, *e.g.*, LMSCNet \[[19](#bib.bib9 "LMSCNet: Lightweight multiscale 3D semantic completion")\], JS3CNet \[[32](#bib.bib10 "Sparse single sweep LiDAR point cloud segmentation via learning contextual shape priors from scene completion")\], use LiDAR or depth with 3D CNNs to complete sparse scans. LMSCNet \[[19](#bib.bib9 "LMSCNet: Lightweight multiscale 3D semantic completion")\] fills occlusions and JS3CNet \[[32](#bib.bib10 "Sparse single sweep LiDAR point cloud segmentation via learning contextual shape priors from scene completion")\] introduces context-shape priors, but LiDAR-based approaches faced point sparsity and sensor cost. Regarding camera-based SOPs, MonoScene \[[3](#bib.bib14 "MonoScene: Monocular 3D semantic scene completion")\] shows that a monocular image can reconstruct a scene, while multi-camera $360^{\circ}$ systems like SurroundOcc \[[30](#bib.bib1 "SurroundOcc: Multi-camera 3D occupancy prediction for autonomous driving")\] provide $360^{\circ}$ coverage. Transformer models, *e.g.*, TPVFormer \[[10](#bib.bib2 "Tri-perspective view for vision-based 3D semantic occupancy prediction")\], OccFormer \[[34](#bib.bib11 "OccFormer: Dual-path transformer for vision-based 3D semantic occupancy prediction")\], VoxFormer \[[11](#bib.bib12 "VoxFormer: Sparse voxel transformer for camera-based 3D semantic scene completion")\], COTR \[[17](#bib.bib13 "COTR: Compact occupancy transformer for vision-based 3D occupancy prediction")\], introduce 2D–3D fusion techniques further improving accuracy. To further improve performance and robustness, some works fuse multiple sensor modalities so their complementary strengths can compensate for one another. For instance, OpenOccupancy \[[26](#bib.bib15 "OpenOccupancy: a large scale benchmark for surrounding semantic occupancy perception")\] shows LiDAR-camera fusion outperforms single-sensor, while OccFusion \[[18](#bib.bib16 "OccFusion: Multi-sensor fusion framework for 3D semantic occupancy prediction")\] fuses images with LiDAR for superior accuracy. Furthermore, FusionOcc \[[33](#bib.bib17 "FusionOcc: Multi-modal fusion for 3D occupancy prediction")\] and MS-Occ \[[31](#bib.bib18 "MS-Occ: Multi-stage LiDAR-camera fusion for 3D semantic occupancy prediction")\] show that adding LiDAR or radar improves occupancy but adds complexity.

Meanwhile, semantic occupancy research continues to expand. POP3D \[[22](#bib.bib19 "POP-3D: Open-vocabulary 3D occupancy prediction from images")\] explores open-vocabulary 3D occupancy by aligning visual features with language models, while QuadricFormer \[[35](#bib.bib22 "QuadricFormer: Scene as superquadrics for 3D semantic occupancy prediction")\] improves efficiency through compact object-centric superquadric representations. Temporal extensions such as UniOcc \[[29](#bib.bib23 "UniOcc: A unified benchmark for occupancy forecasting and prediction in autonomous driving")\] and STCOcc \[[13](#bib.bib24 "STCOcc: Sparse spatial-temporal cascade renovation for 3D occupancy and scene flow prediction")\] incorporate time-series information to predict occupancy dynamics.

Despite these advances, most existing methods assume all camera views are available at inference time. In practice, autonomous systems may encounter missing or occluded inputs. Our work addresses this limitation by explicitly enhancing robustness to sensor failure scenarios.

![Refer to caption](2603.09737v1/x2.png)

Figure 2: An overview of the proposed $M^{2}$-Occ framework. Multi-view images are first processed by a shared backbone to extract 2D features. To handle missing or corrupted views, the Multi-view Masked Reconstruction (MMR) module leverages spatial overlaps from adjacent cameras to reconstruct the lost features. These features are then lifted into a unified 3D volume. Finally, the Feature Memory Module (FMM) refines the 3D voxel representations by retrieving high-level global semantic prototypes, ensuring structural and semantic consistency before generating the dense 3D occupancy prediction.

### II-B Missing-View BEV Perception

Robustness to incomplete sensors has been actively studied in BEV-based perception and mapping. UniBEV \[[25](#bib.bib5 "UniBEV: Multi-modal 3D object detection with uniform bev encoders for robustness against missing sensor modalities")\] is designed to remain functional under missing sensor modalities. MetaBEV \[[5](#bib.bib7 "MetaBEV: Solving sensor failures for 3D detection and map segmentation")\] addresses sensor corruptions and modality absence for joint BEV tasks. M-BEV \[[4](#bib.bib4 "M-BEV: Masked BEV perception for robust autonomous driving")\] is explicitly trained with whole-view masking and reconstructs missing-view features using cross-view context. Beyond detection, SafeMap \[[7](#bib.bib3 "SafeMap: Robust HD map construction from incomplete observations")\] strengthens BEV map construction by correcting BEV representations from complete observations, whereas FlexMap \[[24](#bib.bib6 "FlexMap: Generalized HD map construction from flexible camera configurations")\] targets missing views without per-configuration retraining. In contrast to these BEV-oriented works, our work focuses on dense 3D semantic occupancy prediction under missing surround-view cameras and introduces feature-level view recovery together with voxel-level semantic regularization to maintain consistent volumetric semantics.

### II-C Masked Visual Modeling

Masked visual modeling has emerged as a powerful self-supervised learning paradigm across various domains. Inspired by Masked Autoencoders (MAE) \[[8](#bib.bib34 "Masked autoencoders are scalable vision learners")\] in natural language processing and computer vision, recent works \[[15](#bib.bib30 "BEV-MAE: Bird’s eye view masked autoencoders for point cloud pre-training in autonomous driving scenarios")\] adapt this framework to BEV-based perception. For instance, MAE-style approaches in BEV, such as BEVT \[[23](#bib.bib31 "BEVT: BERT pretraining of video transformers")\] and VideoMAE \[[21](#bib.bib32 "VideoMAE: Masked autoencoders are data-efficient learners for self-supervised video pre-training")\], employ patch-level masking to pre-train models for scalable feature learning. However, these methods primarily focus on general representation learning and do not address the specific challenges of missing or corrupted sensor inputs in autonomous driving.

To address this, the M-BEV framework \[[4](#bib.bib4 "M-BEV: Masked BEV perception for robust autonomous driving")\] introduces a novel Masked View Reconstruction (MVR) module tailored specifically for BEV perception. Unlike MAE, which masks random patches within a single image, MVR masks the entire camera view and reconstructs it using the spatiotemporal context of neighboring views. This design is crucial for handling sensor failures in the real world, where entire views may be missing. Furthermore, while previous research (e.g., MetaBEV \[[5](#bib.bib7 "MetaBEV: Solving sensor failures for 3D detection and map segmentation")\]) has explored cross-modal fusion (e.g., LiDAR and camera) to mitigate sensor failures, the M-BEV framework achieves robustness using only camera input, making it more cost-effective and scalable.

Masked visual modeling has demonstrated strong potential in BEV perception and related vision tasks, yet it remains underexplored in semantic occupancy prediction. This task is vital for autonomous driving, providing comprehensive 3D scene understanding for safe navigation.

However, robustness under missing-view conditions—especially those caused by sensor failure—has received limited attention. This paper specifically targets this critical scenario to ensure reliable perception under hardware malfunctions or communication dropouts.

## III Method

### III-A Overall Architecture

As shown in Fig. [2](#S2.F2 "Figure 2 ‣ II-A Semantic Occupancy Prediction ‣ II Related Work ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs"), the proposed $M^{2}$-Occ framework is designed to lift multi-view 2D images into a dense 3D semantic occupancy representation while ensuring robustness against sensor failures. The architecture follows a standard 2D-to-3D paradigm. Let $\mathcal{I}=\{I_{i}\}_{i=1}^{N}$ denote the set of input images from $N$ surround-view cameras. First, a shared image backbone (e.g., ResNet-101 \[[9](#bib.bib21 "Deep residual learning for image recognition")\]) with a Feature Pyramid Network (FPN) \[[14](#bib.bib33 "Feature pyramid networks for object detection")\] serves as the encoder $\mathcal{E}$ to extract multi-scale 2D features $F_{2D}=\mathcal{E}(\mathcal{I})$. Subsequently, a 2D-to-3D view transformation module $\mathcal{T}$, based on spatial cross-attention, lifts these features into a unified 3D volume space, yielding $V_{3D}=\mathcal{T}(F_{2D})$. Finally, a 3D occupancy head $\mathcal{H}$ processes the volume to predict the voxel-wise semantic labels $Y$. The entire pipeline can be formulated as:

$$ Y=\mathcal{H}(V_{3D})=\mathcal{H}(\mathcal{T}(\mathcal{E}(\mathcal{I}))) $$

To mitigate the impact of missing views, we introduce the Multi-view Masked Reconstruction (MMR) module during the feature extraction stage $\mathcal{E}(\cdot)$ and the Feature Memory Module (FMM) during the volume refinement stage $\mathcal{T}(\cdot)$.

![Refer to caption](2603.09737v1/x3.png)

Figure 3: An overview of Multi-view Masked Reconstruction (MMR). The MMR module extracts overlapping boundary features from adjacent unmasked views and concatenates them with a central learnable mask token. A lightweight transformer decoder then processes this structural prior to reconstruct the missing view’s representations, preserving spatial continuity.

### III-B Multi-view Masked Reconstruction (MMR)

The structure of the Multi-view Masked Reconstruction (MMR) module is shown in Fig. [3](#S3.F3 "Figure 3 ‣ III-A Overall Architecture ‣ III Method ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs").

#### III-B1 Perspective Relationship Modeling

In autonomous driving setups like nuScenes, cameras are mounted with significant overlaps to cover the full $360^{\circ}$ field of view. We explicitly model this physical layout as a cyclic graph to exploit spatial redundancy. For any camera view $v_{i}$, we identify its spatial neighborhood $\mathcal{N}(v_{i})$ consisting of the immediately adjacent left and right cameras:

$$ \mathcal{N}(v_{i})={v_{(i-1)\pmod{N}},\ v_{(i+1)\pmod{N}}}. $$

This graph allows us to locate the specific sources of complementary visual information when a particular view suffers from occlusion or failure.

#### III-B2 Overlap-based Feature Aggregation

When a specific view $v_{i}$ is masked (simulating a failure), we cannot estimate the environmental information solely based on its own visual input. However, the boundaries of the missing view are often visible in the neighboring cameras. To leverage this context information, we perform a feature cropping and splicing operation. Let $\mathbf{f}_{left}$ and $\mathbf{f}_{right}$ represent the feature maps of the left and right neighbors, respectively. We extract the overlapping boundary regions of width $w_{ov}$ (which corresponds to the physical overlap area) and concatenate them with a learnable mask token $\mathbf{e}_{mask}$. This synthesized feature $\mathbf{f}_{ref}$ serves as a structural prior for reconstruction:

$$ \mathbf{f}_{ref}=\text{Concat}(\mathbf{f}_{left}[:,-w_{ov}:],\ \mathbf{e}_{mask},\ \mathbf{f}_{right}[:,:w_{ov}]), $$

where $\text{Concat}(\cdot)$ denotes channel-wise concatenation of the feature slices. Here, the mask token acts as a placeholder for the central blind spot, initializing the query for the subsequent generative process.

![Refer to caption](2603.09737v1/x4.png)

Figure 4: A comparison between the single-proto strategy and the multi-proto strategy. While the single-proto approach maintains one global centroid per semantic class, the multi-proto strategy captures intra-class variance by learning multiple sub-prototypes and dynamically retrieving them based on feature similarity.

#### III-B3 Masked Reconstruction Mechanism

To reconstruct missing details from the rough structural prior, we employ a lightweight transformer decoder $\mathcal{D}$. Since geometric correspondence is crucial, we augment the reference features $\mathbf{f}_{ref}$ with a learnable positional embedding $\mathbf{p}_{pos}$ to preserve spatial awareness. The decoder, composed of stacked residual transformer layers with layer scaling, refines the features to approximate the original unmasked representation $\hat{\mathbf{f}}_{i}$:

$$ \hat{\mathbf{f}}_{i}=\mathcal{D}(\mathbf{f}_{ref}+\mathbf{p}_{pos}). $$

This process forces the network to learn the spatial continuity of the environment, enabling it to infer the contents of a missing view based on context.

#### III-B4 MMR Loss Function

To ensure the reconstructed features are semantically meaningful and aligned with the original feature distribution, we impose a reconstruction constraint between the estimated unmasked representation $\hat{\mathbf{f}}_{i}$ and the original unmasked representation $\mathbf{f}^{gt}_{i}$. Crucially, we calculate the loss only on the set of masked view indices $\mathcal{M}$ to prevent the network from learning identity mappings for unmasked views. We use the Mean Squared Error (MSE) loss:

$$ \mathcal{L}_{MMR}=\frac{1}{|\mathcal{M}|}\sum_{i\in\mathcal{M}}\left\lVert\hat{\mathbf{f}}_{i}-\mathbf{f}_{i}^{\mathrm{gt}}\right\rVert_{2}^{2}. $$

### III-C Feature Memory Module (FMM)

TABLE I: Comparison of different methods under various missing camera views (IoU, mIoU, and per-category IoU metrics).

| Setting | Method | IoU | mIoU | 
 barrier

 | 

bicycle

 | 

bus

 | 

car

 | 

const.veh.

 | 

motorcycle

 | 

pedestrian

 | 

traffic cone

 | 

trailer

 | 

truck

 | 

drive.surf.

 | 

other flat

 | 

sidewalk

 | 

terrain

 | 

manmade

 | 

vegetation

 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Standard | SurroundOcc (Wei et al. 2023 \[[30](#bib.bib1 "SurroundOcc: Multi-camera 3D occupancy prediction for autonomous driving")\]) | 32.38 | 20.48 | 21.17 | 11.65 | 28.02 | 31.48 | 10.12 | 14.84 | 14.13 | 11.15 | 13.92 | 23.41 | 39.77 | 21.55 | 25.82 | 23.74 | 14.66 | 22.32 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Front | SurroundOcc (Wei et al. 2023 \[[30](#bib.bib1 "SurroundOcc: Multi-camera 3D occupancy prediction for autonomous driving")\]) | 25.03 | 15.54 | 17.03 | 10.37 | 19.81 | 26.90 | 9.72 | 11.36 | 12.35 | 10.20 | 11.77 | 16.56 | 16.44 | 15.09 | 18.88 | 18.81 | 13.18 | 20.12 |
| Ours(SurroundOcc) | 30.40 | 16.98 | 17.20 | 5.36 | 21.44 | 26.89 | 10.88 | 9.23 | 11.71 | 7.18 | 11.74 | 20.30 | 36.22 | 16.02 | 23.33 | 21.06 | 12.95 | 20.16 |
| Front Right | SurroundOcc (Wei et al. 2023 \[[30](#bib.bib1 "SurroundOcc: Multi-camera 3D occupancy prediction for autonomous driving")\]) | 30.56 | 18.70 | 19.12 | 10.94 | 26.12 | 28.72 | 8.13 | 13.24 | 12.46 | 10.02 | 12.56 | 21.20 | 38.29 | 19.98 | 23.70 | 21.78 | 12.83 | 20.16 |
| Ours(SurroundOcc) | 31.17 | 18.86 | 18.64 | 9.06 | 27.42 | 28.81 | 9.56 | 12.65 | 11.97 | 8.91 | 12.97 | 22.22 | 38.62 | 20.38 | 24.54 | 22.51 | 12.96 | 20.52 |
| Front Left | SurroundOcc (Wei et al. 2023 \[[30](#bib.bib1 "SurroundOcc: Multi-camera 3D occupancy prediction for autonomous driving")\]) | 30.74 | 18.65 | 20.25 | 10.64 | 25.14 | 29.00 | 8.26 | 13.96 | 12.22 | 10.28 | 12.80 | 20.30 | 38.24 | 18.57 | 23.53 | 21.90 | 12.97 | 20.34 |
| Ours(SurroundOcc) | 31.25 | 18.85 | 19.26 | 9.03 | 27.03 | 29.59 | 9.40 | 12.81 | 11.86 | 9.29 | 12.98 | 21.21 | 38.79 | 19.63 | 24.44 | 22.49 | 13.13 | 20.66 |
| Back | SurroundOcc (Wei et al. 2023 \[[30](#bib.bib1 "SurroundOcc: Multi-camera 3D occupancy prediction for autonomous driving")\]) | 23.94 | 15.26 | 16.54 | 10.88 | 18.39 | 23.16 | 8.90 | 12.67 | 12.50 | 8.70 | 11.51 | 16.85 | 27.51 | 13.53 | 18.88 | 18.05 | 11.13 | 14.92 |
| Ours(SurroundOcc) | 28.87 | 16.01 | 16.23 | 6.20 | 20.41 | 23.66 | 10.38 | 9.91 | 10.51 | 5.71 | 11.72 | 18.26 | 35.02 | 16.20 | 22.47 | 20.17 | 11.90 | 17.34 |
| Back Left | SurroundOcc (Wei et al. 2023 \[[30](#bib.bib1 "SurroundOcc: Multi-camera 3D occupancy prediction for autonomous driving")\]) | 30.35 | 18.69 | 20.38 | 9.84 | 26.09 | 28.84 | 9.52 | 13.87 | 12.57 | 10.53 | 12.39 | 20.94 | 38.11 | 18.47 | 23.29 | 21.79 | 12.64 | 19.70 |
| Ours(SurroundOcc) | 31.08 | 18.98 | 19.66 | 8.77 | 28.00 | 29.18 | 10.17 | 12.80 | 12.36 | 9.28 | 12.83 | 21.81 | 38.98 | 20.00 | 24.40 | 22.49 | 12.81 | 20.12 |
| Back Right | SurroundOcc (Wei et al. 2023 \[[30](#bib.bib1 "SurroundOcc: Multi-camera 3D occupancy prediction for autonomous driving")\]) | 30.62 | 18.83 | 18.84 | 10.45 | 25.99 | 29.22 | 8.68 | 14.13 | 13.20 | 9.89 | 13.03 | 20.99 | 38.25 | 20.22 | 24.18 | 21.61 | 12.75 | 19.87 |
| Ours(SurroundOcc) | 31.19 | 19.04 | 18.27 | 9.02 | 27.50 | 29.69 | 9.60 | 12.91 | 12.79 | 8.97 | 13.66 | 22.13 | 38.90 | 20.59 | 24.80 | 22.53 | 13.06 | 20.29 |

While MMR recovers the geometric structure, the reconstructed features may still suffer from blurring or semantic ambiguity. The FMM addresses this by introducing a global memory bank $\mathbf{M}$ that stores high-quality semantic prototypes, acting as a “long-term memory” to refine the transient observations.A comparison between the Single-Proto and Multi-Proto strategies is provided in Figure [4](#S3.F4 "Figure 4 ‣ III-B2 Overlap-based Feature Aggregation ‣ III-B Multi-view Masked Reconstruction (MMR) ‣ III Method ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs").

#### III-C1 Single-Proto Strategy

In this simplified strategy, we assume each semantic class $k$ can be represented by a single global centroid. We maintain a prototype $\mathbf{m}_{k}$ which acts as the ideal feature representation for class $k$. To ensure stability during training, we update the prototype using a momentum moving average of the mean feature $\bar{\mathbf{f}}_{k}$ of all voxels assigned to class $k$ in the current batch:

$$ \mathbf{m}_{k}^{(t)}=(1-\lambda)\mathbf{m}_{k}^{(t-1)}+\lambda\cdot\bar{\mathbf{f}}_{k}, $$

where $\lambda$ is a momentum coefficient set to 0.1. This effectively filters out noise and outliers from individual mini-batches.

#### III-C2 Multi-Proto Strategy

Real-world objects exhibit high intra-class variance (e.g., a “truck” can be a pickup or a semi-trailer). To capture this diversity, we extend the memory to store $N_{p}$ sub-prototypes per class.

Similarity Calculation: For a query voxel feature $\mathbf{x}$, we compute its cosine similarity with all sub-prototypes $\mathbf{m}_{k,j}$ to determine which subclass it likely belongs to:

$$ s_{k,j}=\frac{\mathbf{x}\cdot\mathbf{m}_{k,j}}{|\mathbf{x}||\mathbf{m}_{k,j}|}. $$

Softmax Weighting: These similarity scores are normalized via a softmax function with temperature $\tau$ to produce retrieval weights $\alpha_{k,j}$, where a lower temperature emphasizes the most relevant prototypes while higher values yield a smoother distribution:

$$ \alpha_{k,j}=\frac{\exp(s_{k,j}/\tau)}{\sum_{j^{\prime}}\exp(s_{k,j^{\prime}}/\tau)}. $$

#### III-C3 Memory-Enhanced Feature

Finally, we inject the retrieved semantic knowledge back into the 3D volume. Using the predicted class probability $P(k)$ as a gate, we aggregate the weighted prototypes and add them to the original feature $\mathbf{x}$ as a residual correction:

$$ \mathbf{x}^{\prime}=\mathbf{x}+\sum_{k=1}^{K}\left(P(k)\sum_{j=1}^{N_{p}}\alpha_{k,j}\mathbf{m}_{k,j}\right), $$

where $K$ is the number of semantic classes, $P(k)$ denotes the predicted probability for class $k$, and $\alpha_{k,j}$ are the retrieval weights for sub-prototypes. This step significantly sharpens the semantic boundaries, especially in regions reconstructed by MMR, where the original features might be noisy.

## IV Experiments

In this section, we conduct comprehensive experiments to evaluate the proposed $M^{2}$-Occ. Table [I](#S3.T1 "TABLE I ‣ III-C Feature Memory Module (FMM) ‣ III Method ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs") presents the comparison under various single-view missing scenarios. Table [II](#S4.T2 "TABLE II ‣ IV-C Robustness to Single-View Failures ‣ IV Experiments ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs") analyzes the robustness as the number of missing cameras increases. Table [III](#S4.T3 "TABLE III ‣ IV-E Ablation Study of the Model Components ‣ IV Experiments ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs") ablates the contribution of each module and compares prototype strategies. Table [IV](#S4.T4 "TABLE IV ‣ IV-G Model Efficiency ‣ IV Experiments ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs") reports the computational overhead.

### IV-A Datasets and Metrics

To rigorously evaluate the effectiveness of the proposed $M^{2}$-Occ framework in both standard perception and various sensor-failure scenarios, we conduct extensive experiments on the nuScenes dataset \[[2](#bib.bib8 "nuScenes: A multimodal dataset for autonomous driving")\], which is a large-scale, multimodal dataset that has become a cornerstone for benchmarking autonomous driving algorithms in complex urban environments. Developed by Motional, it is the first dataset to provide a full 360-degree sensor suite coverage, comprising data from 6 cameras, 1 LiDAR, 5 RADAR units, and a high-precision GPS/IMU system. The dataset contains $1,000$ driving scenes captured in Boston and Singapore, totaling approximately $1.4$ million images and $390k$ LiDAR sweeps. With rigorous annotations for $23$ object categories across diverse weather conditions and times of day, nuScenes serves as a primary benchmark for tasks such as 3D object detection, multi-object tracking, and trajectory prediction.

In addition, we adopt the dense voxel annotations generated by SurroundOcc through multi-frame LiDAR point cloud aggregation and Poisson surface reconstruction, instead of sparse 3D bounding boxes, enabling the model to perceive the complete 3D structure and occupancy state of the surrounding environment rather than performing only simple object detection.

We report the standard metrics for semantic occupancy: Intersection over Union (IoU) for geometric completeness and mean IoU (mIoU) for semantic accuracy.

### IV-B Implementation Details

Driving scenes in the real world are often complex. We choose a challenging setting to mimic real situations: we randomly discard images of corresponding views using our Random View Masking (RVM) module during training, and mask specific views during testing to evaluate robustness. For the baseline model, we follow the official implementation of SurroundOcc \[[30](#bib.bib1 "SurroundOcc: Multi-camera 3D occupancy prediction for autonomous driving")\]. We use ResNet-101 \[[9](#bib.bib21 "Deep residual learning for image recognition")\] initialized with FCOS3D pre-trained weights as the visual encoder. The model is trained for $24$ epochs using the AdamW optimizer \[[16](#bib.bib20 "Decoupled weight decay regularization")\] with a learning rate of $2\times 10^{-4}$ and a weight decay of $0.01$. The voxel volume is set to $200\times 200\times 16$, covering a range of $\[-50m,50m\]$ in the ground plane. The lightweight transformer decoder $\mathcal{D}$ in the Multi-view Masked Reconstruction module consists of $6$ Transformer blocks, each with 8 attention heads and an MLP ratio of $4$.

### IV-C Robustness to Single-View Failures

TABLE II: Ablation on the number of missing camera views.

| Missing Count | Method | IoU | mIoU |
| --- | --- | --- | --- |
| 1 View | Baseline | 28.42 | 17.55 |
| MMR | 30.52 | 17.76 |
| MMR & FMM | 30.66 | 17.86 |
| 3 Views | Baseline | 20.52 | 11.96 |
| MMR | 25.87 | 11.98 |
| MMR & FMM | 26.06 | 12.15 |
| 5 Views | Baseline | 13.35 | 4.99 |
| MMR | 18.17 | 4.97 |
| MMR & FMM | 18.36 | 5.04 |

Table [I](#S3.T1 "TABLE I ‣ III-C Feature Memory Module (FMM) ‣ III Method ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs") reports results when each individual camera view is removed. Across all single-view failure scenarios, $M^{2}$-Occ consistently improves geometric IoU compared to the baseline model. Under the safety-critical missing rear-view condition, IoU increases from $23.94\%$ to $28.87\%$ ($+4.93\%$). Similar improvements are observed when removing the front view ($25.03\%$ → $30.40\%$) and front-left view ($30.74\%$ → $31.25\%$).

This suggests that the proposed framework primarily restores large-scale spatial structures such as road surfaces and vehicle volumes, which dominate the scene geometry. While geometric IoU consistently improves, gains in mIoU are more nuanced in certain single-view settings. A closer examination of per-category results in Table [I](#S3.T1 "TABLE I ‣ III-C Feature Memory Module (FMM) ‣ III Method ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs") reveals an important limitation: while $M^{2}$-Occ achieves substantial gains on large-scale structures (*e.g.*, “drive.surf.” improves from $27.51\%$ to $35.02\%$ under missing back-view), performance on small objects exhibits mixed results. For instance, under the same missing back-view setting, “pedestrian” IoU drops from $12.50\%$ to $10.51\%$, and “traffic cone” decreases from $8.70\%$ to $5.71\%$.

This performance degradation on small objects can be attributed to two factors. First, the Multi-view Masked Reconstruction (MMR) module relies on overlapping boundary regions between adjacent cameras, which may not provide sufficient spatial resolution to capture fine-grained details of distant or small instances. Second, the reconstructed features inherently lose high-frequency information during the feature-level generation process, making it particularly challenging to preserve the precise boundaries and semantic attributes of small objects.

Similar trends are observed across other missing-view configurations, suggesting that while our framework successfully recovers large-scale geometry and dominant object categories, reconstructing fine semantic details for small objects under incomplete observations remains an open challenge.

### IV-D Scaling Behavior under Multi-View Dropout

We further evaluate performance when multiple cameras are simultaneously removed (Table [II](#S4.T2 "TABLE II ‣ IV-C Robustness to Single-View Failures ‣ IV Experiments ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs")). When the baseline model suffers from single-view missing, its IoU and mIoU drop to $28.42\%$ and $17.55\%$, respectively. In contrast, our $M^{2}$-Occ effectively alleviates such performance degradation, achieving $30.66\%$ and $17.86\%$, respectively.

As the number of missing views increases, the robustness gap widens substantially. With three missing views, the baseline decreases to $20.52\%$ IoU and $11.96\%$, while $M^{2}$-Occ maintains $26.06\%$ and $12.15\%$. Under the most extreme setting of five missing views, the baseline further drops to $13.35\%$ and $4.99\%$, whereas $M^{2}$-Occ retains $18.36\%$ and $5.04\%$.

Notably, in severe multi-view dropout scenarios, improvements are observed in both IoU and mIoU. This indicates that when structural evidence becomes highly sparse, the combination of reconstruction and semantic regularization stabilizes both geometry and core semantic predictions.

### IV-E Ablation Study of the Model Components

TABLE III: Module ablation and prototype strategy comparison (SP: Single-Proto, MP: Multi-Proto).

| Missing | MMR | SP | MP | IoU | mIoU |
| --- | --- | --- | --- | --- | --- |
| $\times$ | $\times$ | $\times$ | $\times$ | 30.13 | 15.31 |
| ✓ | $\times$ | $\times$ | $\times$ | 26.76 | 13.21 |
| ✓ | ✓ | $\times$ | $\times$ | 28.19 | 13.79 |
| ✓ | ✓ | ✓ | $\times$ | 28.38 | 13.55 |
| ✓ | ✓ | $\times$ | ✓ | 27.76 | 12.15 |

Table [III](#S4.T3 "TABLE III ‣ IV-E Ablation Study of the Model Components ‣ IV Experiments ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs") provides a detailed ablation study of the Multi-view Masked Reconstruction (MMR) and Feature Memory Module (FMM) under missing-view conditions.

When a view is masked without any recovery mechanism, the baseline IoU drops from $30.13\%$ to $26.76\%$, indicating that the performance degradation primarily stems from incomplete spatial coverage and broken cross-view correspondences. Introducing MMR alone improves IoU to $28.19\%$, recovering $+1.43\%$ over the degraded baseline. This gain confirms that feature-level reconstruction effectively restores structural continuity by exploiting spatial overlap between adjacent cameras. Notably, the improvement is more pronounced in geometric IoU than mIoU, suggesting that MMR mainly compensates for large-scale spatial structures (*e.g.*, road surfaces and vehicle volumes), which dominate occupancy estimation.

![Refer to caption](2603.09737v1/visual.png)

Figure 5: Visualizations on the nuScenes validation set \[[2](#bib.bib8 "nuScenes: A multimodal dataset for autonomous driving")\]. Our method achieves promising results in the reconstruction of the missing $F$ view across various scenarios, even under weak lighting conditions.

The FMM injects global class-level priors to suppress ambiguity in uncertain voxels, especially in regions where the reconstructed features remain noisy or partially incomplete. Incorporating FMM with the single-prototype strategy further increases IoU to $28.38\%$. In the ablation study, we observe that although the mIoU is improved compared with the baseline, introducing the FMM sometimes leads to fluctuations in the mIoU gain, which may be caused by the uncertainty of long-tailed categories.

In contrast, the multi-prototype variant achieves a lower IoU of $27.76\%$. This suggests that under missing-view conditions, fine-grained prototype assignment may introduce instability. When visual evidence is sparse, the similarity-based routing mechanism can amplify noise or assign voxels to sub-prototypes based on incomplete cues, leading to over-fragmented semantic representations. Therefore, a single, stable class centroid appears more robust than multiple sub-centroids when observations are heavily corrupted.

Overall, the ablation results demonstrate a clear division of labor between the two modules: MMR restores geometric completeness by reconstructing structural features, while FMM enhances feature representation by imposing global class-level constraints. Their combination achieves stable performance under incomplete multi-view observations.

### IV-F Qualitative Results

![Refer to caption](2603.09737v1/feature.png)

Figure 6: Visualization for feature maps. Our reconstructed features can largely replace the original features.

In Figure [5](#S4.F5 "Figure 5 ‣ IV-E Ablation Study of the Model Components ‣ IV Experiments ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs"), we visualize the occupancy prediction results. In the “missing FRONT view” scenario, the baseline method fails to detect the road and vehicles in the front blind spot, producing fragmented geometry and missing semantics. In stark contrast, $M^{2}$-Occ successfully hallucinates the drivable surface and vehicle structures, closely matching the Ground Truth and demonstrating its ability to infer occluded regions from contextual cues. This qualitative result aligns with our quantitative improvements, where our method achieves substantial gains across most categories.

While our framework exhibits strong capability in reconstructing large-scale structures (e.g., roads, vehicles) under missing views, we acknowledge that fine-grained details of small objects (e.g., distant pedestrians) remain challenging to fully recover. Nevertheless, the overall perceptual quality is significantly enhanced: across diverse scenarios with missing front views, $M^{2}$-Occ consistently produces coherent scene layouts where the baseline collapses, confirming its effectiveness as a robust solution for real-world deployment where sensor failures are inevitable.

In Figure [6](#S4.F6 "Figure 6 ‣ IV-F Qualitative Results ‣ IV Experiments ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs"), we present the original feature maps, the feature maps when a certain viewpoint is missing, and the reconstructed feature maps under several scenarios. Our reconstructed features can serve as a reasonable substitute for the original features, particularly for the detection of large objects, including drivable areas, various types of vehicles, and man-made structures. Meanwhile, the reconstruction of small objects and fine-grained textures remains challenging and requires further optimization in future work.

### IV-G Model Efficiency

TABLE IV: Experimental results of the compute overhead.

| Method | Number of Missing Views | Latency(s) | Memory(G) |
| --- | --- | --- | --- |
| Baseline | \- | 0.50 | 5.927 |
| --- | --- | --- | --- |
| Ours | 1 | 0.77 | 6.077 |
| 2 | 0.91 |
| 3 | 1.00 |
| 4 | 1.11 |
| 5 | 1.25 |

We compared the inference time and memory usage of the baseline and our method, as shown in Table [IV](#S4.T4 "TABLE IV ‣ IV-G Model Efficiency ‣ IV Experiments ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs"). All experiments were conducted on an RTX A6000 GPU. Compared with the baseline, our method increases video memory consumption by only approximately $0.15$ GB (about $2.5\%$) while achieving significant performance gains.

The inference latency increases with the number of missing views, as the MMR module sequentially recovers features for each missing view using a transformer decoder. This trade-off introduces a computational overhead that is justified by the substantial improvement in perception reliability under sensor failure, making it suitable for safety-critical driving applications.

## V Conclusion and Future Work

This work addresses semantic occupancy prediction under incomplete multi-camera observations, a practical yet underexplored challenge for autonomous driving. We propose $M^{2}$-Occ, which enhances robustness via feature-level reconstruction and class-level semantic regularization. On the SurroundOcc benchmark, $M^{2}$-Occ consistently restores geometric completeness under sensor failures. In safety-critical missing back-view scenarios, it recovers blind-spot structures, achieving $+4.93$ IoU and $+0.75$ mIoU gains over the baseline. As more cameras are removed, the robustness advantage grows, underscoring the value of spatial redundancy and semantic priors when visual evidence is sparse. Without adding sensors or modifying the backbone, $M^{2}$-Occ reduces reliance on fully functional surround-view systems, contributing to more reliable autonomous perception.

We acknowledge limitations in preserving fine-grained details for small objects under severe missing-view conditions. Future work will explore multi-resolution feature reconstruction, uncertainty-aware refinement, and temporal consistency to better recover small objects and mitigate the impact of instantaneous sensor failures.

## References

-   \[1\] J. Behley, M. Garbade, A. Milioto, J. Quenzel, S. Behnke, C. Stachniss, and J. Gall (2019) SemanticKITTI: A dataset for semantic scene understanding of LiDAR sequences. In Proc. ICCV, pp. 9296–9306. Cited by: [§II-A](#S2.SS1.p1.2 "II-A Semantic Occupancy Prediction ‣ II Related Work ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs"). -   \[2\] H. Caesar et al. (2020) nuScenes: A multimodal dataset for autonomous driving. In Proc. CVPR, pp. 11618–11628. Cited by: [§I](#S1.p1.1 "I Introduction ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs"), [§I](#S1.p2.1 "I Introduction ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs"), [Figure 5](#S4.F5 "In IV-E Ablation Study of the Model Components ‣ IV Experiments ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs"), [Figure 5](#S4.F5.2.1 "In IV-E Ablation Study of the Model Components ‣ IV Experiments ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs"), [§IV-A](#S4.SS1.p1.5 "IV-A Datasets and Metrics ‣ IV Experiments ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs"). -   \[3\] A. Cao and R. de Charette (2022) MonoScene: Monocular 3D semantic scene completion. In Proc. CVPR, pp. 3981–3991. Cited by: [§I](#S1.p1.1 "I Introduction ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs"), [§II-A](#S2.SS1.p1.2 "II-A Semantic Occupancy Prediction ‣ II Related Work ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs"). -   \[4\] S. Chen, Y. Ma, Y. Qiao, and Y. Wang (2024) M-BEV: Masked BEV perception for robust autonomous driving. In Proc. AAAI, pp. 1183–1191. Cited by: [§II-B](#S2.SS2.p1.1 "II-B Missing-View BEV Perception ‣ II Related Work ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs"), [§II-C](#S2.SS3.p2.1 "II-C Masked Visual Modeling ‣ II Related Work ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs"). -   \[5\] C. Ge et al. (2023) MetaBEV: Solving sensor failures for 3D detection and map segmentation. In Proc. ICCV, pp. 8687–8697. Cited by: [§II-B](#S2.SS2.p1.1 "II-B Missing-View BEV Perception ‣ II Related Work ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs"), [§II-C](#S2.SS3.p2.1 "II-C Masked Visual Modeling ‣ II Related Work ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs"). -   \[6\] A. Geiger, P. Lenz, and R. Urtasun (2012) Are we ready for autonomous driving? The KITTI vision benchmark suite. In Proc. CVPR, pp. 3354–3361. Cited by: [§I](#S1.p1.1 "I Introduction ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs"). -   \[7\] X. Hao et al. (2025) SafeMap: Robust HD map construction from incomplete observations. In Proc. ICML, Cited by: [§II-B](#S2.SS2.p1.1 "II-B Missing-View BEV Perception ‣ II Related Work ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs"). -   \[8\] K. He, X. Chen, S. Xie, Y. Li, P. Dollár, and R. Girshick (2022) Masked autoencoders are scalable vision learners. In Proc. CVPR, pp. 15979–15988. Cited by: [§II-C](#S2.SS3.p1.1 "II-C Masked Visual Modeling ‣ II Related Work ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs"). -   \[9\] K. He, X. Zhang, S. Ren, and J. Sun (2016) Deep residual learning for image recognition. In Proc. CVPR, pp. 770–778. Cited by: [§III-A](#S3.SS1.p1.9 "III-A Overall Architecture ‣ III Method ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs"), [§IV-B](#S4.SS2.p1.8 "IV-B Implementation Details ‣ IV Experiments ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs"). -   \[10\] Y. Huang, W. Zheng, Y. Zhang, J. Zhou, and J. Lu (2023) Tri-perspective view for vision-based 3D semantic occupancy prediction. In Proc. CVPR, pp. 9223–9232. Cited by: [§I](#S1.p1.1 "I Introduction ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs"), [§II-A](#S2.SS1.p1.2 "II-A Semantic Occupancy Prediction ‣ II Related Work ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs"). -   \[11\] Y. Li et al. (2023) VoxFormer: Sparse voxel transformer for camera-based 3D semantic scene completion. In Proc. CVPR, pp. 9087–9098. Cited by: [§I](#S1.p1.1 "I Introduction ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs"), [§II-A](#S2.SS1.p1.2 "II-A Semantic Occupancy Prediction ‣ II Related Work ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs"). -   \[12\] Z. Li et al. (2022) BEVFormer: Learning bird’s-eye-view representation from multi-camera images via spatiotemporal transformers. In Proc. ECCV, pp. 1–18. Cited by: [§I](#S1.p1.1 "I Introduction ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs"). -   \[13\] Z. Liao, P. Wei, S. Chen, H. Wang, and Z. Ren (2025) STCOcc: Sparse spatial-temporal cascade renovation for 3D occupancy and scene flow prediction. In Proc. CVPR, pp. 1516–1526. Cited by: [§II-A](#S2.SS1.p2.1 "II-A Semantic Occupancy Prediction ‣ II Related Work ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs"). -   \[14\] T. Lin, P. Dollár, R. Girshick, K. He, B. Hariharan, and S. Belongie (2017) Feature pyramid networks for object detection. In Proc. CVPR, pp. 936–944. Cited by: [§III-A](#S3.SS1.p1.9 "III-A Overall Architecture ‣ III Method ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs"). -   \[15\] Z. Lin, Y. Wang, S. Qi, N. Dong, and M. Yang (2024) BEV-MAE: Bird’s eye view masked autoencoders for point cloud pre-training in autonomous driving scenarios. In Proc. AAAI, pp. 3531–3539. Cited by: [§II-C](#S2.SS3.p1.1 "II-C Masked Visual Modeling ‣ II Related Work ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs"). -   \[16\] I. Loshchilov and F. Hutter (2019) Decoupled weight decay regularization. In Proc. ICLR, Cited by: [§IV-B](#S4.SS2.p1.8 "IV-B Implementation Details ‣ IV Experiments ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs"). -   \[17\] Q. Ma, X. Tan, Y. Qu, L. Ma, Z. Zhang, and Y. Xie (2024) COTR: Compact occupancy transformer for vision-based 3D occupancy prediction. In Proc. CVPR, pp. 19936–19945. Cited by: [§II-A](#S2.SS1.p1.2 "II-A Semantic Occupancy Prediction ‣ II Related Work ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs"). -   \[18\] Z. Ming, J. S. Berrio, M. Shan, and S. Worrall (2025) OccFusion: Multi-sensor fusion framework for 3D semantic occupancy prediction. IEEE Transactions on Intelligent Vehicles 10 (5), pp. 3421–3433. Cited by: [§II-A](#S2.SS1.p1.2 "II-A Semantic Occupancy Prediction ‣ II Related Work ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs"). -   \[19\] L. Roldão, R. de Charette, and A. Verroust-Blondet (2020) LMSCNet: Lightweight multiscale 3D semantic completion. In Proc. 3DV, pp. 111–119. Cited by: [§II-A](#S2.SS1.p1.2 "II-A Semantic Occupancy Prediction ‣ II Related Work ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs"). -   \[20\] S. Song, F. Yu, A. Zeng, A. X. Chang, M. Savva, and T. Funkhouser (2017) Semantic scene completion from a single depth image. In Proc. CVPR, pp. 190–198. Cited by: [§II-A](#S2.SS1.p1.2 "II-A Semantic Occupancy Prediction ‣ II Related Work ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs"). -   \[21\] Z. Tong, Y. Song, J. Wang, and L. Wang (2022) VideoMAE: Masked autoencoders are data-efficient learners for self-supervised video pre-training. In Proc. NeurIPS, pp. 10078–10093. Cited by: [§II-C](#S2.SS3.p1.1 "II-C Masked Visual Modeling ‣ II Related Work ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs"). -   \[22\] A. Vobecky et al. (2023) POP-3D: Open-vocabulary 3D occupancy prediction from images. In Proc. NeurIPS, pp. 50545–50557. Cited by: [§II-A](#S2.SS1.p2.1 "II-A Semantic Occupancy Prediction ‣ II Related Work ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs"). -   \[23\] R. Wang et al. (2022) BEVT: BERT pretraining of video transformers. In Proc. CVPR, pp. 14713–14723. Cited by: [§II-C](#S2.SS3.p1.1 "II-C Masked Visual Modeling ‣ II Related Work ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs"). -   \[24\] R. Wang et al. (2026) FlexMap: Generalized HD map construction from flexible camera configurations. arXiv preprint arXiv:2601.22376. Cited by: [§II-B](#S2.SS2.p1.1 "II-B Missing-View BEV Perception ‣ II Related Work ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs"). -   \[25\] S. Wang, H. Caesar, L. Nan, and J. F. Kooij (2024) UniBEV: Multi-modal 3D object detection with uniform bev encoders for robustness against missing sensor modalities. In Proc. IV, pp. 2776–2783. Cited by: [§II-B](#S2.SS2.p1.1 "II-B Missing-View BEV Perception ‣ II Related Work ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs"). -   \[26\] X. Wang et al. (2023) OpenOccupancy: a large scale benchmark for surrounding semantic occupancy perception. In Proc. ICCV, pp. 17804–17813. Cited by: [§II-A](#S2.SS1.p1.2 "II-A Semantic Occupancy Prediction ‣ II Related Work ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs"). -   \[27\] X. Wang et al. (2023) OpenOccupancy: A large scale benchmark for surrounding semantic occupancy perception. In Proc. ICCV, pp. 17804–17813. Cited by: [§I](#S1.p1.1 "I Introduction ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs"). -   \[28\] X. Wang, Z. Zhu, C. Guan, et al. (2023) DriveDreamer: towards real-world-driven world models for autonomous driving. arXiv preprint arXiv:2309.09777. Cited by: [§I](#S1.p5.1 "I Introduction ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs"). -   \[29\] Y. Wang et al. (2025) UniOcc: A unified benchmark for occupancy forecasting and prediction in autonomous driving. In Proc. ICCV, pp. 25560–25570. Cited by: [§II-A](#S2.SS1.p2.1 "II-A Semantic Occupancy Prediction ‣ II Related Work ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs"). -   \[30\] Y. Wei, L. Zhao, W. Zheng, Z. Zhu, J. Zhou, and J. Lu (2023) SurroundOcc: Multi-camera 3D occupancy prediction for autonomous driving. In Proc. ICCV, pp. 21672–21683. Cited by: [§I](#S1.p1.1 "I Introduction ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs"), [§I](#S1.p3.1 "I Introduction ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs"), [§II-A](#S2.SS1.p1.2 "II-A Semantic Occupancy Prediction ‣ II Related Work ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs"), [TABLE I](#S3.T1.4.1.11.9.2 "In III-C Feature Memory Module (FMM) ‣ III Method ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs"), [TABLE I](#S3.T1.4.1.13.11.2 "In III-C Feature Memory Module (FMM) ‣ III Method ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs"), [TABLE I](#S3.T1.4.1.2.2.2 "In III-C Feature Memory Module (FMM) ‣ III Method ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs"), [TABLE I](#S3.T1.4.1.3.1.2 "In III-C Feature Memory Module (FMM) ‣ III Method ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs"), [TABLE I](#S3.T1.4.1.5.3.2 "In III-C Feature Memory Module (FMM) ‣ III Method ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs"), [TABLE I](#S3.T1.4.1.7.5.2 "In III-C Feature Memory Module (FMM) ‣ III Method ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs"), [TABLE I](#S3.T1.4.1.9.7.2 "In III-C Feature Memory Module (FMM) ‣ III Method ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs"), [§IV-B](#S4.SS2.p1.8 "IV-B Implementation Details ‣ IV Experiments ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs"). -   \[31\] Z. Wei et al. (2026) MS-Occ: Multi-stage LiDAR-camera fusion for 3D semantic occupancy prediction. IEEE Robotics and Automation Letters 11 (1), pp. 370–377. Cited by: [§II-A](#S2.SS1.p1.2 "II-A Semantic Occupancy Prediction ‣ II Related Work ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs"). -   \[32\] X. Yan et al. (2021) Sparse single sweep LiDAR point cloud segmentation via learning contextual shape priors from scene completion. In Proc. AAAI, pp. 3101–3109. Cited by: [§II-A](#S2.SS1.p1.2 "II-A Semantic Occupancy Prediction ‣ II Related Work ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs"). -   \[33\] S. Zhang, Y. Zhai, J. Mei, and Y. Hu (2024) FusionOcc: Multi-modal fusion for 3D occupancy prediction. In Proc. MM, pp. 787–796. Cited by: [§II-A](#S2.SS1.p1.2 "II-A Semantic Occupancy Prediction ‣ II Related Work ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs"). -   \[34\] Y. Zhang, Z. Zhu, and D. Du (2023) OccFormer: Dual-path transformer for vision-based 3D semantic occupancy prediction. Proc. ICCV, pp. 9399–9409. Cited by: [§II-A](#S2.SS1.p1.2 "II-A Semantic Occupancy Prediction ‣ II Related Work ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs"). -   \[35\] S. Zuo, W. Zheng, X. Han, L. Yang, Y. Pan, and J. Lu (2025) QuadricFormer: Scene as superquadrics for 3D semantic occupancy prediction. In Proc. NeurIPS, Cited by: [§II-A](#S2.SS1.p2.1 "II-A Semantic Occupancy Prediction ‣ II Related Work ‣ 𝑀²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs").

Experimental support, please [view the build logs](./2603.09737v1/__stdout.txt) for errors. Generated by [L A T E xml ![[LOGO]](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](https://math.nist.gov/~BMiller/LaTeXML/)  .

## Instructions for reporting errors

We are continuing to improve HTML versions of papers, and your feedback helps enhance accessibility and mobile support. To report errors in the HTML that will help us improve conversion and rendering, choose any of the methods listed below:

-   Click the "Report Issue" ( ) button, located in the page header.

**Tip:** You can select the relevant text first, to include it in your report.

Our team has already identified [the following issues](https://github.com/arXiv/html_feedback/issues). We appreciate your time reviewing and reporting rendering errors we may not have found yet. Your efforts will help us improve the HTML versions for all readers, because disability should not be a barrier to accessing research. Thank you for your continued support in championing open access for all.

Have a free development cycle? Help support accessibility at arXiv! Our collaborators at LaTeXML maintain a [list of packages that need conversion](https://github.com/brucemiller/LaTeXML/wiki/Porting-LaTeX-packages-for-LaTeXML), and welcome [developer contributions](https://github.com/brucemiller/LaTeXML/issues).

BETA