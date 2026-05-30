import torch
from mmcv.runner import force_fp32
from torch.nn import functional as F

from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result, Box3DMode, Coord3DMode
from mmdet.models import build_detector
from mmdet3d.models.builder import build_backbone, build_head, build_neck, build_roi_extractor
from projects.mmdet3d_plugin.ops import build_mlp

from mmseg.models import SEGMENTORS
from mmdet3d.models import builder
from mmdet3d.ops import Voxelization, furthest_point_sample
from projects.mmdet3d_plugin.ops import scatter_v2, get_inner_win_inds
from scipy.sparse.csgraph import connected_components
from mmdet.core import multi_apply, build_bbox_coder
from mmdet3d.models.detectors.single_stage import SingleStage3DDetector
from mmdet3d.models.segmentors.base import Base3DSegmentor
from .single_stage_fsd import SingleStageFSD
import cv2, os, copy
import time, shutil
from torch import nn as nn
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes as LB
from mmdet3d.core import show_result as show_o3d_result
try:
    from torchex import connected_components as cc_gpu
except ImportError:
    cc_gpu = None
import numpy as np

@DETECTORS.register_module()
class FSF(SingleStageFSD):
    """
    Fully Sparse Fusion (FSF) 3D目标检测器（论文：Fully Sparse Fusion for 3D Object Detection, TPAMI 2024）

    整体架构（对应论文 Figure 2）：
    1. segmentor         — 稀疏点云分割网络（FSD），预测每个点的前背景标签和投票中心偏移
    2. img_cross_attn    — 图像增强分割：将2D mask信息通过 segmentor_updated_mlp 注入点特征
    3. frustum_forward   — 视锥查询分支（Camera Query）：以2D实例mask为索引对点云聚合
    4. fsd_forward       — LiDAR查询分支（LiDAR Query）：以图聚类结果为索引对点云聚合
    5. combine + refine  — 合并两路query并进行多阶段精炼（Multi-Stage Refinement）
    """
    def __init__(self,
                backbone,
                segmentor,            # 点云前景分割网络（FSD-based segmentor）
                voxel_layer=None,
                voxel_encoder=None,
                middle_encoder=None,
                neck=None,
                frustum_obj_head=None,  # 视锥查询检测头（FrustumClusterHead）
                frustum_sir=None,       # 视锥分支的 SIR（Set-in-Region）特征聚合层
                bbox_head=None,         # LiDAR查询检测头（SparseClusterHeadV2）
                roi_head=None,
                train_cfg=None,
                test_cfg=None,
                cluster_assigner=None,
                pretrained=None,
                tanh_dims=3,
                init_cfg=None,
                encode_2d_mlp_cfg=dict(
                    in_channel=16,
                    mlp_channel=[128, 128],
                    norm_cfg=dict(type='LN', eps=1e-3),
                    act='gelu',
                ),
                refine_encode_2d_mlp_cfg=None,
                num_classes=10,         # nuScenes=10, AV2=26
                num_cams=6,             # nuScenes使用6个相机
                vis_dir=None,
                encode_label_only=False,
                class_names=None,
                min_pts=5,
                bbox_coder=None,
                roi_extractor=None,     # 精炼阶段用于从点云中提取ROI内点的ROI extractor
                single_refine_sir_layer=None,  # 精炼阶段的SIR层
                mlp_cfg=dict(
                embed_dims=256,
                norm_cfg=dict(type='LN', eps=1e-3),
                act='gelu',
                lidar_img_input_dim=128 * 3 * 2 + 128,  # LiDAR特征 + 图像特征拼接维度
                ),
                fsd_begin_idx=1000,     # FSD query的obj_id偏移量，用于区分frustum和fsd的query
                refined_obj_head=None,  # 精炼阶段的检测头列表（每个stage一个）
                segmentor_updated_mlp=dict(
                    in_channel=10,
                    mlp_channel=[128, 67 + 64],  # 输出用于更新点特征（67=分割logit, 64=投票特征）
                    norm_cfg=dict(type='LN', eps=1e-3),
                    act='gelu',
                ),
                tta_test_cfg={},
                use_frustum=True,       # 是否使用视锥查询分支
                use_fsd=True,           # 是否使用LiDAR查询分支
                voxel_downsampling_size=None,
                is_argo=False,          # 是否为Argoverse2数据集（影响2D特征编码方式）
                max_refine_queries=None,
                ):
        super().__init__(
            backbone=backbone,
            segmentor=segmentor,
            voxel_layer=voxel_layer,
            voxel_encoder=voxel_encoder,
            middle_encoder=middle_encoder,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            cluster_assigner=cluster_assigner,
            pretrained=pretrained,
            init_cfg=init_cfg,
        )
        self.runtime_info = dict()
        self.tanh_dims = tanh_dims
        self.num_classes = num_classes
        self.num_cams = num_cams
        self.vis_dir = vis_dir
        self.encode_label_only = encode_label_only
        self.class_names = class_names
        self.min_pts = min_pts
        
        self.mlp_cfg = mlp_cfg
        self.embed_dims = self.mlp_cfg.get('embed_dims', 256)      # query特征维度，默认256
        self.norm_cfg = self.mlp_cfg.get('norm_cfg', dict(type='LN', eps=1e-3))
        self.act = self.mlp_cfg.get('act', 'gelu')
        # lidar_img_input_dim: LiDAR特征(128*3*2=768) + 图像特征(128) 拼接后的维度
        self.lidar_img_input_dim = self.mlp_cfg.get('lidar_img_input_dim', 128 * 3 * 2 + 128)
        self.lidar_input_dim = self.mlp_cfg.get('lidar_input_dim', 128 * 3 * 2)  # 纯LiDAR特征维度

        self.use_fsd = use_fsd

        # ---- 视锥查询分支（Camera Query Branch）----
        self.use_frustum = use_frustum
        self.frustum_obj_head = build_head(frustum_obj_head)  # 负责视锥 query 的分类和回归
        self.frustum_sir = build_head(frustum_sir)            # SIR 层：将一个 frustum 内的点特征聚合为 object-level 特征
        # 将 LiDAR+Image 拼接特征投影到 embed_dims 维度（用于多阶段精炼的残差连接）
        self.combine_frustum_feat_mlp = build_mlp(self.lidar_img_input_dim, [self.embed_dims, ], self.norm_cfg, act=self.act)

        # 2D特征编码 MLP：将 2D 检测框/分数/类别等信息编码为向量
        self.encode_2d_mlp_cfg = encode_2d_mlp_cfg
        self.encode_2d_mlp = build_mlp(
                                self.encode_2d_mlp_cfg['in_channel'],
                                self.encode_2d_mlp_cfg['mlp_channel'],
                                self.encode_2d_mlp_cfg['norm_cfg'],
                                is_head=False,
                                act=self.encode_2d_mlp_cfg['act']
                            )

        # ---- LiDAR 查询分支（LiDAR Query Branch）----
        # 将纯 LiDAR 特征投影到 embed_dims 维度（用于多阶段精炼的残差连接）
        self.combine_fsd_feat_mlp = build_mlp(self.lidar_input_dim, [self.embed_dims, ], self.norm_cfg, act=self.act)

        # ---- 图像增强分割（Image-Enhanced Segmentation）----
        # 将每个点从2D mask查询到的图像语义信息编码，通过残差加法更新点特征
        # 初始化权重为0确保训练初始阶段不破坏已预训练的分割特征
        self.segmentor_updated_mlp = build_mlp(segmentor_updated_mlp['in_channel'],
                                                segmentor_updated_mlp['mlp_channel'],
                                                segmentor_updated_mlp['norm_cfg'],
                                                is_head=True,
                                                act=segmentor_updated_mlp['act']
                                        )
        # 零初始化最后一层，保证训练开始时图像更新量为0（不影响分割预训练权重）
        nn.init.constant_(self.segmentor_updated_mlp[-1].weight, 0.)
        nn.init.constant_(self.segmentor_updated_mlp[-1].bias, 0.)

        # ---- 多阶段精炼（Multi-Stage Refinement）----
        self.fsd_begin_idx = fsd_begin_idx  # FSD query 的 obj_id 从此值开始，用于区分两路 query
        self.num_extra_stages = len(refined_obj_head)
        if self.num_extra_stages > 0:
            self.bbox_coder = build_bbox_coder(bbox_coder)
            self.roi_extractor = build_roi_extractor(roi_extractor)  # 从点云中按 bbox ROI 提取点
            # 每个精炼 stage 对应一个 SIR 层
            self.refine_sir_layers = nn.ModuleList([build_head(single_refine_sir_layer) for _ in range(self.num_extra_stages)])

            self.refine_encode_2d_mlp_cfg = refine_encode_2d_mlp_cfg
            # 精炼阶段各 stage 的 2D 图像特征编码 MLP
            self.refine_img_mlp = nn.ModuleList([build_mlp(
                                                    self.refine_encode_2d_mlp_cfg['in_channel'],
                                                    self.refine_encode_2d_mlp_cfg['mlp_channel'],
                                                    self.refine_encode_2d_mlp_cfg['norm_cfg'],
                                                    is_head=False,
                                                    act=self.refine_encode_2d_mlp_cfg['act']
                                                ) for _ in range(self.num_extra_stages)])
            # 各 stage 的 LiDAR+Image 特征融合 MLP
            self.lidar_img_mlp = nn.ModuleList([build_mlp(self.lidar_input_dim, [self.embed_dims, self.embed_dims], self.norm_cfg, act=self.act) for _ in range(self.num_extra_stages)])
            # 位置编码：将 query 中心坐标(x,y,z) 编码为位置嵌入，注入 query 特征
            self.position_encoder = nn.ModuleList([build_mlp(3, [self.embed_dims, self.embed_dims], self.norm_cfg, act=self.act) for _ in range(self.num_extra_stages)])
            # 输出投影层：融合当前特征、残差特征、位置特征
            self.out_proj = nn.ModuleList([build_mlp(self.embed_dims, [self.embed_dims, self.embed_dims], self.norm_cfg, act=self.act, is_head=True) for _ in range(self.num_extra_stages)])
            # 各精炼 stage 的检测头
            self.frustum_refined_head = nn.ModuleList([build_head(refined_obj_head[idx]) for idx in range(self.num_extra_stages)])
        self.tta_test_cfg = tta_test_cfg
        self.voxel_downsampling_size = voxel_downsampling_size
        self.is_argo = is_argo
        self.max_refine_queries = max_refine_queries

    def prj_points_2d(self, points, lidar2img, img_h, img_w):
        """将 LiDAR 点云投影到各相机图像上，获取归一化 2D 坐标。

        对应论文 Section 3.2 的视锥分组步骤：
        通过 lidar2img 变换矩阵将 3D 点投影到 2D 图像坐标系，
        再归一化到 [-1, 1] 范围（用于 F.grid_sample 的坐标格式）。

        Args:
            points: (N, 3) LiDAR 坐标系下的点坐标
            lidar2img: (6, 4, 4) 从 LiDAR 到各相机图像的变换矩阵
            img_h, img_w: 图像高宽
        Returns:
            pts_2d: (6, N, 2) 各相机下的归一化 2D 坐标，无效点设为 -2
        """
        # N,3 -> N,4，齐次坐标
        pts_4d = torch.cat([points, torch.ones_like(points[..., 0:1])], dim=-1)
        # 投影到各相机: (6,4,4) x (N,4)^T -> (6,N,4)
        pts_2d = pts_4d @ lidar2img.permute(0, 2, 1)
        # 深度必须为正才有效
        depth_valid_mask = pts_2d[..., 2] > 1e-3

        # 透视除法：将齐次坐标转为像素坐标
        max_depth = torch.finfo(pts_2d.dtype).max
        pts_2d[..., 2] = torch.clip(pts_2d[..., 2], min=1e-5, max=max_depth)
        pts_2d[..., 0] /= pts_2d[..., 2]  # u = X / Z
        pts_2d[..., 1] /= pts_2d[..., 2]  # v = Y / Z

        # 归一化到 [0,1]
        pts_2d[..., 0] /= img_w
        pts_2d[..., 1] /= img_h

        pts_2d = pts_2d[..., :2]
        # 转为 F.grid_sample 要求的 [-1, 1] 范围
        pts_2d = (pts_2d - 0.5) * 2

        # 超出图像范围的点设为无效
        img_valid_mask = ((pts_2d[..., 0:1] > -1.0)
                 & (pts_2d[..., 0:1] < 1.0)
                 & (pts_2d[..., 1:2] > -1.0)
                 & (pts_2d[..., 1:2] < 1.0)).squeeze(-1)

        valid_mask = depth_valid_mask & img_valid_mask
        # 无效点设为 -2（超出 grid_sample 范围，采样值为 0）
        pts_2d[~valid_mask] = -2.0
        return pts_2d

    def points_in_mask(self, points, mask_data, lidar2img):
        """design for a single batch with 6 cams
        mask_data: 6, H, W
        return:
        obj_id_of_pts: 6, N
        """

        num_cams, num_classes, img_h, img_w = mask_data.shape

        #6, N, 2
        pts_2d = self.prj_points_2d(points, lidar2img, img_h, img_w)

        obj_id_list = []
        for cam_id in range(num_cams):
            pts_2d_cam = pts_2d[cam_id].unsqueeze(0).unsqueeze(1)
            mask_cam = mask_data[cam_id].to(dtype=pts_2d.dtype).unsqueeze(0)
            #mask_cam: 1, 10, 900, 1600
            #pts_2d_cam: 1, 1, N, 2
            obj_id_cam = F.grid_sample(mask_cam, pts_2d_cam, mode='nearest')
            obj_id_cam = obj_id_cam.squeeze(2).long()
            obj_id_list.append(obj_id_cam)
        #6, 10, N
        obj_id_tensor = torch.cat(obj_id_list, dim=0)
        return obj_id_tensor.permute(2, 0, 1) #N, 6, 10

    def frustum_gather(self,
                        batch_idx,
                        points,
                        mask_data,
                        mask_anno,
                        img_metas):
        """视锥分组（Frustum Grouping）：为每个点查询其所属的 2D 实例 mask ID。

        对应论文 Section 3.2 的核心步骤：
        每个 LiDAR 点通过投影找到它在各相机图像中对应的像素，
        再从预计算好的实例 mask 图上读取该像素的 mask（实例 ID），
        从而将点云按 2D 实例进行分组。

        Args:
            batch_idx: (N,) 各点所属 batch 索引
            points: (N, 3+) 点坐标（使用原始无增广坐标用于投影）
            mask_data: (B, 6, C, H, W) 预计算的 2D 实例 mask 图（C=类别数）
            mask_anno: (B, max_objs, 9) 2D 检测框标注信息
            img_metas: 图像元信息（含 lidar2img 变换矩阵）
        Returns:
            obj_id_tensor: (N, 6, C) 每个点在各相机各类别下的实例 ID
        """
        device = batch_idx.device
        bz, num_cams, num_classes = mask_data.shape[0:3]
        obj_id_tensor = batch_idx.new_zeros((batch_idx.shape[0], num_cams, num_classes))
        for bidx in range(bz):
            # 按 batch 分别处理
            bz_mask = (batch_idx == bidx)
            points_bz = points[bz_mask]
            mask_data_bz = mask_data[bidx]
            img_metas_bz = img_metas[bidx]

            # 获取该 batch 的 LiDAR→Image 变换矩阵
            lidar2img_bz = torch.tensor(
                [data for data in img_metas_bz['lidar2img']],
                device=device,
                dtype=torch.float32
            )
            # 用 grid_sample 从 mask 图上采样每个点对应的实例 ID
            obj_id_tensor_bz = self.points_in_mask(points_bz[:, :3],
                                mask_data_bz,
                                lidar2img_bz,
            )
            obj_id_tensor[bz_mask] = obj_id_tensor_bz
        return obj_id_tensor

    def double_overlap_pts(self, 
                        pts_feat, 
                        bz_coor, 
                        points, 
                        obj_id_tensor, 
                        point_fg_weights
                    ):
        obj_id_tensor = obj_id_tensor.reshape(obj_id_tensor.shape[0], -1)
        overlaps_tensor = (obj_id_tensor > 0).sum(-1)
        max_overlap_num = overlaps_tensor.max() + 1
        pts_feat_clone = pts_feat.clone()
        bz_coor_clone = bz_coor.clone()
        points_clone = points.clone()
        point_fg_weights_clone = point_fg_weights.clone()

        raw_obj_id_tensor = obj_id_tensor.max(-1)[0]
        for overlap_num in range(2, max_overlap_num):
            overlaps_mask = overlaps_tensor == overlap_num
            if overlaps_mask.sum() == 0:
                continue

            pad_pts_feat = pts_feat_clone[overlaps_mask].clone().repeat(overlap_num -1, 1)
            pad_bz_coor = bz_coor_clone[overlaps_mask].clone().repeat(overlap_num -1, 1)
            pad_points = points_clone[overlaps_mask].clone().repeat(overlap_num -1, 1)
            pad_point_fg_weights = point_fg_weights_clone[overlaps_mask].clone().repeat(overlap_num -1)

            pts_feat = torch.cat([pts_feat, pad_pts_feat], dim=0)
            bz_coor = torch.cat([bz_coor, pad_bz_coor], dim=0)
            point_fg_weights = torch.cat([point_fg_weights, pad_point_fg_weights], dim=0)
            points = torch.cat([points, pad_points], dim=0)

            #generate pad_obj_id_tensor
            sort_value = obj_id_tensor[overlaps_mask].topk(overlap_num, dim=-1)[0]
            for pad_idx in range(1, overlap_num):
                pad_obj_id_tensor = sort_value[:, pad_idx].clone()
                raw_obj_id_tensor = torch.cat([raw_obj_id_tensor, pad_obj_id_tensor], dim=0)

        return pts_feat, bz_coor, points, raw_obj_id_tensor, point_fg_weights

    def extract_fg_pts(self, 
                    pts_feat, 
                    bz_coor, 
                    points, 
                    obj_id_tensor, 
                    point_fg_weights
                ):
        fg_mask = obj_id_tensor.sum((-2, -1)) > 0
        return pts_feat[fg_mask], bz_coor[fg_mask], points[fg_mask], \
            obj_id_tensor[fg_mask], point_fg_weights[fg_mask]

    def limit_refine_queries(self, obj_centers, obj_coors, obj_result, obj_feats, preds_2d):
        """Limit per-batch queries entering the memory-heavy refine ROI stage."""
        max_queries = self.max_refine_queries
        if max_queries is None or max_queries <= 0 or obj_coors.numel() == 0:
            return obj_centers, obj_coors, obj_result, obj_feats, preds_2d

        batch_values = obj_coors[:, 0].long()
        batch_size = 0
        if len(obj_result) > 0:
            batch_size = len(next(iter(obj_result.values())))
        batch_size = max(batch_size, int(batch_values.max().item()) + 1)

        selected_global = []
        selected_local_by_batch = []
        score_lists = obj_result.get('cls_logits', None)

        for bidx in range(batch_size):
            batch_indices = (batch_values == bidx).nonzero(as_tuple=False).squeeze(-1)
            if batch_indices.numel() == 0:
                selected_local_by_batch.append(batch_indices)
                continue

            count = batch_indices.numel()
            if score_lists is not None and bidx < len(score_lists) and score_lists[bidx].numel() > 0:
                score_tensor = score_lists[bidx]
                count = min(count, score_tensor.shape[0])
                scores = torch.sigmoid(score_tensor[:count]).amax(dim=-1)
            else:
                scores = torch.arange(count, device=batch_indices.device, dtype=obj_feats.dtype)

            if count > max_queries:
                keep_local = torch.topk(scores, k=max_queries, largest=True, sorted=False).indices
                keep_local = keep_local.sort()[0]
            else:
                keep_local = torch.arange(count, device=batch_indices.device, dtype=torch.long)

            selected_local_by_batch.append(keep_local)
            selected_global.append(batch_indices[:count].index_select(0, keep_local.to(device=batch_indices.device)))

        if len(selected_global) == 0:
            return obj_centers, obj_coors, obj_result, obj_feats, preds_2d

        selected_global = torch.cat(selected_global, dim=0)
        limited_result = {}
        for key, values in obj_result.items():
            limited_values = []
            for bidx, value in enumerate(values):
                if bidx < len(selected_local_by_batch):
                    keep_local = selected_local_by_batch[bidx].to(device=value.device)
                    keep_local = keep_local[keep_local < value.shape[0]]
                    limited_values.append(value.index_select(0, keep_local))
                else:
                    limited_values.append(value)
            limited_result[key] = limited_values

        return (
            obj_centers.index_select(0, selected_global.to(device=obj_centers.device)),
            obj_coors.index_select(0, selected_global.to(device=obj_coors.device)),
            limited_result,
            obj_feats.index_select(0, selected_global.to(device=obj_feats.device)),
            preds_2d.index_select(0, selected_global.to(device=preds_2d.device)),
        )

    def map_voxel_center_to_point(self, voxel_mean, voxel2point_inds):
        return voxel_mean[voxel2point_inds]

    def get_cluster_delta_weighted(self, points, sir_coors, point_weights):
        #compute all frames
        point_weights = point_weights.clamp(min=1e-5).detach()
        input_feat = torch.cat([points[:, :3] * point_weights,
                                point_weights], dim=-1)
        voxel_mean_feat, voxel_mean_coors, unq_inv = scatter_v2(
                                            input_feat, 
                                            sir_coors, 
                                            mode='avg', 
                                        )
        voxel_center = voxel_mean_feat[:, :3] / voxel_mean_feat[:, 3:4]

        points_center = self.map_voxel_center_to_point(
            voxel_center, unq_inv)

        f_cluster = (points[:, :3] - points_center[:, :3])
        return f_cluster, voxel_center, voxel_mean_coors
    
    def get_cluster_delta_avg(self, points, sir_coors, point_weights):
        input_feat = points[:, :3]
        voxel_mean_feat, voxel_mean_coors, unq_inv = scatter_v2(
                                            input_feat, 
                                            sir_coors, 
                                            mode='avg', 
                                        )
        voxel_center = voxel_mean_feat[:, :3]
        points_center = self.map_voxel_center_to_point(
            voxel_center, unq_inv)
        
        #visualize the obj of the big delta 
        f_cluster = (points[:, :3] - points_center[:, :3])
        return f_cluster, voxel_center, voxel_mean_coors
    
    def get_point_fg_weights(self, seg_logits):
        """
        seg_logits : N, 11
        return 
        point_weights: N, 1
        """
        seg_logits_softmax = seg_logits.softmax(1)
        seg_prob_bg = seg_logits_softmax[:, -1]
        seg_prob_fg = 1 - seg_prob_bg
        return seg_prob_fg

    def get_sir_coors(self,
                      bz_coor,
                      obj_id_tensor,
                      point_fg_weights,
                      ):
        sir_coors = torch.cat([bz_coor, 
                               torch.zeros_like(bz_coor), 
                               obj_id_tensor.unsqueeze(-1),], dim=-1)
        return sir_coors, obj_id_tensor     

    def get_cluster_delta_from_center(self,
                                    points,
                                    sir_coors,
                                    cluster_center
                                    ):
        pts_mean_feat, pts_mean_coors, unq_inv = scatter_v2(
                                                points, 
                                                sir_coors, 
                                                mode='avg', 
                                            )

        points_center = self.map_voxel_center_to_point(
            cluster_center, unq_inv)

        points_delta = (points[:, :3] - points_center[:, :3])
        return points_delta

    def frustum_pooling(self,
                        pts_feat,
                        bz_coor,
                        points,
                        obj_id_tensor,
                        point_fg_weights,
                        img_metas=None,
                        cluster_center=None,
                    ):
        """视锥池化（Frustum Pooling）：以实例 mask ID 为 key，聚合同一实例的点特征。

        对应论文 Section 3.2 的 SIR 聚合步骤：
        1. 先过滤掉不属于任何实例的背景点（extract_fg_pts）
        2. 对出现在多个 mask 中的点进行复制（double_overlap_pts），使其可归属多个实例
        3. 以 (batch_id, obj_id) 为 key 计算加权中心（get_cluster_delta_weighted），
           得到每个点相对其所属实例中心的偏移量
        4. 送入 frustum_sir（SIR层）做集合特征提取，输出实例级别特征

        Args:
            pts_feat: (N, C) 点特征
            obj_id_tensor: (N, 6, C_cls) 每点在各相机各类别的实例 ID
            point_fg_weights: (N,) 点的前景概率权重（来自分割分支 softmax）
        Returns:
            final_cluster_feats: (K, C) K 个实例的聚合特征
            out_coors: (K, 3) 各实例的坐标索引 (batch, ?, obj_id)
            cluster_center: (K, 3) 各实例的加权中心坐标
        """
        # 只保留属于至少一个前景实例的点
        pts_feat, bz_coor, points, obj_id_tensor, point_fg_weights = \
                self.extract_fg_pts(pts_feat,
                                    bz_coor,
                                    points,
                                    obj_id_tensor,
                                    point_fg_weights,
                                    )
        if obj_id_tensor.sum() == 0:
            # 若视锥分支无任何前景输出，构造空占位输出，避免后续崩溃
            fake_num = 1
            points = points.new_zeros(fake_num, points.shape[-1])
            pts_feat = pts_feat.new_zeros(fake_num, pts_feat.shape[-1])
            sir_coors = bz_coor.new_zeros(fake_num, 3)
            points_delta = points.new_zeros(fake_num, 3)
            cluster_center = points.new_zeros(fake_num, 3)
        else:
            # 对同时出现在多个 mask 中的点进行复制，使每份只归属一个实例
            pts_feat, bz_coor, points, obj_id_tensor, point_fg_weights = \
                self.double_overlap_pts(pts_feat,
                                        bz_coor,
                                        points,
                                        obj_id_tensor,
                                        point_fg_weights)

            # 构造 SIR 坐标 (batch_id, 0, obj_id)，用于按实例分组
            sir_coors, obj_id_tensor = self.get_sir_coors(bz_coor,
                                        obj_id_tensor,
                                        point_fg_weights)

            if cluster_center is None:
                # 以前景概率为权重计算每个实例的加权质心，得到点-中心偏移
                points_delta, cluster_center, cluster_coors\
                                = self.get_cluster_delta_weighted(
                                            points,
                                            sir_coors,
                                            point_fg_weights.unsqueeze(-1)
                                        )
            else:
                # 精炼阶段：直接用解码后的 bbox 中心计算偏移
                points_delta = self.get_cluster_delta_from_center(
                                        points,
                                        sir_coors,
                                        cluster_center
                                    )

        # SIR（Set-in-Region）：对同一实例内的点做集合特征提取
        # 输入：点坐标、点特征、分组坐标、点-中心偏移；输出：实例级特征
        out_feats, final_cluster_feats, out_coors = \
            self.frustum_sir(points, pts_feat, sir_coors, f_cluster=points_delta)

        if out_coors.shape[0] == 0:
            out_coors = out_coors.new_zeros((0, 3))
        return final_cluster_feats, out_coors, cluster_center

    def encode_preds_2d(self, preds_2d, img_w, img_h, encode_single_cls=True):
        """select and encode preds_2d
        preds_2d: K, 9
        return 
        feats_2d: K, 128,
        """
        ##[0: 4],    4,        5,      6,      7,         8,
        ## bbox, score, category, cam_id, obj_id, valid_flag
        bbox_2d, score, category, cam_id = \
            preds_2d[:, :4], preds_2d[:, 4:5], preds_2d[:, 5], preds_2d[:, 6]
        en_bbox_2d = bbox_2d.clone()
        en_bbox_2d[:, 0::2] /= img_w
        en_bbox_2d[:, 1::2] /= img_h
        en_score = score
        en_category = F.one_hot(category.long(), num_classes=self.num_classes + 1)
        en_cam_id = F.one_hot(cam_id.long(), num_classes=self.num_cams)
        if self.encode_label_only:
            en_feat = en_category.float()
        elif encode_single_cls:
            en_feat = torch.cat(
                [en_bbox_2d, en_score, en_category.float()], dim=-1
            )
        else:
            #encode ten cls score
            en_feat = en_score
        return en_feat

    def get_single_cls_preds_2d(self, mask_anno, obj_coors):
        """
        mask_anno: bz, 250, 9
        obj_coors, K, 12
        """
        batch_tensor, obj_id_tensor = obj_coors[:, 0], obj_coors[:, 2]
        num_objs = obj_coors.shape[0]
        num_mask_annos = mask_anno.shape[-1]
        batch_size = mask_anno.shape[0]
        preds_2d_flat = torch.zeros(
                        (num_objs, num_mask_annos), 
                        dtype=torch.float32, 
                        device=obj_coors.device
                    )

        for bidx in range(batch_size):
            bz_mask = batch_tensor == bidx
            
            preds_2d_flat_bz = preds_2d_flat[bz_mask]
            obj_id_tensor_bz = obj_id_tensor[bz_mask] - 1

            valid_mask = obj_id_tensor_bz >= 0
            preds_2d_flat_bz[valid_mask] = mask_anno[bidx][obj_id_tensor_bz[valid_mask]]
            preds_2d_bz_invalid = preds_2d_flat_bz[~valid_mask]
            preds_2d_bz_invalid[:, 5] = self.num_classes
            preds_2d_flat_bz[~valid_mask] = preds_2d_bz_invalid

            preds_2d_flat[bz_mask] = preds_2d_flat_bz
        return preds_2d_flat
    
    def get_all_cls_preds_2d(self, mask_anno, batch_tensor, obj_id_tensor):
        """
        mask_anno: bz, 250, 9
        obj_coors, K, 10
        """
        num_objs = batch_tensor.shape[0]
        num_mask_annos = mask_anno.shape[-1]
        num_classes = obj_id_tensor.shape[-1]
        batch_size = mask_anno.shape[0]
        preds_2d_flat = torch.zeros(
                        (num_objs, num_classes, num_mask_annos), 
                        dtype=torch.float32, 
                        device=batch_tensor.device
                    )

        for bidx in range(batch_size):
            bz_mask = batch_tensor == bidx
            
            preds_2d_flat_bz = preds_2d_flat[bz_mask]
            obj_id_tensor_bz = obj_id_tensor[bz_mask] - 1

            valid_mask = obj_id_tensor_bz >= 0
            preds_2d_flat_bz[valid_mask] = mask_anno[bidx][obj_id_tensor_bz[valid_mask]]
            preds_2d_bz_invalid = preds_2d_flat_bz[~valid_mask]
            preds_2d_bz_invalid[:, 5] = num_classes
            preds_2d_flat_bz[~valid_mask] = preds_2d_bz_invalid


            preds_2d_flat[bz_mask] = preds_2d_flat_bz
        return preds_2d_flat #(num_objs, num_classes, num_mask_annos)

    def encode_2d_feats(self, preds_2d, img_w, img_h, encode_mlp):
        #preds_2d   (num_objs, num_classes, num_mask_annos)
        if len(preds_2d.shape) == 3:
            num_objs, num_classes, num_mask_annos = preds_2d.shape
            encoded_2d = self.encode_preds_2d(
                                preds_2d.reshape(-1, num_mask_annos),
                                img_w, 
                                img_h, 
                                encode_single_cls=self.is_argo
                            )
            if not self.is_argo:
                encoded_2d = encoded_2d.reshape(num_objs, num_classes)
        else:
            encoded_2d = self.encode_preds_2d(preds_2d, img_w, img_h)
        encoded_2d_feat = encode_mlp(encoded_2d)
        return encoded_2d_feat

    def split_points_last_3dim(self, points):
        no_aug_points = []
        new_points = []
        for point in points:
            no_aug_points.append(point[:, -3:])
            new_points.append(point[:, :-3])
        return new_points, no_aug_points

    def combine_by_batch(self, data_list, batch_idx, batch_size):
        data_list_flat = data_list[0].new_zeros((batch_idx.shape[0], data_list[0].shape[-1]))
        for bidx in range(batch_size):
            bz_mask = batch_idx == bidx
            data_list_flat[bz_mask] = data_list[bidx]
        return data_list_flat   

    def fsd_forward(self,
                    seg_out_dict,
                    img_metas
                    ):
        """激光雷达查询分支前向传播（LiDAR Query Branch Forward）。

        对应论文 Section 3.3：利用图聚类算法（connected components）将前景点按
        实例聚类，再用 SIR 层提取实例级别特征，最后用 bbox_head 做检测。
        此分支完全不依赖图像，是纯 LiDAR 的实例查询。

        流程：分割结果 → 采样前景点 → 图聚类 → SIR聚合 → bbox_head检测
        """
        dict_to_sample = dict(
            seg_points=seg_out_dict['seg_points'],
            seg_logits=seg_out_dict['seg_logits'].detach(),
            seg_vote_preds=seg_out_dict['seg_vote_preds'].detach(),
            seg_feats=seg_out_dict['seg_feats'],
            batch_idx=seg_out_dict['batch_idx'],
            vote_offsets=seg_out_dict['offsets'].detach(),
        )
        if self.cfg.get('pre_voxelization_size', None) is not None:
            dict_to_sample = self.pre_voxelize(dict_to_sample)
        # 从分割结果中采样前景点，并计算投票后的中心位置预测
        sampled_out = self.sample(dict_to_sample, dict_to_sample['vote_offsets'])

        # 图聚类：将前景点按空间近邻关系聚类为实例（connected components）
        # 返回每个点的簇属(cls_id, batch_id, cluster_id)
        cluster_inds_list, valid_mask_list = self.cluster_assigner(sampled_out['center_preds'], sampled_out['batch_idx'], origin_points=sampled_out['seg_points'])
        pts_cluster_inds = torch.cat(cluster_inds_list, dim=0)  # [N, 3]: (cls_id, batch_idx, cluster_id)

        sampled_out = self.update_sample_results_by_mask(sampled_out, valid_mask_list)
        combined_out = self.combine_classes(sampled_out, ['seg_points', 'seg_logits', 'seg_vote_preds', 'seg_feats', 'center_preds'])

        points = combined_out['seg_points']
        # 拼接分割 logit、投票预测和点特征作为 SIR 输入
        pts_feats = torch.cat([combined_out['seg_logits'], combined_out['seg_vote_preds'], combined_out['seg_feats']], dim=1)
        assert len(pts_cluster_inds) == len(points) == len(pts_feats)

        # SIR 层：按实例分组提取特征
        extracted_outs = self.extract_feat(points, pts_feats, pts_cluster_inds, img_metas, combined_out['center_preds'])
        cluster_feats = extracted_outs['cluster_feats']
        cluster_xyz = extracted_outs['cluster_xyz']    # 实例中心坐标
        cluster_inds = extracted_outs['cluster_inds']  # [class, batch, group_id]

        assert (cluster_inds[:, 0]).max().item() < self.num_classes
        obj_feat = cluster_feats
        outs = self.bbox_head(obj_feat)  # 分类分数 + 回归偏移
        return obj_feat, cluster_xyz, cluster_inds, outs

    def frustum_forward(self,
                        seg_out_dict,
                        mask_anno,
                        mask_data,
                        point_infos,
                        img_metas,
                        cluster_center=None,
                        ):
        pts_feat = seg_out_dict['seg_feats']
        batch_idx = seg_out_dict['batch_idx']
        points = seg_out_dict['seg_points']
        seg_logits = seg_out_dict['seg_logits']

        point_fg_weights = self.get_point_fg_weights(seg_logits)

        batch_size = mask_anno.shape[0]
        points_info_flat = self.combine_by_batch(point_infos, batch_idx, batch_size)

        #grouping
        obj_id_tensor = self.frustum_gather(batch_idx, 
                                            points_info_flat, #no_aug_points
                                            mask_data, 
                                            mask_anno, 
                                            img_metas)
                
        #pooling
        lidar_feat, obj_coors, obj_centers = self.frustum_pooling(pts_feat, 
                                                    batch_idx.unsqueeze(-1), 
                                                    points, 
                                                    obj_id_tensor,
                                                    point_fg_weights,
                                                    img_metas,
                                                    cluster_center
                                                    )

        #get preds 2d
        preds_2d = self.get_single_cls_preds_2d(mask_anno, obj_coors)
       
        img_feat = self.encode_2d_feats(
            preds_2d,
            img_w=mask_data.shape[-1],
            img_h=mask_data.shape[-2],
            encode_mlp=self.encode_2d_mlp,
        )
        lidar_img_feat = torch.cat([lidar_feat, img_feat], dim=-1)
        obj_feat = lidar_img_feat
        frustum_obj_result = self.frustum_obj_head(obj_feat)
        
        return obj_feat, obj_centers, obj_coors, frustum_obj_result, preds_2d

    def combine_frustum_and_fsd(self,
                                frustum_obj_centers,
                                frustum_obj_coors,
                                frustum_obj_result,
                                frustum_obj_feats,
                                frustum_preds_2d,
                                fsd_obj_centers,
                                fsd_obj_coors,
                                fsd_obj_result,
                                fsd_obj_feats,
                            ):
        """合并视锥查询和 LiDAR 查询两路 query（对应论文 Figure 2 的 Combine 步骤）。

        关键设计：
        - 直接拼接两路 query 的中心坐标、特征和检测结果
        - FSD query 的 obj_id 加上 fsd_begin_idx 作为偏移，避免与 Frustum query 冲突
        - 将两路特征分别经过各自的 MLP 投影到相同的 embed_dims 维度再拼接
        """
        obj_centers = torch.cat([frustum_obj_centers, fsd_obj_centers], dim=0)

        # 调整 FSD query 的坐标格式并增加 obj_id 偏移
        fsd_obj_coors_re = fsd_obj_coors.clone()
        fsd_obj_coors_re[:, 0] = fsd_obj_coors[:, 1]  # batch_id 移到第0列
        fsd_obj_coors_re[:, 1] = fsd_obj_coors[:, 0]  # cls_id 移到第1列
        fsd_obj_coors_re[:, 2] += self.fsd_begin_idx   # obj_id 加偏移以区分两路 query
        obj_coors = torch.cat([frustum_obj_coors, fsd_obj_coors_re], dim=0)

        # 合并两路的检测结果（各 batch 内按就拼接）
        obj_result = {}
        for key in frustum_obj_result.keys():
            batch_size = len(frustum_obj_result[key])
            obj_result[key] = []
            for bidx in range(batch_size):
                data = torch.cat([frustum_obj_result[key][bidx], fsd_obj_result[key][bidx]], dim=0)
                obj_result[key].append(data)

        # 两路特征分别经过各自 MLP 投影到 embed_dims 维度，再拼接
        frustum_obj_feats_com = self.combine_frustum_feat_mlp(frustum_obj_feats)
        fsd_obj_feats_com = self.combine_fsd_feat_mlp(fsd_obj_feats)
        obj_feats = torch.cat([frustum_obj_feats_com, fsd_obj_feats_com], dim=0)

        # FSD query 没有2D预测信息，用零占位
        fsd_preds_2d = frustum_preds_2d.new_zeros((fsd_obj_feats.shape[0], frustum_preds_2d.shape[1]))
        preds_2d = torch.cat([frustum_preds_2d, fsd_preds_2d], dim=0)

        return obj_centers, obj_coors, obj_result, obj_feats, preds_2d

    def img_cross_attn(self,
                      point_infos,
                      batch_idx,
                      mask_anno,
                      mask_data,
                      img_metas,
                      encode_mlp,
                      ext_pts_inds=None,
                      ):
        batch_size = mask_anno.shape[0]
        points_info_flat = self.combine_by_batch(point_infos, batch_idx, batch_size)
        if ext_pts_inds is not None:
            points_info_flat = points_info_flat[ext_pts_inds]
            batch_idx = batch_idx[ext_pts_inds]
        #grouping
        obj_id_tensor = self.frustum_gather(batch_idx, 
                                            points_info_flat, #no_aug_points
                                            mask_data, 
                                            mask_anno, 
                                            img_metas)
        #N, 6, 10
        _, num_cams, num_classes = obj_id_tensor.shape
        cam_select_value = obj_id_tensor.sum(-1).max(-1)[1] 
        cam_select_mask = F.one_hot(cam_select_value, num_cams).bool().unsqueeze(-1) 
        points_obj_id_multi_cls = obj_id_tensor.masked_select(cam_select_mask).reshape(-1, num_classes)   #N, 10      
        preds_2d = self.get_all_cls_preds_2d(mask_anno, batch_idx, points_obj_id_multi_cls)

        img_feat = self.encode_2d_feats(
            preds_2d,
            img_w=mask_data.shape[-1],
            img_h=mask_data.shape[-2],
            encode_mlp=encode_mlp,
        )

        return img_feat

    def segmentor_feat_inhance_train(self, seg_out_tuple, point_infos, mask_anno, mask_data, img_metas):
        (neck_out, pts_coors, points, labels, vote_targets, vote_mask) = seg_out_tuple
        batch_idx = pts_coors[:, 0]
        losses = dict()

        pts_lidar_feats = neck_out[0]
        valid_pts_mask = neck_out[1]
        points = points[valid_pts_mask]
        pts_coors = pts_coors[valid_pts_mask]
        labels = labels[valid_pts_mask]
        vote_targets = vote_targets[valid_pts_mask]
        vote_mask = vote_mask[valid_pts_mask]
        assert pts_lidar_feats.size(0) == labels.size(0)

        pts_updated_feats = self.img_cross_attn(point_infos,
                            batch_idx,
                            mask_anno,
                            mask_data,
                            img_metas,
                            encode_mlp=self.segmentor_updated_mlp,
                            )
        
        pts_feats = pts_lidar_feats + pts_updated_feats

        loss_decode, preds_dict = self.segmentor.segmentation_head.forward_train(pts_feats, img_metas, labels, vote_targets, vote_mask, return_preds=True)
        losses.update(loss_decode)

        vote_preds = preds_dict['vote_preds']

        offsets = self.segmentor.segmentation_head.decode_vote_targets(vote_preds)

        output_dict = dict(
            seg_points=points,
            seg_logits=preds_dict['seg_logits'],
            seg_vote_preds=preds_dict['vote_preds'],
            offsets=offsets,
            seg_feats=pts_feats,
            batch_idx=pts_coors[:, 0],
            losses=losses
        )
        return output_dict

    def segmentor_feat_inhance_test(self, seg_out_tuple, point_infos, mask_anno, mask_data, img_metas):
        (neck_out, pts_coors, points) = seg_out_tuple
        batch_idx = pts_coors[:, 0]
        
        
        pts_lidar_feats = neck_out[0]
        valid_pts_mask = neck_out[1]
        points = points[valid_pts_mask]
        pts_coors = pts_coors[valid_pts_mask]

        pts_updated_feats = self.img_cross_attn(point_infos,
                            batch_idx,
                            mask_anno,
                            mask_data,
                            img_metas,
                            encode_mlp=self.segmentor_updated_mlp,
                            )
        
        pts_feats = pts_lidar_feats + pts_updated_feats

        seg_logits, vote_preds = self.segmentor.segmentation_head.forward_test(pts_feats, img_metas, self.segmentor.test_cfg)

        offsets = self.segmentor.segmentation_head.decode_vote_targets(vote_preds)

        output_dict = dict(
            seg_points=points,
            seg_logits=seg_logits,
            seg_vote_preds=vote_preds,
            offsets=offsets,
            seg_feats=pts_feats,
            batch_idx=pts_coors[:, 0],
        )
        return output_dict

    def forward_train(self,
                      points,
                      img_metas,
                      no_aug_gt_bboxes_3d,
                      no_aug_gt_labels_3d,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      mask_data,
                      mask_anno,
                      gt_bboxes_ignore=None,
                      img=None,):
        """训练入口（对应论文 Figure 2 的完整流程）。

        4个阶段：
        1. 分割 + 图像增强（segmentation + image-enhanced segmentation）
        2. 视锥查询（Camera Query）
        3. LiDAR查询（LiDAR Query）
        4. 合并 + 多阶段精炼（Multi-Stage Refinement）

        Args:
            no_aug_gt_bboxes_3d: 原始无增广的真实框，用于 Frustum 分气赋 targetr（观察2D投影赋目标时不用增广 mask）
            mask_data: (B,6,C,H,W) 预计算的 2D 实例 mask
            mask_anno: (B,max_obj,9) 2D 检测框的属性信息
        """
        if self.voxel_downsampling_size is not None:
            points = self.segmentor.voxel_downsample(points)
        # 将点云抆尾的最后3列（无增广坐标）分离出来，用于将点投影到图像
        points, point_infos = self.split_points_last_3dim(points)

        no_aug_gt_bboxes_3d = [b[l >= 0] for b, l in zip(no_aug_gt_bboxes_3d, no_aug_gt_labels_3d)]
        no_aug_gt_labels_3d = [l[l >= 0] for l in no_aug_gt_labels_3d]
        gt_bboxes_3d = [b[l >= 0] for b, l in zip(gt_bboxes_3d, gt_labels_3d)]
        gt_labels_3d = [l[l >= 0] for l in gt_labels_3d]

        losses = {}

        # --- 阶段1：分割 + 图像增强分割 ---
        # extract_feat_only=True: 只做点云编码，不做完整分割 forward
        seg_out_tuple = self.segmentor(points=points, img_metas=img_metas, gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d, as_subsegmentor=True, extract_feat_only=True)
        # 将 2D mask 信息通过 cross-attention 注入分割特征，然后做完整分割 forward
        seg_out_dict = self.segmentor_feat_inhance_train(seg_out_tuple, point_infos, mask_anno, mask_data, img_metas)

        seg_loss = seg_out_dict['losses']
        losses.update(seg_loss)

        pts_feat = seg_out_dict['seg_feats']
        batch_idx = seg_out_dict['batch_idx']
        points = seg_out_dict['seg_points']

        # --- 阶段2：视锥查询（Camera Query Branch）---
        frustum_obj_feats, frustum_obj_centers, frustum_obj_coors, frustum_obj_result, frustum_preds_2d \
                                                        = self.frustum_forward(seg_out_dict,
                                                                                mask_anno,
                                                                                mask_data,
                                                                                point_infos,
                                                                                img_metas,
                                                                                cluster_center=None)
        frustum_query_losses = self.frustum_obj_head.loss(
                                frustum_obj_result['cls_logits'],
                                frustum_obj_result['reg_preds'],
                                frustum_obj_centers,
                                frustum_obj_coors,
                                no_aug_gt_bboxes_3d,
                                no_aug_gt_labels_3d,
                                gt_bboxes_3d,
                                gt_labels_3d,
                                frustum_preds_2d,
                                img_metas,
                                )
        for loss_key, loss_value in frustum_query_losses.items():
            losses[f'frustum_' + loss_key] = loss_value

        # --- 阶段3： LiDAR 查询（LiDAR Query Branch）---
        fsd_obj_feats, fsd_obj_centers, fsd_obj_coors, fsd_obj_result = self.fsd_forward(seg_out_dict, img_metas)
        fsd_loss_inputs = (fsd_obj_result['cls_logits'], fsd_obj_result['reg_preds']) + (fsd_obj_centers, fsd_obj_coors) + (gt_bboxes_3d, gt_labels_3d, img_metas)
        fsd_query_loss = self.bbox_head.loss(
            *fsd_loss_inputs, iou_logits=fsd_obj_result.get('iou_logits', None), gt_bboxes_ignore=gt_bboxes_ignore)
        for loss_key, loss_value in fsd_query_loss.items():
            losses[f'fsd_' + loss_key] = loss_value

        # --- 阶段4：合并两路 query + 多阶段精炼 ---
        obj_centers, obj_coors, obj_result, obj_feats, preds_2d = \
            self.combine_frustum_and_fsd(
                frustum_obj_centers,
                frustum_obj_coors,
                frustum_obj_result,
                frustum_obj_feats,
                frustum_preds_2d,
                fsd_obj_centers,
                fsd_obj_coors,
                fsd_obj_result,
                fsd_obj_feats,
            )

        obj_centers, obj_coors, obj_result, obj_feats, preds_2d = self.limit_refine_queries(
            obj_centers, obj_coors, obj_result, obj_feats, preds_2d)

        if self.num_extra_stages > 0:
            multi_stage_losses = self.multi_stage_refine_train(obj_centers,
                                                    obj_coors,
                                                    obj_result,
                                                    points,
                                                    point_infos,
                                                    pts_feat,
                                                    batch_idx,
                                                    mask_data,
                                                    mask_anno,
                                                    no_aug_gt_bboxes_3d,
                                                    no_aug_gt_labels_3d,
                                                    gt_bboxes_3d,
                                                    gt_labels_3d,
                                                    preds_2d,
                                                    img_metas,
                                                    obj_feats,
                                                    )
            losses.update(multi_stage_losses)
        return losses
    
    def multi_stage_refine_train(self,
                            obj_centers,
                            obj_coors,
                            obj_result,
                            points,
                            point_infos,
                            pts_feat,
                            batch_idx,
                            mask_data,
                            mask_anno,
                            no_aug_gt_bboxes_3d,
                            no_aug_gt_labels_3d,
                            gt_bboxes_3d,
                            gt_labels_3d,
                            preds_2d,
                            img_metas,
                            res_query_feat,
                            ):
        mutli_stage_losses = {}
        for i_stage in range(self.num_extra_stages):
            old_obj_result = obj_result
            obj_centers, obj_result, res_query_feat\
                                        = self.each_stage_refine(i_stage,
                                                            obj_centers,
                                                            obj_coors,
                                                            obj_result,
                                                            points,
                                                            point_infos,
                                                            pts_feat,
                                                            batch_idx,
                                                            mask_data,
                                                            mask_anno,
                                                            img_metas,
                                                            res_query_feat,
                                                            )
            refined_query_losses = self.frustum_refined_head[i_stage].loss(
                                obj_result['cls_logits'],
                                obj_result['reg_preds'],
                                obj_centers,
                                obj_coors,
                                no_aug_gt_bboxes_3d,
                                no_aug_gt_labels_3d,
                                gt_bboxes_3d,
                                gt_labels_3d,
                                preds_2d,
                                img_metas,
                                obj_result.get('iou_logits', None),
                                old_obj_result['cls_logits'],
                                old_obj_result['reg_preds'],
                                )
            stage_refined_query_losses = {}
            for loss_key, loss_value in refined_query_losses.items():
                stage_refined_query_losses[f'stage_{i_stage}_' + loss_key] = loss_value
            mutli_stage_losses.update(stage_refined_query_losses)
        return mutli_stage_losses

    def multi_stage_refine_test(self,
                            obj_centers,
                            obj_coors,
                            obj_result,
                            points,
                            point_infos,
                            pts_feat,
                            batch_idx,
                            mask_data,
                            mask_anno,
                            preds_2d,
                            img_metas,
                            res_query_feat,
                            ):
        all_stage_bbox_list = []
        obj_centers_list = []
        obj_result_list = []
        for i_stage in range(self.num_extra_stages):
            old_obj_result = copy.deepcopy(obj_result)
            obj_centers, obj_result, res_query_feat\
                                        = self.each_stage_refine(i_stage,
                                                            obj_centers,
                                                            obj_coors,
                                                            obj_result,
                                                            points,
                                                            point_infos,
                                                            pts_feat,
                                                            batch_idx,
                                                            mask_data,
                                                            mask_anno,
                                                            img_metas,
                                                            res_query_feat,
                                                            )
            obj_centers_list.append(obj_centers)
            obj_result_list.append(obj_result)

            each_stage_bbox_list = self.frustum_refined_head[i_stage].get_bboxes(
                                            obj_result['cls_logits'],
                                            obj_result['reg_preds'],
                                            preds_2d, 
                                            obj_centers,
                                            obj_coors, 
                                            img_metas,
                                            iou_logits=obj_result.get('iou_logits', None)
                                            )
            all_stage_bbox_list.append(each_stage_bbox_list)
        return all_stage_bbox_list[-1]

    def query_feat_refine(self,
                        points,
                        pts_feat,
                        batch_idx,
                        input_bbox_rois,
                        i_stage,
                        point_infos,
                        mask_anno,
                        mask_data,
                        img_metas,
                        ):
        ext_pts_inds, ext_pts_roi_inds, ext_pts_info = self.roi_extractor(
                                                    points[:, :3], # intensity might be in pts_xyz
                                                    batch_idx,
                                                    input_bbox_rois[:, :8],  # [22,8]
                                                )

        extracted_points = points[ext_pts_inds]
        extracted_points_feats = pts_feat[ext_pts_inds]

        pts_img_feat = self.img_cross_attn(point_infos,
                                            batch_idx,
                                            mask_anno,
                                            mask_data,
                                            img_metas,
                                            self.refine_img_mlp[i_stage],
                                            ext_pts_inds,
                                            )
    
        ext_pts_feats_updated = torch.cat([extracted_points_feats, pts_img_feat], dim=-1)
        lidar_feat, lidar_mask = self.refine_sir_layers[i_stage](extracted_points,
                                        ext_pts_feats_updated,
                                        ext_pts_info,
                                        ext_pts_roi_inds,
                                        input_bbox_rois)
        return lidar_feat

    def each_stage_refine(self,
                        i_stage,
                        old_obj_centers,
                        obj_coors,
                        old_obj_result,
                        points,
                        point_infos,
                        pts_feat,
                        batch_idx,
                        mask_data,
                        mask_anno,
                        img_metas,
                        res_query_feat,
                        ):
        """单个精炼 stage 的前向传播（对应论文 Section 3.4 Multi-Stage Refinement）。

        每个 stage 的工作流程：
        1. 解码上一阶段的回归预测，得到更精确的 bbox ROI
        2. 用 roi_extractor 在点云中提取 ROI 内的点
        3. 用 img_cross_attn 引入图像特征，拼接到点特征上
        4. 用 refine_sir_layers 做点特征聚合，得到新的 LiDAR query 特征
        5. cur_query = lidar_img_mlp(lidar_feat)
           query_feat = out_proj(cur_query + res_feat + pos_feat)  # 残差连接
        6. 用 frustum_refined_head 做分类和回归

        Args:
            res_query_feat: 上一阶段的 query 特征（用于残差连接）
        Returns:
            obj_centers: 新的实例中心坐标
            obj_result: 新的分类+回归结果
            query_feat: 新的 query 特征（作为下一阶段的残差连接输入）
        """
        if len(old_obj_centers) == 0:
            lidar_img_feat = old_obj_centers.new_zeros((0, self.lidar_input_dim))
            obj_centers = old_obj_centers.clone()
        else:
            # 解码上一阶段的 bbox 预测得到新的 ROI
            input_bbox_rois = self.decode_stage_bboxes(old_obj_centers, obj_coors[:, 0], old_obj_result['reg_preds'])
            obj_centers = input_bbox_rois[:, 1:4]  # 更新 query 中心坐标

            # 提取 ROI 内点特征（LiDAR + Image 融合）
            lidar_img_feat = self.query_feat_refine(points,
                                                    pts_feat,
                                                    batch_idx,
                                                    input_bbox_rois,
                                                    i_stage,
                                                    point_infos,
                                                    mask_anno,
                                                    mask_data,
                                                    img_metas,)

        # 将新提取的点特征投影到 embed_dims
        cur_query_feat = self.lidar_img_mlp[i_stage](lidar_img_feat)
        # 中心坐标位置编码
        pos_feat = self.position_encoder[i_stage](obj_centers.detach())
        # 残差连接：当前特征、上阶段残差特征、位置特征三者相加再投影
        query_feat = self.out_proj[i_stage](cur_query_feat + res_query_feat + pos_feat)

        # 精炼阶段的分类 + 回归预测
        obj_result = self.frustum_refined_head[i_stage](query_feat)

        return obj_centers, obj_result, query_feat

    def decode_stage_bboxes(self, obj_centers, bz_coors, reg_preds):
        batch_size = len(reg_preds)
        decode_size = reg_preds[0].shape[-1] - 1
        bboxes_tensor = reg_preds[0].new_zeros((bz_coors.shape[0], decode_size), dtype=torch.float32)
        for bidx in range(batch_size):
            bz_mask = bz_coors == bidx
            bboxes_bz = self.bbox_coder.decode(reg_preds[bidx], obj_centers[bz_mask])
            bboxes_tensor[bz_mask] = bboxes_bz.to(dtype=bboxes_tensor.dtype)
        bboxes_tensor_roi = torch.cat([bz_coors.to(dtype=bboxes_tensor.dtype).unsqueeze(-1), bboxes_tensor], dim=-1)
        return bboxes_tensor_roi

    def forward_test(self,
                    points,
                    img_metas,
                    mask_data,
                    mask_anno,
                    **kwargs,
                ):
        num_augs = len(points)
        if num_augs == 1:
            return self.simple_test(points[0],
                           img_metas[0],
                           mask_data[0],
                           mask_anno[0],
                           **kwargs,
                    )
        else:
            return self.aug_test(points, img_metas, mask_data, mask_anno, **kwargs)

    def simple_test(self,
                    points,
                    img_metas,
                    mask_data,
                    mask_anno,
                    **kwargs,
                ):
        if self.voxel_downsampling_size is not None:
            points = self.segmentor.voxel_downsample(points)
        points, point_infos = self.split_points_last_3dim(points)
            
        ##1. segmentation
        seg_out_tuple = self.segmentor.simple_test(points, img_metas, extract_feat_only=True, rescale=False) 
        seg_out_dict = self.segmentor_feat_inhance_test(seg_out_tuple, point_infos, mask_anno, mask_data, img_metas)
        pts_feat = seg_out_dict['seg_feats']
        batch_idx = seg_out_dict['batch_idx']
        points = seg_out_dict['seg_points']

        ##2. camera queries
        frustum_obj_feats, frustum_obj_centers, frustum_obj_coors, frustum_obj_result, frustum_preds_2d \
                                                        = self.frustum_forward(seg_out_dict,
                                                                                mask_anno,
                                                                                mask_data,
                                                                                point_infos,
                                                                                img_metas,
                                                                                cluster_center=None,)

        ##3. LiDAR queries
        fsd_obj_feats, fsd_obj_centers, fsd_obj_coors, fsd_obj_result = self.fsd_forward(seg_out_dict, img_metas)

        ##4. Query Refinement
        obj_centers, obj_coors, obj_result, obj_feats, preds_2d = \
            self.combine_frustum_and_fsd(
                frustum_obj_centers,
                frustum_obj_coors,
                frustum_obj_result,
                frustum_obj_feats,
                frustum_preds_2d,
                fsd_obj_centers,
                fsd_obj_coors,
                fsd_obj_result,
                fsd_obj_feats,
            )

        obj_centers, obj_coors, obj_result, obj_feats, preds_2d = self.limit_refine_queries(
            obj_centers, obj_coors, obj_result, obj_feats, preds_2d)
            
        if self.num_extra_stages >= 0:
            bbox_list = self.multi_stage_refine_test(obj_centers,
                                                obj_coors,
                                                obj_result,
                                                points,
                                                point_infos,
                                                pts_feat,
                                                batch_idx,
                                                mask_data,
                                                mask_anno,
                                                preds_2d,
                                                img_metas,
                                                obj_feats,
                                                )
     
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]

        return bbox_results
    
