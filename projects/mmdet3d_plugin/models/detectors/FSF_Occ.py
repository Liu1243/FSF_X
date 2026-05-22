"""
FSF_Occ: Fully Sparse Fusion with Occupancy-Guided Amodal Completion
=====================================================================
在 FSF 基线基础上，将 SparseOcc 的稀疏占据感知思想注入视锥分支：

核心改动（仅重写 frustum_forward 和 forward_train/simple_test）：

    原始 FSF 视锥分支：
        视锥点集 P_j
            → 加权质心 C_obs（易偏出 GT 框）
            → SIR 池化 → 实例特征

    FSF_Occ 视锥分支：
        视锥点集 P_j
            ↓ OccMLP       → 过滤背景虚空点 → P_valid（纯净前景）
            ↓ AmodalCenter → 修正中心 C_pred = C_obs + Δc
            ↓ SIR 池化（cluster_center=C_pred）→ 更准确的实例特征

    额外训练信号：
        L_occ          (Focal Loss)  监督占据预测
        L_amodal_center (Smooth-L1)  监督中心修正

    总损失：
        L_total_new = L_total + λ1·L_occ + λ2·L_amodal_center
"""

import torch
from torch import nn
from mmdet.models import DETECTORS
from mmdet3d.core.bbox import bbox3d2result

from .FSF import FSF
from projects.mmdet3d_plugin.models.utils.frustum_occ_filter import FrustumOccFilter
from projects.mmdet3d_plugin.ops import scatter_v2


@DETECTORS.register_module()
class FSF_Occ(FSF):
    """带稀疏占据引导非模态补全的 FSF 检测器。

    相较于 FSF，在 __init__ 中额外构建 FrustumOccFilter，
    并重写 frustum_forward，其余完全继承 FSF 的行为。

    额外 Args（通过 frustum_occ_filter_cfg dict 传入）：
        occ_mlp_cfg    (dict): OccMLP 构建参数
            - in_channels (int): 点特征维度，应与 seg_feat 维度一致
            - hidden_dims (list): 隐层维度
            - norm_cfg / act : 标准配置
        amodal_head_cfg (dict): AmodalCenterHead 构建参数
            - in_channels (int): 聚合后实例特征维度（等于 in_channels of OccMLP）
            - hidden_dims (list)
        occ_thr         (float): 占据过滤阈值 τ，默认 0.3
        loss_occ_weight (float): L_occ 权重 λ1，默认 1.0
        loss_center_weight (float): L_center 权重 λ2，默认 0.5

    示例 cfg（追加到 model dict 中）：
        frustum_occ_filter_cfg=dict(
            occ_mlp_cfg=dict(
                in_channels=131,  # 67(seg_logits)+64(seg_feats)
                hidden_dims=[64, 32],
                norm_cfg=dict(type='LN', eps=1e-3),
                act='gelu',
            ),
            amodal_head_cfg=dict(
                in_channels=131,
                hidden_dims=[64, 32],
                norm_cfg=dict(type='LN', eps=1e-3),
                act='gelu',
            ),
            occ_thr=0.3,
            loss_occ_weight=1.0,
            loss_center_weight=0.5,
        )
    """

    def __init__(self, frustum_occ_filter_cfg: dict = None, **kwargs):
        super().__init__(**kwargs)

        # ---- 构建 FrustumOccFilter ---- #
        if frustum_occ_filter_cfg is None:
            frustum_occ_filter_cfg = {}

        # 若未配置，使用默认参数（in_channels=131 对应 seg_feat 维度）
        default_in_ch = 67 + 64  # VoteSegHead: 67 logits + 64 feats = 131
        occ_mlp_cfg = frustum_occ_filter_cfg.get('occ_mlp_cfg', dict(
            in_channels=default_in_ch,
            hidden_dims=[64, 32],
            norm_cfg=dict(type='LN', eps=1e-3),
            act='gelu',
        ))
        refine_mlp_cfg = frustum_occ_filter_cfg.get('refine_mlp_cfg', dict(
            in_channels=default_in_ch + 1,
            hidden_dims=[64, 32],
            norm_cfg=dict(type='LN', eps=1e-3),
            act='gelu',
        ))
        completion_head_cfg = frustum_occ_filter_cfg.get('completion_head_cfg', dict(
            in_channels=default_in_ch,
            hidden_dims=[64, 32],
            norm_cfg=dict(type='LN', eps=1e-3),
            act='gelu',
        ))

        self.frustum_occ_filter = FrustumOccFilter(
            occ_mlp_cfg=occ_mlp_cfg,
            refine_mlp_cfg=refine_mlp_cfg,
            completion_head_cfg=completion_head_cfg,
            occ_thr=frustum_occ_filter_cfg.get('occ_thr', 0.3),
            loss_occ_weight=frustum_occ_filter_cfg.get('loss_occ_weight', 1.0),
            loss_center_weight=frustum_occ_filter_cfg.get('loss_center_weight', 0.5),
            loss_size_weight=frustum_occ_filter_cfg.get('loss_size_weight', 0.25),
            loss_visibility_weight=frustum_occ_filter_cfg.get('loss_visibility_weight', 0.25),
            min_points_per_instance=frustum_occ_filter_cfg.get('min_points_per_instance', 1),
            class_min_points_per_instance=frustum_occ_filter_cfg.get(
                'class_min_points_per_instance', None
            ),
        )
        self.completion_descriptor_dim = frustum_occ_filter_cfg.get('completion_descriptor_dim', 10)

    # ---------------------------------------------------------------------- #
    #  辅助：获取 SIR 坐标并计算加权中心（供 OccFilter 调用前用）
    # ---------------------------------------------------------------------- #

    def _get_sir_coors_and_obs_centers(
        self,
        pts_feat,
        bz_coor,
        points,
        obj_id_tensor,
        point_fg_weights,
    ):
        """提取前景点并计算 SIR 分组坐标 + 观测加权质心。

        仅做 extract_fg_pts → double_overlap_pts → get_sir_coors
        → get_cluster_delta_weighted，与父类 frustum_pooling 前半段等价，
        但不进入 frustum_sir（SIR 做特征聚合），以便 OccFilter 介入。

        Returns:
            pts_feat_fg  (N_fg, C)
            bz_coor_fg   (N_fg, 1)
            points_fg    (N_fg, 3+)
            sir_coors    (N_fg, 3)
            obs_centers  (K, 3)   -- 加权质心
            cluster_coors (K, 3)  -- 每个实例的坐标索引
            batch_idx_fg (N_fg,)
            point_fg_weights_fg (N_fg,)
        """
        # 仅保留属于至少一个实例的点（与父类 frustum_pooling 逻辑一致）
        pts_feat_fg, bz_coor_fg, points_fg, obj_id_tensor_fg, point_fg_weights_fg = \
            self.extract_fg_pts(
                pts_feat, bz_coor, points, obj_id_tensor, point_fg_weights
            )

        if obj_id_tensor_fg.sum() == 0:
            # 边界情况：无任何前景点
            fake_num = 1
            sir_coors = bz_coor_fg.new_zeros((fake_num, 3))
            obs_centers = points_fg.new_zeros((fake_num, 3))
            cluster_coors = bz_coor_fg.new_zeros((fake_num, 3))
            return (
                pts_feat_fg, bz_coor_fg, points_fg,
                sir_coors, obs_centers, cluster_coors,
                bz_coor_fg.squeeze(-1), point_fg_weights_fg
            )

        # 对多视角重叠点进行复制（与父类一致）
        pts_feat_fg, bz_coor_fg, points_fg, obj_id_tensor_fg, point_fg_weights_fg = \
            self.double_overlap_pts(
                pts_feat_fg, bz_coor_fg, points_fg,
                obj_id_tensor_fg, point_fg_weights_fg
            )

        # 构造 SIR 分组坐标
        sir_coors, _ = self.get_sir_coors(bz_coor_fg, obj_id_tensor_fg, point_fg_weights_fg)

        # 计算加权观测质心
        _, obs_centers, cluster_coors = self.get_cluster_delta_weighted(
            points_fg, sir_coors, point_fg_weights_fg.unsqueeze(-1)
        )

        return (
            pts_feat_fg, bz_coor_fg, points_fg,
            sir_coors, obs_centers, cluster_coors,
            bz_coor_fg.squeeze(-1), point_fg_weights_fg
        )

    def _lookup_point_classes_from_sir(self, mask_anno, sir_coors):
        """Map per-point SIR object ids back to nuScenes class ids."""
        class_ids = sir_coors.new_full((sir_coors.shape[0],), -1)
        if mask_anno is None or sir_coors.numel() == 0:
            return class_ids

        batch_tensor = sir_coors[:, 0].long()
        obj_ids = sir_coors[:, 2].long() - 1
        for bidx in range(mask_anno.shape[0]):
            valid_mask = (
                (batch_tensor == bidx)
                & (obj_ids >= 0)
                & (obj_ids < mask_anno.shape[1])
            )
            if valid_mask.any():
                class_ids[valid_mask] = mask_anno[bidx, obj_ids[valid_mask], 5].long()
        return class_ids

    # ---------------------------------------------------------------------- #
    #  重写 frustum_forward（核心集成点）
    # ---------------------------------------------------------------------- #

    def frustum_forward(
        self,
        seg_out_dict,
        mask_anno,
        mask_data,
        point_infos,
        img_metas,
        cluster_center=None,
        gt_bboxes_3d=None,      # 训练时传入，测试时为 None
    ):
        """带占据过滤与非模态中心修正的视锥分支前向传播。

        相较于 FSF.frustum_forward，差异点：
        1. 在 SIR 池化 **之前** 计算观测质心 C_obs；
        2. 将点特征和 SIR 坐标送入 FrustumOccFilter，得到：
            - valid_mask  : 过滤背景后的有效点掩码
            - C_pred      : 修正后的非模态中心
            - occ_losses  : 占据损失 + 中心损失（训练时）
        3. 用 C_pred 作为 cluster_center 传入 frustum_pooling，
           替代原始的加权质心，以获得更准确的点-中心偏移。

        Args:
            gt_bboxes_3d (list[LiDARInstance3DBoxes]):
                训练时每个 batch 的真实框，用于生成 OccFilter 的监督信号。

        Returns:
            与 FSF.frustum_forward 完全相同的五元组，并额外返回 occ_losses dict。
        """
        pts_feat = seg_out_dict['seg_feats']
        batch_idx = seg_out_dict['batch_idx']
        points = seg_out_dict['seg_points']
        seg_logits = seg_out_dict['seg_logits']

        point_fg_weights = self.get_point_fg_weights(seg_logits)

        batch_size = mask_anno.shape[0]
        points_info_flat = self.combine_by_batch(point_infos, batch_idx, batch_size)

        # ---- 视锥分组（与父类相同）---- #
        obj_id_tensor = self.frustum_gather(
            batch_idx, points_info_flat, mask_data, mask_anno, img_metas
        )

        # ---- 提取前景点 + 计算观测质心（不进入 SIR）---- #
        (pts_feat_fg, bz_coor_fg, points_fg,
         sir_coors, obs_centers, cluster_coors,
         batch_idx_fg, point_fg_weights_fg) = \
            self._get_sir_coors_and_obs_centers(
                pts_feat, batch_idx.unsqueeze(-1), points, obj_id_tensor, point_fg_weights
            )
        point_class_ids_fg = self._lookup_point_classes_from_sir(mask_anno, sir_coors)

        # ---- FrustumOccFilter：占据过滤 + 非模态中心修正 ---- #
        valid_mask, C_pred, occ_losses, completion_outputs, _ = self.frustum_occ_filter(
            pts_feat=pts_feat_fg,
            points=points_fg,
            sir_coors=sir_coors,
            obs_centers=obs_centers,
            batch_idx=batch_idx_fg,
            gt_bboxes_3d_list=gt_bboxes_3d,    # None 时测试模式，自动跳过损失
            point_class_ids=point_class_ids_fg,
        )

        # ---- 将有效点掩码应用到特征（软过滤：背景点置零）---- #
        pts_feat_filtered = pts_feat_fg.clone()
        pts_feat_filtered[~valid_mask] = 0.0
        # NOTE: 选择软过滤而非真正剔除行，是为了保持 sir_coors 的分组结构不变，
        # 等价于让被过滤的背景点对 SIR MaxPool 贡献为 0。

        # ---- 用 C_pred 直接进行 SIR 聚合（绕过 frustum_pooling 中的重复预处理）---- #
        # 原因：_get_sir_coors_and_obs_centers 已执行了 extract_fg_pts + double_overlap_pts，
        # 不应再通过 frustum_pooling 重复这些步骤（会造成维度不匹配）。
        # 这里直接计算 points_delta，再送入 frustum_sir。

        # 用 C_pred 计算点-中心偏移（相比 C_obs 更准确）
        if C_pred.shape[0] == obs_centers.shape[0] and obs_centers.shape[0] > 0:
            points_delta = self.get_cluster_delta_from_center(
                points_fg, sir_coors, C_pred
            )
            effective_cluster_center = C_pred
        else:
            # 边界：K 不匹配，退回加权质心偏移
            points_delta, effective_cluster_center, _ = self.get_cluster_delta_weighted(
                points_fg, sir_coors, point_fg_weights_fg.unsqueeze(-1)
            )

        # SIR 聚合（frustum_sir：Set-in-Region feature extraction）
        out_feats, final_cluster_feats, out_coors = \
            self.frustum_sir(points_fg, pts_feat_filtered, sir_coors, f_cluster=points_delta)

        if out_coors.shape[0] == 0:
            out_coors = out_coors.new_zeros((0, 3))

        lidar_feat = final_cluster_feats
        obj_coors = out_coors
        obj_centers = effective_cluster_center

        # ---- 2D 特征编码（与父类相同）---- #
        preds_2d = self.get_single_cls_preds_2d(mask_anno, obj_coors)
        img_feat = self.encode_2d_feats(
            preds_2d,
            img_w=mask_data.shape[-1],
            img_h=mask_data.shape[-2],
            encode_mlp=self.encode_2d_mlp,
        )
        completion_descriptor = completion_outputs.get('descriptor', None)
        if completion_descriptor is None or completion_descriptor.shape[0] != lidar_feat.shape[0]:
            completion_descriptor = lidar_feat.new_zeros((lidar_feat.shape[0], self.completion_descriptor_dim))
        lidar_img_feat = torch.cat([lidar_feat, img_feat, completion_descriptor], dim=-1)
        obj_feat = lidar_img_feat
        frustum_obj_result = self.frustum_obj_head(obj_feat)

        return obj_feat, obj_centers, obj_coors, frustum_obj_result, preds_2d, occ_losses

    # ---------------------------------------------------------------------- #
    #  重写 forward_train（注入 occ_losses）
    # ---------------------------------------------------------------------- #

    def forward_train(
        self,
        points,
        img_metas,
        no_aug_gt_bboxes_3d,
        no_aug_gt_labels_3d,
        gt_bboxes_3d,
        gt_labels_3d,
        mask_data,
        mask_anno,
        gt_bboxes_ignore=None,
        img=None,
    ):
        """训练入口，在父类 forward_train 基础上注入两个额外损失项。

        主要变化：视锥分支返回 6 元组（多了 occ_losses），
        并将 occ_losses 合并进总损失 dict。
        """
        if self.voxel_downsampling_size is not None:
            points = self.segmentor.voxel_downsample(points)

        points, point_infos = self.split_points_last_3dim(points)

        no_aug_gt_bboxes_3d = [b[l >= 0] for b, l in zip(no_aug_gt_bboxes_3d, no_aug_gt_labels_3d)]
        no_aug_gt_labels_3d = [l[l >= 0] for l in no_aug_gt_labels_3d]
        gt_bboxes_3d_filt   = [b[l >= 0] for b, l in zip(gt_bboxes_3d, gt_labels_3d)]
        gt_labels_3d_filt   = [l[l >= 0] for l in gt_labels_3d]

        losses = {}

        # --- 阶段1：分割 + 图像增强分割（与 FSF 完全相同）---
        seg_out_tuple = self.segmentor(
            points=points, img_metas=img_metas,
            gt_bboxes_3d=gt_bboxes_3d_filt, gt_labels_3d=gt_labels_3d_filt,
            as_subsegmentor=True, extract_feat_only=True
        )
        seg_out_dict = self.segmentor_feat_inhance_train(
            seg_out_tuple, point_infos, mask_anno, mask_data, img_metas
        )
        losses.update(seg_out_dict['losses'])

        pts_feat = seg_out_dict['seg_feats']
        batch_idx = seg_out_dict['batch_idx']
        points = seg_out_dict['seg_points']

        # --- 阶段2：视锥查询（带 OccFilter）---
        (frustum_obj_feats, frustum_obj_centers, frustum_obj_coors,
         frustum_obj_result, frustum_preds_2d, occ_losses) = \
            self.frustum_forward(
                seg_out_dict, mask_anno, mask_data, point_infos, img_metas,
                cluster_center=None,
                gt_bboxes_3d=gt_bboxes_3d_filt,   # 传入 GT 以生成监督信号
            )

        # 注入占据相关损失（加前缀避免与其他损失 key 冲突）
        for k, v in occ_losses.items():
            losses[f'frustum_occ_{k}'] = v

        # 视锥检测头损失（与 FSF 相同）
        frustum_query_losses = self.frustum_obj_head.loss(
            frustum_obj_result['cls_logits'],
            frustum_obj_result['reg_preds'],
            frustum_obj_centers,
            frustum_obj_coors,
            no_aug_gt_bboxes_3d,
            no_aug_gt_labels_3d,
            gt_bboxes_3d_filt,
            gt_labels_3d_filt,
            frustum_preds_2d,
            img_metas,
        )
        for k, v in frustum_query_losses.items():
            losses[f'frustum_{k}'] = v

        # --- 阶段3：LiDAR 查询（与 FSF 完全相同）---
        fsd_obj_feats, fsd_obj_centers, fsd_obj_coors, fsd_obj_result = \
            self.fsd_forward(seg_out_dict, img_metas)
        fsd_loss_inputs = (
            fsd_obj_result['cls_logits'], fsd_obj_result['reg_preds'],
            fsd_obj_centers, fsd_obj_coors,
            gt_bboxes_3d_filt, gt_labels_3d_filt, img_metas
        )
        fsd_query_loss = self.bbox_head.loss(
            *fsd_loss_inputs,
            iou_logits=fsd_obj_result.get('iou_logits', None),
            gt_bboxes_ignore=gt_bboxes_ignore
        )
        for k, v in fsd_query_loss.items():
            losses[f'fsd_{k}'] = v

        # --- 阶段4：合并 + 多阶段精炼（与 FSF 完全相同）---
        obj_centers, obj_coors, obj_result, obj_feats, preds_2d = \
            self.combine_frustum_and_fsd(
                frustum_obj_centers, frustum_obj_coors, frustum_obj_result,
                frustum_obj_feats, frustum_preds_2d,
                fsd_obj_centers, fsd_obj_coors, fsd_obj_result, fsd_obj_feats,
            )

        if self.num_extra_stages > 0:
            multi_stage_losses = self.multi_stage_refine_train(
                obj_centers, obj_coors, obj_result, points, point_infos,
                pts_feat, batch_idx, mask_data, mask_anno,
                no_aug_gt_bboxes_3d, no_aug_gt_labels_3d,
                gt_bboxes_3d_filt, gt_labels_3d_filt,
                preds_2d, img_metas, obj_feats,
            )
            losses.update(multi_stage_losses)

        return losses

    # ---------------------------------------------------------------------- #
    #  重写 simple_test（测试时 frustum_forward 返回 6 元组，需兼容）
    # ---------------------------------------------------------------------- #

    def simple_test(self, points, img_metas, mask_data, mask_anno, imgs=None, rescale=False):
        """测试入口，适配 frustum_forward 新的 6 元组返回值。"""
        if self.voxel_downsampling_size is not None:
            points = self.segmentor.voxel_downsample(points)

        points, point_infos = self.split_points_last_3dim(points)

        # 使用 segmentor.simple_test（而非直接调用 segmentor()）
        # 直接调用 segmentor() 会因缺少 return_loss 参数而路由到 forward_train
        seg_out_tuple = self.segmentor.simple_test(points, img_metas, extract_feat_only=True, rescale=False)
        seg_out_dict = self.segmentor_feat_inhance_test(seg_out_tuple, point_infos, mask_anno, mask_data, img_metas)

        pts_feat = seg_out_dict['seg_feats']
        batch_idx = seg_out_dict['batch_idx']
        points_updated = seg_out_dict['seg_points']

        # 视锥分支
        (frustum_obj_feats, frustum_obj_centers, frustum_obj_coors,
         frustum_obj_result, frustum_preds_2d, _) = \
            self.frustum_forward(
                seg_out_dict, mask_anno, mask_data, point_infos, img_metas,
                cluster_center=None, gt_bboxes_3d=None,  # 测试模式不传 GT
            )

        # LiDAR 分支
        fsd_obj_feats, fsd_obj_centers, fsd_obj_coors, fsd_obj_result = \
            self.fsd_forward(seg_out_dict, img_metas)

        # 合并
        obj_centers, obj_coors, obj_result, obj_feats, preds_2d = \
            self.combine_frustum_and_fsd(
                frustum_obj_centers, frustum_obj_coors, frustum_obj_result,
                frustum_obj_feats, frustum_preds_2d,
                fsd_obj_centers, fsd_obj_coors, fsd_obj_result, fsd_obj_feats,
            )

        if self.num_extra_stages >= 0:
            bbox_list = self.multi_stage_refine_test(
                obj_centers, obj_coors, obj_result, points_updated, point_infos,
                pts_feat, batch_idx, mask_data, mask_anno, preds_2d, img_metas, obj_feats,
            )

        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results
