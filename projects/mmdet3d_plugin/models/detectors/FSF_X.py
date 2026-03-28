"""
FSF_X: Fully Sparse Fusion – Combined Architecture
====================================================
三个创新点的统一实现：

1. H-Mamba 跨模态序列化特征交互
   继承自 FSF_HMamba，重写 combine_frustum_and_fsd，将拼接后的多模态
   实例序列送入 HMambaInteraction（选择性状态空间模型），以 O(m·N) 复杂度
   完成跨实例感知，输出融合特征。

2. 基于稀疏占据的视锥非模态补全（FrustumOccFilter）
   继承自 FSF_Occ 的 frustum_forward 逻辑：
   - OccMLP：逐点预测占据概率，过滤背景虚空点
   - AmodalCenterHead：预测非模态中心修正量 Δc，
     纠正因遮挡导致的观测质心偏移
   额外训练信号：L_occ (FocalLoss) + L_amodal_center (SmoothL1)

3. 可微最优传输标签分配（OTA / Sinkhorn-Knopp）
   frustum_obj_head 和 refined_obj_head 均使用 FSFOTAHead，
   将标签分配重公式化为熵正则化最优传输问题，
   通过 Sinkhorn-Knopp 迭代在对数域求解，输出软分配矩阵 π*。

继承链：FSF_X → FSF_HMamba → FSF
重写方法：__init__  /  frustum_forward  /  forward_train  /  simple_test
"""

import torch
from mmdet.models import DETECTORS
from mmdet3d.core.bbox import bbox3d2result

from .FSF_HMamba import FSF_HMamba
from projects.mmdet3d_plugin.models.utils.frustum_occ_filter import FrustumOccFilter


@DETECTORS.register_module()
class FSF_X(FSF_HMamba):
    """三创新点组合检测器。

    额外 Args（在 FSF_HMamba 之上）：
        frustum_occ_filter_cfg (dict): FrustumOccFilter 配置，键同 FSF_Occ。
            若为 None，则跳过 OccFilter（退化为纯 FSF_HMamba）。

    注：frustum_obj_head / refined_obj_head 应在 config 中指定为
        FSFOTAHead（OTA 标签分配），其余由父类处理。
    """

    def __init__(self, frustum_occ_filter_cfg: dict = None, **kwargs):
        # FSF_HMamba.__init__ 内部会调用 FSF.__init__，完成所有基础模块的构建，
        # 包括 HMambaInteraction（通过 hmamba_cfg kwarg 传入）
        super().__init__(**kwargs)

        # ── FrustumOccFilter ──────────────────────────────────────────────
        if frustum_occ_filter_cfg is None:
            # 不使用 OccFilter，退化为 FSF_HMamba + OTA
            self.frustum_occ_filter = None
        else:
            default_in_ch = 67 + 64  # VoteSegHead logits(67) + feats(64)
            occ_mlp_cfg = frustum_occ_filter_cfg.get('occ_mlp_cfg', dict(
                in_channels=default_in_ch,
                hidden_dims=[64, 32],
                norm_cfg=dict(type='LN', eps=1e-3),
                act='gelu',
            ))
            amodal_head_cfg = frustum_occ_filter_cfg.get('amodal_head_cfg', dict(
                in_channels=default_in_ch,
                hidden_dims=[64, 32],
                norm_cfg=dict(type='LN', eps=1e-3),
                act='gelu',
            ))
            self.frustum_occ_filter = FrustumOccFilter(
                occ_mlp_cfg=occ_mlp_cfg,
                amodal_head_cfg=amodal_head_cfg,
                occ_thr=frustum_occ_filter_cfg.get('occ_thr', 0.3),
                loss_occ_weight=frustum_occ_filter_cfg.get('loss_occ_weight', 1.0),
                loss_center_weight=frustum_occ_filter_cfg.get('loss_center_weight', 0.5),
            )

    # ── 从 FSF_Occ 移植的工具方法 ──────────────────────────────────────────

    def _get_sir_coors_and_obs_centers(
        self, pts_feat, bz_coor, points, obj_id_tensor, point_fg_weights
    ):
        """提取前景点并计算 SIR 分组坐标 + 观测加权质心（不进入 SIR 聚合）。"""
        pts_feat_fg, bz_coor_fg, points_fg, obj_id_tensor_fg, point_fg_weights_fg = \
            self.extract_fg_pts(pts_feat, bz_coor, points, obj_id_tensor, point_fg_weights)

        if obj_id_tensor_fg.sum() == 0:
            fake_num = 1
            sir_coors   = bz_coor_fg.new_zeros((fake_num, 3))
            obs_centers = points_fg.new_zeros((fake_num, 3))
            cluster_coors = bz_coor_fg.new_zeros((fake_num, 3))
            return (pts_feat_fg, bz_coor_fg, points_fg,
                    sir_coors, obs_centers, cluster_coors,
                    bz_coor_fg.squeeze(-1), point_fg_weights_fg)

        pts_feat_fg, bz_coor_fg, points_fg, obj_id_tensor_fg, point_fg_weights_fg = \
            self.double_overlap_pts(
                pts_feat_fg, bz_coor_fg, points_fg,
                obj_id_tensor_fg, point_fg_weights_fg
            )

        sir_coors, _ = self.get_sir_coors(bz_coor_fg, obj_id_tensor_fg, point_fg_weights_fg)
        _, obs_centers, cluster_coors = self.get_cluster_delta_weighted(
            points_fg, sir_coors, point_fg_weights_fg.unsqueeze(-1)
        )
        return (pts_feat_fg, bz_coor_fg, points_fg,
                sir_coors, obs_centers, cluster_coors,
                bz_coor_fg.squeeze(-1), point_fg_weights_fg)

    # ── 重写 frustum_forward（注入 OccFilter）────────────────────────────

    def frustum_forward(
        self,
        seg_out_dict,
        mask_anno,
        mask_data,
        point_infos,
        img_metas,
        cluster_center=None,
        gt_bboxes_3d=None,
    ):
        """带 OccFilter 的视锥前向传播（无 OccFilter 时退化为 FSF 原逻辑）。

        Returns:
            (obj_feat, obj_centers, obj_coors, frustum_obj_result,
             preds_2d, occ_losses)
            occ_losses 为空 dict 表示无 OccFilter 或测试模式。
        """
        if self.frustum_occ_filter is None:
            # 无 OccFilter：调用父类（FSF_HMamba→FSF）的 frustum_forward，
            # 但需要包装返回值以保持 6-tuple 一致。
            result = super(FSF_HMamba, self).frustum_forward(
                seg_out_dict, mask_anno, mask_data, point_infos, img_metas,
                cluster_center=cluster_center,
            )
            # 父类返回 5-tuple，补空 occ_losses
            return (*result, {})

        # ── 以下逻辑与 FSF_Occ.frustum_forward 完全对应 ──────────────────
        pts_feat   = seg_out_dict['seg_feats']
        batch_idx  = seg_out_dict['batch_idx']
        points     = seg_out_dict['seg_points']
        seg_logits = seg_out_dict['seg_logits']

        point_fg_weights = self.get_point_fg_weights(seg_logits)

        batch_size = mask_anno.shape[0]
        points_info_flat = self.combine_by_batch(point_infos, batch_idx, batch_size)

        obj_id_tensor = self.frustum_gather(
            batch_idx, points_info_flat, mask_data, mask_anno, img_metas
        )

        (pts_feat_fg, bz_coor_fg, points_fg,
         sir_coors, obs_centers, cluster_coors,
         batch_idx_fg, point_fg_weights_fg) = \
            self._get_sir_coors_and_obs_centers(
                pts_feat, batch_idx.unsqueeze(-1), points, obj_id_tensor, point_fg_weights
            )

        # OccFilter：占据过滤 + 非模态中心修正
        valid_mask, C_pred, occ_losses = self.frustum_occ_filter(
            pts_feat=pts_feat_fg,
            points=points_fg,
            sir_coors=sir_coors,
            obs_centers=obs_centers,
            batch_idx=batch_idx_fg,
            gt_bboxes_3d_list=gt_bboxes_3d,
        )

        # 软过滤（背景点特征置零，保持 SIR 分组结构）
        pts_feat_filtered = pts_feat_fg.clone()
        pts_feat_filtered[~valid_mask] = 0.0

        # 使用修正中心计算点-中心偏移
        if C_pred.shape[0] == obs_centers.shape[0] and obs_centers.shape[0] > 0:
            points_delta = self.get_cluster_delta_from_center(
                points_fg, sir_coors, C_pred
            )
            effective_cluster_center = C_pred
        else:
            points_delta, effective_cluster_center, _ = self.get_cluster_delta_weighted(
                points_fg, sir_coors, point_fg_weights_fg.unsqueeze(-1)
            )

        # SIR 聚合
        out_feats, final_cluster_feats, out_coors = \
            self.frustum_sir(points_fg, pts_feat_filtered, sir_coors, f_cluster=points_delta)

        if out_coors.shape[0] == 0:
            out_coors = out_coors.new_zeros((0, 3))

        obj_coors   = out_coors
        obj_centers = effective_cluster_center
        lidar_feat  = final_cluster_feats

        # 2D 特征编码
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

        return obj_feat, obj_centers, obj_coors, frustum_obj_result, preds_2d, occ_losses

    # ── forward_train（注入 occ_losses）────────────────────────────────────

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
        if self.voxel_downsampling_size is not None:
            points = self.segmentor.voxel_downsample(points)

        points, point_infos = self.split_points_last_3dim(points)

        no_aug_gt_bboxes_3d = [b[l >= 0] for b, l in zip(no_aug_gt_bboxes_3d, no_aug_gt_labels_3d)]
        no_aug_gt_labels_3d = [l[l >= 0] for l in no_aug_gt_labels_3d]
        gt_bboxes_3d_filt   = [b[l >= 0] for b, l in zip(gt_bboxes_3d, gt_labels_3d)]
        gt_labels_3d_filt   = [l[l >= 0] for l in gt_labels_3d]

        losses = {}

        # ── 阶段1：分割 ─────────────────────────────────────────────────
        seg_out_tuple = self.segmentor(
            points=points, img_metas=img_metas,
            gt_bboxes_3d=gt_bboxes_3d_filt, gt_labels_3d=gt_labels_3d_filt,
            as_subsegmentor=True, extract_feat_only=True
        )
        seg_out_dict = self.segmentor_feat_inhance_train(
            seg_out_tuple, point_infos, mask_anno, mask_data, img_metas
        )
        losses.update(seg_out_dict['losses'])

        pts_feat  = seg_out_dict['seg_feats']
        batch_idx = seg_out_dict['batch_idx']
        points    = seg_out_dict['seg_points']

        # ── 阶段2：视锥查询（含 OccFilter）────────────────────────────
        (frustum_obj_feats, frustum_obj_centers, frustum_obj_coors,
         frustum_obj_result, frustum_preds_2d, occ_losses) = \
            self.frustum_forward(
                seg_out_dict, mask_anno, mask_data, point_infos, img_metas,
                cluster_center=None,
                gt_bboxes_3d=gt_bboxes_3d_filt,
            )

        # 占据辅助损失（仅当 OccFilter 启用时非空）
        for k, v in occ_losses.items():
            losses[f'frustum_occ_{k}'] = v

        # 视锥检测头损失（FSFOTAHead）
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

        # ── 阶段3：LiDAR 查询 ────────────────────────────────────────
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
            gt_bboxes_ignore=gt_bboxes_ignore,
        )
        for k, v in fsd_query_loss.items():
            losses[f'fsd_{k}'] = v

        # ── 阶段4：H-Mamba 融合 + 多阶段精炼 ────────────────────────
        # combine_frustum_and_fsd 由 FSF_HMamba 提供（含 H-Mamba 交互）
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

    # ── simple_test ─────────────────────────────────────────────────────────

    def simple_test(self, points, img_metas, mask_data, mask_anno, imgs=None, rescale=False):
        if self.voxel_downsampling_size is not None:
            points = self.segmentor.voxel_downsample(points)

        points, point_infos = self.split_points_last_3dim(points)

        seg_out_tuple = self.segmentor.simple_test(
            points, img_metas, extract_feat_only=True, rescale=False
        )
        seg_out_dict = self.segmentor_feat_inhance_test(
            seg_out_tuple, point_infos, mask_anno, mask_data, img_metas
        )

        pts_feat       = seg_out_dict['seg_feats']
        batch_idx      = seg_out_dict['batch_idx']
        points_updated = seg_out_dict['seg_points']

        # 视锥分支（测试时不传 GT，occ_losses 为空 dict）
        (frustum_obj_feats, frustum_obj_centers, frustum_obj_coors,
         frustum_obj_result, frustum_preds_2d, _) = \
            self.frustum_forward(
                seg_out_dict, mask_anno, mask_data, point_infos, img_metas,
                cluster_center=None, gt_bboxes_3d=None,
            )

        # LiDAR 分支
        fsd_obj_feats, fsd_obj_centers, fsd_obj_coors, fsd_obj_result = \
            self.fsd_forward(seg_out_dict, img_metas)

        # H-Mamba 融合
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
