"""
FrustumOccFilter: 稀疏占据引导的视锥点云去噪模块
=====================================================
灵感来源：SparseOcc（arxiv 2312.17118）

核心思想：
    原 FSF 的视锥生成仅将 2D mask 拉伸为 3D 视锥，包含大量背景噪声点。
    本模块借鉴 SparseOcc 的稀疏占据查询思想，对视锥内每个点预测占据概率，
    并过滤掉背景虚空点，输出纯净的前景有效点集合。

    同时，由于过滤后的有效点可能仍存在遮挡，导致实例质心偏离真实 GT 中心，
    本模块额外引入非模态中心回归（Amodal Center Regression），
    通过 SmoothL1 损失监督，使预测中心更准确地落在 GT 框内。

两个子网络：
    - OccMLP         : 逐点占据概率预测，(N, C) → (N, 1)
    - AmodalCenterHead: 实例级非模态中心偏移回归，(K, C') → (K, 3)
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from mmcv.runner import BaseModule, force_fp32
from projects.mmdet3d_plugin.ops import scatter_v2, build_mlp


def ensure_minimum_points_per_instance(
    instance_ids: torch.Tensor,
    scores: torch.Tensor,
    mask: torch.Tensor,
    min_points: int = 1,
    class_ids: torch.Tensor = None,
    class_min_points: dict = None,
) -> torch.Tensor:
    """Guarantee each instance keeps enough top-scoring points after filtering."""
    if instance_ids.numel() == 0:
        return mask

    min_points = max(int(min_points), 1)
    class_min_points = class_min_points or {}
    kept = mask.clone()
    for instance_id in torch.unique(instance_ids):
        instance_mask = instance_ids == instance_id
        target_min_points = min_points
        if class_ids is not None and class_min_points:
            valid_classes = class_ids[instance_mask]
            valid_classes = valid_classes[valid_classes >= 0]
            if valid_classes.numel() > 0:
                cls_id = int(valid_classes[0].item())
                target_min_points = max(
                    target_min_points,
                    int(class_min_points.get(cls_id, target_min_points)),
                )
        kept_count = int(kept[instance_mask].sum().item())
        if kept_count >= target_min_points:
            continue
        global_indices = instance_mask.nonzero(as_tuple=False).squeeze(-1)
        candidate_indices = global_indices[~kept[global_indices]]
        if candidate_indices.numel() == 0:
            continue
        add_count = min(target_min_points - kept_count, int(candidate_indices.numel()))
        candidate_scores = scores[candidate_indices]
        _, topk_order = torch.topk(candidate_scores, k=add_count)
        kept[candidate_indices[topk_order]] = True
    return kept


def compute_bev_orientation_cues(
    points: torch.Tensor,
    instance_ids: torch.Tensor,
    valid_mask: torch.Tensor = None,
) -> torch.Tensor:
    """Estimate per-instance BEV principal direction as cos/sin cues."""
    if instance_ids.numel() == 0:
        return points.new_zeros((0, 2))

    if valid_mask is None:
        valid_mask = torch.ones_like(instance_ids, dtype=torch.bool)

    cues = []
    for instance_id in torch.unique(instance_ids):
        instance_mask = (instance_ids == instance_id) & valid_mask
        xy = points[instance_mask][:, :2]
        if xy.shape[0] < 2:
            cues.append(points.new_zeros(2))
            continue

        centered = xy - xy.mean(dim=0, keepdim=True)
        var_x = (centered[:, 0] * centered[:, 0]).mean()
        var_y = (centered[:, 1] * centered[:, 1]).mean()
        cov_xy = (centered[:, 0] * centered[:, 1]).mean()
        spread = var_x + var_y
        if spread <= 1e-6:
            cues.append(points.new_zeros(2))
            continue

        theta = 0.5 * torch.atan2(2.0 * cov_xy, var_x - var_y)
        cues.append(torch.stack([torch.cos(theta), torch.sin(theta)]))
    return torch.stack(cues, dim=0)


def build_completion_descriptor(
    coarse_scores: torch.Tensor,
    refine_scores: torch.Tensor,
    instance_ids: torch.Tensor,
    pred_size_residuals: torch.Tensor,
    pred_visibility: torch.Tensor,
    center_offsets: torch.Tensor,
    orientation_cues: torch.Tensor = None,
) -> torch.Tensor:
    """Build compact per-instance descriptors for detector-side reuse."""
    if instance_ids.numel() == 0:
        return pred_size_residuals.new_zeros((0, 10))

    descriptors = []
    unique_instance_ids = torch.unique(instance_ids)
    if orientation_cues is None or orientation_cues.shape[0] != unique_instance_ids.shape[0]:
        orientation_cues = pred_size_residuals.new_zeros((unique_instance_ids.shape[0], 2))
    for row_idx, instance_id in enumerate(unique_instance_ids):
        mask = instance_ids == instance_id
        descriptors.append(torch.cat([
            coarse_scores[mask].mean().reshape(1),
            coarse_scores[mask].max().reshape(1),
            refine_scores[mask].mean().reshape(1),
            pred_size_residuals[row_idx],
            pred_visibility[row_idx].reshape(1),
            center_offsets[row_idx].norm().reshape(1),
            orientation_cues[row_idx],
        ]))
    return torch.stack(descriptors, dim=0)


class OccMLP(BaseModule):
    """逐点占据概率预测网络（对应 SparseOcc 的 Sparsification 思想）。

    对视锥内的每一个点（体素），预测其被前景对象占据的概率，
    从而识别并剔除背景虚空点。

    Args:
        in_channels (int): 输入点特征维度
        hidden_dims (list[int]): 隐层维度列表
        norm_cfg (dict): 归一化层配置
        act (str): 激活函数类型，支持 'relu'/'gelu'
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dims: list = [64, 32],
        norm_cfg: dict = dict(type='LN', eps=1e-3),
        act: str = 'gelu',
    ):
        super().__init__()
        self.mlp = build_mlp(
            in_channels,
            hidden_dims + [1],
            norm_cfg,
            is_head=True,
            act=act,
        )

    def forward(self, pts_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pts_feat (Tensor): (N, C) 点特征

        Returns:
            occ_prob (Tensor): (N,) 占据概率 ∈ [0, 1]
        """
        logit = self.mlp(pts_feat)          # (N, 1)
        occ_prob = logit.squeeze(-1).sigmoid()  # (N,)
        return occ_prob


class InstanceCompletionHead(BaseModule):
    """实例级几何补全头。

    对每个视锥实例（每个 obj_id 对应一个实例），聚合有效占据点的特征，
    联合回归：
        - 非模态中心偏移
        - 尺度残差
        - 可见率

    Args:
        in_channels (int): 输入特征维度（聚合后的实例特征维度）
        hidden_dims (list[int]): 隐层维度
        norm_cfg (dict): 归一化配置
        act (str): 激活函数
        xyz_normalizer (list): xyz 坐标归一化系数（用于 SIR 聚合时的特征拼接）
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dims: list = [64, 32],
        norm_cfg: dict = dict(type='LN', eps=1e-3),
        act: str = 'gelu',
    ):
        super().__init__()

        self.center_mlp = build_mlp(
            in_channels,
            hidden_dims + [3],
            norm_cfg,
            is_head=True,
            act=act,
        )
        self.size_mlp = build_mlp(
            in_channels,
            hidden_dims + [3],
            norm_cfg,
            is_head=True,
            act=act,
        )
        self.visibility_mlp = build_mlp(
            in_channels,
            hidden_dims + [1],
            norm_cfg,
            is_head=True,
            act=act,
        )

    def forward(
        self,
        pts_feat: torch.Tensor,
        sir_coors: torch.Tensor,
        valid_mask: torch.Tensor,
    ):
        """对每个实例内有效点做 MaxPool 聚合后回归补全量。

        Args:
            pts_feat (Tensor): (N, C) 全部点特征（未过滤）
            sir_coors (Tensor): (N, 3) SIR 分组坐标 (batch_id, ?, obj_id)
            valid_mask (Tensor): (N,) bool，OccMLP 输出的有效点掩码

        Returns:
            cluster_offsets (Tensor): (K, 3) 每个实例的中心偏移量
            size_residuals (Tensor): (K, 3) 每个实例的尺度残差
            visibility     (Tensor): (K,)   每个实例的可见率
            cluster_coors  (Tensor): (K, 3) 每个实例的 SIR 坐标
        """
        # 只使用有效（前景）点做聚合
        if valid_mask.sum() == 0:
            # 边界情况：全部被过滤掉，回零偏移
            unique_coors = sir_coors.new_zeros((1, 3))
            offsets = pts_feat.new_zeros((1, 3))
            size_residuals = pts_feat.new_zeros((1, 3))
            visibility = pts_feat.new_zeros(1)
            return offsets, size_residuals, visibility, unique_coors

        valid_feat = pts_feat[valid_mask]       # (N_valid, C)
        valid_coors = sir_coors[valid_mask]     # (N_valid, 3)

        # 按实例（sir_coors 唯一键）做 Max Pooling
        agg_feat, agg_coors, _ = scatter_v2(
            valid_feat, valid_coors, mode='max'
        )
        # agg_feat: (K, C)  agg_coors: (K, 3)

        cluster_offsets = self.center_mlp(agg_feat)    # (K, 3)
        size_residuals = self.size_mlp(agg_feat)       # (K, 3)
        visibility = self.visibility_mlp(agg_feat).squeeze(-1).sigmoid()  # (K,)
        return cluster_offsets, size_residuals, visibility, agg_coors


class FrustumOccFilter(BaseModule):
    """视锥占据感知过滤与非模态中心修正模块（FSF_Occ 核心创新）。

    集成点：替换 FSF 原 `frustum_forward` 中的纯视锥池化流程：

        原流程：
            视锥点集 P_j → SIR 加权质心 → frustum_pooling → 实例特征

        FSF_Occ 流程：
            视锥点集 P_j
                ↓ OccMLP         → 过滤背景，得有效点集 P_valid
                ↓ AmodalCenterHead → 修正实例中心 C_pred = C_obs + Δc
                ↓ frustum_pooling(cluster_center=C_pred)
                → 更纯净的实例特征

    训练时额外计算：
        L_occ          : Focal Loss 监督占据预测
        L_amodal_center: Smooth-L1 监督中心偏移回归

    Args:
        occ_mlp_cfg (dict):    OccMLP 的构建参数
        amodal_head_cfg (dict): AmodalCenterHead 的构建参数
        occ_thr (float):       占据概率过滤阈值 τ，默认 0.3
        loss_occ_weight (float): L_occ 在总损失中的系数 λ1，默认 1.0
        loss_center_weight (float): L_center 的系数 λ2，默认 0.5
    """

    def __init__(
        self,
        occ_mlp_cfg: dict,
        refine_mlp_cfg: dict = None,
        completion_head_cfg: dict = None,
        amodal_head_cfg: dict = None,
        occ_thr: float = 0.3,
        loss_occ_weight: float = 1.0,
        loss_center_weight: float = 0.5,
        loss_size_weight: float = 0.25,
        loss_visibility_weight: float = 0.25,
        min_points_per_instance: int = 1,
        class_min_points_per_instance: dict = None,
    ):
        super().__init__()
        self.occ_thr = occ_thr
        self.loss_occ_weight = loss_occ_weight
        self.loss_center_weight = loss_center_weight
        self.loss_size_weight = loss_size_weight
        self.loss_visibility_weight = loss_visibility_weight
        self.min_points_per_instance = max(int(min_points_per_instance), 1)
        self.class_min_points_per_instance = {
            int(cls_id): max(int(min_points), 1)
            for cls_id, min_points in (class_min_points_per_instance or {}).items()
        }

        self.occ_mlp = OccMLP(**occ_mlp_cfg)

        if refine_mlp_cfg is None:
            refine_mlp_cfg = dict(
                in_channels=occ_mlp_cfg['in_channels'] + 1,
                hidden_dims=occ_mlp_cfg.get('hidden_dims', [64, 32]),
                norm_cfg=occ_mlp_cfg.get('norm_cfg', dict(type='LN', eps=1e-3)),
                act=occ_mlp_cfg.get('act', 'gelu'),
            )
        if completion_head_cfg is None:
            completion_head_cfg = amodal_head_cfg if amodal_head_cfg is not None else dict(
                in_channels=occ_mlp_cfg['in_channels'],
                hidden_dims=[64, 32],
                norm_cfg=dict(type='LN', eps=1e-3),
                act='gelu',
            )

        self.refine_mlp = OccMLP(**refine_mlp_cfg)
        self.completion_head = InstanceCompletionHead(**completion_head_cfg)
        # Backward-compatible alias for existing comments/call sites.
        self.amodal_center_head = self.completion_head

    # ---------------------------------------------------------------------- #
    #  GT 生成辅助函数
    # ---------------------------------------------------------------------- #

    @torch.no_grad()
    def get_occ_gt(
        self,
        points: torch.Tensor,
        batch_idx: torch.Tensor,
        gt_bboxes_3d_list: list,
    ) -> torch.Tensor:
        """根据点是否落在 GT 3D 框内，生成逐点占据二值标签。

        Args:
            points    (Tensor): (N, 3+) 点坐标
            batch_idx (Tensor): (N,)    各点所属 batch 索引
            gt_bboxes_3d_list (list):   每个 batch 的 LiDARInstance3DBoxes 列表

        Returns:
            occ_gt (Tensor): (N,) float，1 表示前景占据，0 表示背景
        """
        occ_gt = points.new_zeros(points.shape[0])
        for bidx, gt_bboxes in enumerate(gt_bboxes_3d_list):
            bz_mask = (batch_idx == bidx)
            if bz_mask.sum() == 0 or len(gt_bboxes) == 0:
                continue
            pts_bz = points[bz_mask][:, :3]    # (N_bz, 3)
            # points_in_boxes_gpu 要求 boxes 为 7 维（不含速度），截取前7列
            from mmdet3d.ops.roiaware_pool3d import points_in_boxes_gpu
            box_idx = points_in_boxes_gpu(
                pts_bz.unsqueeze(0),
                gt_bboxes.tensor[:, :7].unsqueeze(0).to(pts_bz.device)
            ).squeeze(0)   # (N_bz,) long，-1 表示不在任何框内
            fg_mask = box_idx >= 0          # (N_bz,) bool
            occ_gt[bz_mask] = fg_mask.float()
        return occ_gt

    @torch.no_grad()
    def get_amodal_center_gt(
        self,
        obs_centers: torch.Tensor,
        cluster_coors: torch.Tensor,
        batch_idx_centers: torch.Tensor,
        gt_bboxes_3d_list: list,
        gt_labels_3d_list: list = None,
    ):
        """为每个实例构造非模态中心偏移 GT 和有效实例掩码。

        Args:
            obs_centers (Tensor): (K, 3) 从有效点计算的观测质心
            cluster_coors (Tensor): (K, 3) 每个实例的 SIR 坐标 (batch, ?, obj_id)
            batch_idx_centers (Tensor): (K,) 每个实例的 batch 索引
            gt_bboxes_3d_list (list): GT 框列表（每个 batch 一组）

        Returns:
            center_gt (Tensor): (K, 3) 目标偏移量 Δc = C_gt - C_obs
            valid_center_mask (Tensor): (K,) bool，成功匹配到 GT 的实例
        """
        K = obs_centers.shape[0]
        center_gt = obs_centers.new_zeros(K, 3)
        valid_center_mask = obs_centers.new_zeros(K, dtype=torch.bool)

        for bidx, gt_bboxes in enumerate(gt_bboxes_3d_list):
            bz_mask = (batch_idx_centers == bidx)
            if bz_mask.sum() == 0 or len(gt_bboxes) == 0:
                continue

            obs_centers_bz = obs_centers[bz_mask]   # (K_bz, 3)
            gt_centers = gt_bboxes.gravity_center.to(obs_centers.device)  # (M, 3)

            # 用最近邻方式为每个观测质心匹配 GT 中心
            dists = torch.cdist(obs_centers_bz[:, :2], gt_centers[:, :2])  # BEV 距离
            min_dists, gt_inds = dists.min(dim=-1)  # (K_bz,)

            # 仅对距离在合理范围内 (< 4m) 的实例赋 GT
            match_mask = min_dists < 4.0
            center_gt_bz = obs_centers_bz.clone()
            matched_gt_centers = gt_centers[gt_inds[match_mask]]   # (N_match, 3)
            center_gt_bz[match_mask] = matched_gt_centers

            global_inds = bz_mask.nonzero(as_tuple=False).squeeze(-1)
            center_gt[global_inds] = center_gt_bz - obs_centers_bz
            valid_center_mask[global_inds[match_mask]] = True

        return center_gt, valid_center_mask

    # ---------------------------------------------------------------------- #
    #  损失计算
    # ---------------------------------------------------------------------- #

    @torch.no_grad()
    def get_completion_gt(
        self,
        obs_centers: torch.Tensor,
        cluster_coors: torch.Tensor,
        batch_idx_centers: torch.Tensor,
        gt_bboxes_3d_list: list,
        points: torch.Tensor,
        sir_coors: torch.Tensor,
        valid_mask: torch.Tensor,
    ):
        """Construct weak completion targets for center, size and visibility."""
        K = obs_centers.shape[0]
        center_gt = obs_centers.new_zeros((K, 3))
        size_gt = obs_centers.new_zeros((K, 3))
        visibility_gt = obs_centers.new_zeros(K)
        valid_completion_mask = obs_centers.new_zeros(K, dtype=torch.bool)

        if K == 0:
            return center_gt, size_gt, visibility_gt, valid_completion_mask

        for k in range(K):
            bidx = int(batch_idx_centers[k].item())
            if bidx >= len(gt_bboxes_3d_list):
                continue
            gt_bboxes = gt_bboxes_3d_list[bidx]
            if len(gt_bboxes) == 0:
                continue

            gt_centers = gt_bboxes.gravity_center.to(obs_centers.device)
            dists = torch.cdist(obs_centers[k:k + 1, :2], gt_centers[:, :2]).squeeze(0)
            min_dist, gt_ind = dists.min(dim=0)
            if min_dist >= 4.0:
                continue

            center_gt[k] = gt_centers[gt_ind] - obs_centers[k]

            instance_mask = (sir_coors == cluster_coors[k]).all(dim=1)
            pts_inst = points[instance_mask][:, :3]
            if pts_inst.shape[0] > 0:
                obs_extent = (pts_inst.max(dim=0).values - pts_inst.min(dim=0).values).clamp(min=1e-2)
                gt_dims = gt_bboxes.tensor[gt_ind, 3:6].to(obs_centers.device).clamp(min=1e-2)
                size_gt[k] = torch.log(gt_dims / obs_extent)

                visible_points = valid_mask[instance_mask].float().sum()
                total_points = float(max(int(instance_mask.sum().item()), 1))
                visibility_gt[k] = visible_points / total_points
            valid_completion_mask[k] = True

        return center_gt, size_gt, visibility_gt, valid_completion_mask

    @force_fp32(apply_to=('coarse_occ_prob', 'refine_occ_prob', 'pred_offsets', 'pred_size_residuals', 'pred_visibility'))
    def compute_losses(
        self,
        coarse_occ_prob: torch.Tensor,
        refine_occ_prob: torch.Tensor,
        occ_gt: torch.Tensor,
        pred_offsets: torch.Tensor,
        center_gt: torch.Tensor,
        valid_center_mask: torch.Tensor,
        pred_size_residuals: torch.Tensor,
        size_gt: torch.Tensor,
        pred_visibility: torch.Tensor,
        visibility_gt: torch.Tensor,
        valid_completion_mask: torch.Tensor,
    ) -> dict:
        """计算 coarse/refine occupancy 与实例级补全损失。

        Args:
            coarse_occ_prob (N,) : coarse 占据概率预测值
            refine_occ_prob (N,) : refine 占据概率预测值
            occ_gt     (N,)   : 占据 GT 标签 (0/1)
            pred_offsets (K, 3): 预测中心偏移
            center_gt  (K, 3) : GT 中心偏移
            valid_center_mask (K,): 有效实例掩码

        Returns:
            dict: loss_occ, loss_amodal_center (均已乘以权重系数)
        """
        losses = {}

        # ---- L_occ: Binary Cross-Entropy + Focal-style 权重 ---- #
        alpha, gamma = 0.25, 2.0
        bce_coarse = F.binary_cross_entropy(coarse_occ_prob, occ_gt, reduction='none')
        pt_coarse = coarse_occ_prob * occ_gt + (1 - coarse_occ_prob) * (1 - occ_gt)
        focal_weight = alpha * occ_gt + (1 - alpha) * (1 - occ_gt)
        focal_weight_coarse = focal_weight * (1 - pt_coarse).pow(gamma)
        loss_occ_coarse = (bce_coarse * focal_weight_coarse).mean()
        losses['loss_occ_coarse'] = self.loss_occ_weight * loss_occ_coarse

        bce_refine = F.binary_cross_entropy(refine_occ_prob, occ_gt, reduction='none')
        pt_refine = refine_occ_prob * occ_gt + (1 - refine_occ_prob) * (1 - occ_gt)
        focal_weight_refine = focal_weight * (1 - pt_refine).pow(gamma)
        loss_occ_refine = (bce_refine * focal_weight_refine).mean()
        losses['loss_occ_refine'] = self.loss_occ_weight * loss_occ_refine

        # ---- L_amodal_center: Smooth-L1 ---- #
        if valid_center_mask.sum() > 0:
            pos_pred = pred_offsets[valid_center_mask]
            pos_gt = center_gt[valid_center_mask]
            loss_center = F.smooth_l1_loss(pos_pred, pos_gt, reduction='mean', beta=1.0)
        else:
            loss_center = pred_offsets.sum() * 0.0
        losses['loss_amodal_center'] = self.loss_center_weight * loss_center

        if valid_completion_mask.sum() > 0:
            loss_size = F.smooth_l1_loss(
                pred_size_residuals[valid_completion_mask],
                size_gt[valid_completion_mask],
                reduction='mean',
                beta=1.0,
            )
            loss_visibility = F.smooth_l1_loss(
                pred_visibility[valid_completion_mask],
                visibility_gt[valid_completion_mask],
                reduction='mean',
                beta=1.0,
            )
        else:
            loss_size = pred_size_residuals.sum() * 0.0
            loss_visibility = pred_visibility.sum() * 0.0

        losses['loss_amodal_size'] = self.loss_size_weight * loss_size
        losses['loss_visibility'] = self.loss_visibility_weight * loss_visibility

        return losses

    # ---------------------------------------------------------------------- #
    #  主前向函数
    # ---------------------------------------------------------------------- #

    def forward(
        self,
        pts_feat: torch.Tensor,
        points: torch.Tensor,
        sir_coors: torch.Tensor,
        obs_centers: torch.Tensor,
        batch_idx: torch.Tensor,
        gt_bboxes_3d_list: list = None,
        point_class_ids: torch.Tensor = None,
    ):
        """视锥占据过滤 + 非模态中心修正前向。

        Args:
            pts_feat    (N, C)  : 分割网络输出的点特征
            points      (N, 3+) : 原始点坐标
            sir_coors   (N, 3)  : SIR 分组坐标 (batch, ?, obj_id)
            obs_centers (K, 3)  : 基于加权质心计算的观测实例中心
            batch_idx   (N,)    : 各点的 batch 索引
            gt_bboxes_3d_list   : 训练时提供，用于生成监督信号；测试时为 None

        Returns:
            valid_mask    (N,) bool   : 被预测为有效占据的点的掩码
            C_pred        (K, 3)      : 修正后的非模态中心
            occ_losses    dict or {}  : 训练时含 loss_occ + loss_amodal_center
        """
        # ---- Step 1: coarse-to-fine 占据概率预测 ---- #
        coarse_occ_prob = self.occ_mlp(pts_feat)                  # (N,)
        refine_input = torch.cat([pts_feat, coarse_occ_prob.unsqueeze(-1)], dim=-1)
        refine_occ_prob = self.refine_mlp(refine_input)           # (N,)

        instance_ids = sir_coors[:, -1]
        valid_mask = refine_occ_prob > self.occ_thr
        valid_mask = ensure_minimum_points_per_instance(
            instance_ids=instance_ids,
            scores=refine_occ_prob,
            mask=valid_mask,
            min_points=self.min_points_per_instance,
            class_ids=point_class_ids,
            class_min_points=self.class_min_points_per_instance,
        )
        orientation_cues = compute_bev_orientation_cues(
            points=points,
            instance_ids=instance_ids,
            valid_mask=valid_mask,
        )

        # ---- Step 2: 实例级补全量回归 ---- #
        pred_offsets, pred_size_residuals, pred_visibility, agg_coors = self.completion_head(
            pts_feat, sir_coors, valid_mask
        )
        # agg_coors: (K, 3)  (与 obs_centers 对应的实例顺序可能不同，需对齐)
        # 由于 obs_centers 与 sir_coors 的唯一实例对应关系通过 scatter_v2 建立，
        # 这里 AmodalCenterHead 内已用相同 scatter_v2 保持一致的 key 顺序
        # 因此 pred_offsets 与 obs_centers 行对齐（若 K 相同）
        K_pred = pred_offsets.shape[0]
        K_obs = obs_centers.shape[0]

        if K_pred == K_obs:
            C_pred = obs_centers + pred_offsets     # (K, 3)
        else:
            # K 不匹配时（边界情况），直接用观测中心兜底
            C_pred = obs_centers

        completion_outputs = dict(
            center_offsets=pred_offsets,
            size_residuals=pred_size_residuals,
            visibility=pred_visibility,
            agg_coors=agg_coors,
            descriptor=build_completion_descriptor(
                coarse_scores=coarse_occ_prob,
                refine_scores=refine_occ_prob,
                instance_ids=instance_ids,
                pred_size_residuals=pred_size_residuals,
                pred_visibility=pred_visibility,
                center_offsets=pred_offsets,
                orientation_cues=orientation_cues,
            ),
        )
        score_dict = dict(
            coarse_occ_prob=coarse_occ_prob,
            refine_occ_prob=refine_occ_prob,
        )

        # ---- Step 3: 训练时计算损失 ---- #
        occ_losses = {}
        if gt_bboxes_3d_list is not None and self.training:
            # 3a. 生成 OccGT
            occ_gt = self.get_occ_gt(points, batch_idx, gt_bboxes_3d_list)

            # 3b. 获取 completion GT
            if K_pred == K_obs:
                batch_idx_centers = agg_coors[:, 0]
                center_gt, size_gt, visibility_gt, valid_completion_mask = self.get_completion_gt(
                    obs_centers=obs_centers,
                    cluster_coors=agg_coors,
                    batch_idx_centers=batch_idx_centers,
                    gt_bboxes_3d_list=gt_bboxes_3d_list,
                    points=points,
                    sir_coors=sir_coors,
                    valid_mask=valid_mask,
                )
                valid_center_mask = valid_completion_mask
            else:
                batch_idx_centers = batch_idx.new_zeros(K_obs)
                center_gt = obs_centers.new_zeros((K_obs, 3))
                size_gt = obs_centers.new_zeros((K_obs, 3))
                visibility_gt = obs_centers.new_zeros(K_obs)
                valid_center_mask = obs_centers.new_zeros(K_obs, dtype=torch.bool)
                valid_completion_mask = obs_centers.new_zeros(K_obs, dtype=torch.bool)

            # 3c. 计算损失
            occ_losses = self.compute_losses(
                coarse_occ_prob=coarse_occ_prob,
                refine_occ_prob=refine_occ_prob,
                occ_gt=occ_gt,
                pred_offsets=pred_offsets if K_pred == K_obs else pred_offsets.new_zeros((K_obs, 3)),
                center_gt=center_gt,
                valid_center_mask=valid_center_mask,
                pred_size_residuals=pred_size_residuals if K_pred == K_obs else pred_size_residuals.new_zeros((K_obs, 3)),
                size_gt=size_gt,
                pred_visibility=pred_visibility if K_pred == K_obs else pred_visibility.new_zeros(K_obs),
                visibility_gt=visibility_gt,
                valid_completion_mask=valid_completion_mask,
            )

        return valid_mask, C_pred, occ_losses, completion_outputs, score_dict
