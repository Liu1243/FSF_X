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


class AmodalCenterHead(BaseModule):
    """实例级非模态中心偏移回归网络。

    对每个视锥实例（每个 obj_id 对应一个实例），聚合有效占据点的特征，
    回归出从观测中心 C_obs 到非模态中心 C_gt 的偏移量 Δc。

    修正后的非模态中心：C_pred = C_obs + Δc

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
        xyz_normalizer: list = [20.0, 20.0, 4.0],
    ):
        super().__init__()
        self.xyz_normalizer = xyz_normalizer

        # 输出 3 维偏移（x, y, z）
        self.mlp = build_mlp(
            in_channels,
            hidden_dims + [3],
            norm_cfg,
            is_head=True,
            act=act,
        )

    def forward(
        self,
        pts_feat: torch.Tensor,
        sir_coors: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """对每个实例内有效点做 MaxPool 聚合后回归偏移。

        Args:
            pts_feat (Tensor): (N, C) 全部点特征（未过滤）
            sir_coors (Tensor): (N, 3) SIR 分组坐标 (batch_id, ?, obj_id)
            valid_mask (Tensor): (N,) bool，OccMLP 输出的有效点掩码

        Returns:
            cluster_offsets (Tensor): (K, 3) 每个实例的中心偏移量
            cluster_coors   (Tensor): (K, 3) 每个实例的 SIR 坐标
        """
        # 只使用有效（前景）点做聚合
        if valid_mask.sum() == 0:
            # 边界情况：全部被过滤掉，回零偏移
            unique_coors = sir_coors.new_zeros((1, 3))
            offsets = pts_feat.new_zeros((1, 3))
            return offsets, unique_coors

        valid_feat = pts_feat[valid_mask]       # (N_valid, C)
        valid_coors = sir_coors[valid_mask]     # (N_valid, 3)

        # 按实例（sir_coors 唯一键）做 Max Pooling
        agg_feat, agg_coors, _ = scatter_v2(
            valid_feat, valid_coors, mode='max'
        )
        # agg_feat: (K, C)  agg_coors: (K, 3)

        cluster_offsets = self.mlp(agg_feat)    # (K, 3)
        return cluster_offsets, agg_coors


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
        amodal_head_cfg: dict,
        occ_thr: float = 0.3,
        loss_occ_weight: float = 1.0,
        loss_center_weight: float = 0.5,
    ):
        super().__init__()
        self.occ_thr = occ_thr
        self.loss_occ_weight = loss_occ_weight
        self.loss_center_weight = loss_center_weight

        self.occ_mlp = OccMLP(**occ_mlp_cfg)
        self.amodal_center_head = AmodalCenterHead(**amodal_head_cfg)

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

    @force_fp32(apply_to=('occ_prob', 'pred_offsets'))
    def compute_losses(
        self,
        occ_prob: torch.Tensor,
        occ_gt: torch.Tensor,
        pred_offsets: torch.Tensor,
        center_gt: torch.Tensor,
        valid_center_mask: torch.Tensor,
    ) -> dict:
        """计算占据损失和非模态中心回归损失。

        Args:
            occ_prob   (N,)   : 占据概率预测值
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
        bce = F.binary_cross_entropy(occ_prob, occ_gt, reduction='none')
        pt = occ_prob * occ_gt + (1 - occ_prob) * (1 - occ_gt)
        focal_weight = alpha * occ_gt + (1 - alpha) * (1 - occ_gt)
        focal_weight = focal_weight * (1 - pt).pow(gamma)
        loss_occ = (bce * focal_weight).mean()
        losses['loss_occ'] = self.loss_occ_weight * loss_occ

        # ---- L_amodal_center: Smooth-L1 ---- #
        if valid_center_mask.sum() > 0:
            pos_pred = pred_offsets[valid_center_mask]
            pos_gt = center_gt[valid_center_mask]
            loss_center = F.smooth_l1_loss(pos_pred, pos_gt, reduction='mean', beta=1.0)
        else:
            loss_center = pred_offsets.sum() * 0.0
        losses['loss_amodal_center'] = self.loss_center_weight * loss_center

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
        # ---- Step 1: 逐点占据概率预测 ---- #
        occ_prob = self.occ_mlp(pts_feat)              # (N,)
        valid_mask = occ_prob > self.occ_thr            # (N,) bool

        # 若全部被过滤，保留所有点（避免空 tensor 崩溃）
        if valid_mask.sum() == 0:
            valid_mask = torch.ones_like(valid_mask, dtype=torch.bool)

        # ---- Step 2: 非模态中心偏移回归 ---- #
        # 仅对有效点聚合得到实例级特征，然后回归偏移
        pred_offsets, agg_coors = self.amodal_center_head(
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

        # ---- Step 3: 训练时计算损失 ---- #
        occ_losses = {}
        if gt_bboxes_3d_list is not None and self.training:
            # 3a. 生成 OccGT
            occ_gt = self.get_occ_gt(points, batch_idx, gt_bboxes_3d_list)

            # 3b. 获取 obs_centers 对应的 batch_idx
            # obs_centers 与 agg_coors 的 batch_id 在第 0 列
            # 使用有效点 sir_coors 散射后得到的 agg_coors batch 列
            if K_pred == K_obs:
                batch_idx_centers = agg_coors[:, 0]
            else:
                batch_idx_centers = batch_idx.new_zeros(K_obs)

            # 3c. 生成 AmodalCenterGT
            center_gt, valid_center_mask = self.get_amodal_center_gt(
                obs_centers,
                agg_coors,
                batch_idx_centers,
                gt_bboxes_3d_list,
            )

            # 3d. 计算损失
            occ_losses = self.compute_losses(
                occ_prob, occ_gt,
                pred_offsets if K_pred == K_obs else pred_offsets.new_zeros(K_obs, 3),
                center_gt,
                valid_center_mask,
            )

        return valid_mask, C_pred, occ_losses
