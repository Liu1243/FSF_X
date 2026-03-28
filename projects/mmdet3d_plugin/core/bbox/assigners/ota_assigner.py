"""
Cross-Modal Boundary-Aware Optimal Transport Assignment (OTA)

Reformulates label assignment as an entropy-regularised Optimal Transport (OT)
problem solved via the Sinkhorn-Knopp algorithm in log-space.

Cost matrix:
    C_{i,j} = λ_cls * L_cls(p_i, q_j)
             + λ_giou * L_GIoU(b_i, g_j)
             + λ_center * ||ĉ_i - c_j*||_1

Entropy-regularised OT objective (Sinkhorn):
    min_π Σ_{i,j} π_{ij} C_{ij} + ε H(π)
subject to:  π 1 = μ,  πᵀ 1 = ν

Solved in log-domain for numerical stability.
"""
import torch
import torch.nn.functional as F
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.assigners import AssignResult, BaseAssigner
from mmdet3d.core import bbox_overlaps_3d


@BBOX_ASSIGNERS.register_module()
class OTAAssigner(BaseAssigner):
    """Optimal Transport Assigner for 3-D detection.

    Args:
        num_classes (int): Number of foreground classes.
        lambda_cls (float): Weight for classification cost.
        lambda_giou (float): Weight for GIoU cost.
        lambda_center (float): Weight for L1 centre-distance cost.
        eps (float): Entropy regularisation strength ε.
        sinkhorn_iters (int): Number of Sinkhorn iterations.
        match_score_thre (float): Minimum π value to be treated as positive.
            Predictions whose maximum column score falls below this are
            assigned to background even if argmax selects a foreground GT.
        bg_col_cost (float): Fixed cost of the virtual background column.
            Typically 0 to make background "free to absorb".
        candidate_topk (int): For each GT, only keep the top-k cheapest
            predictions as candidates (−1 = use all).  Reducing this value
            speeds up OT while keeping the most relevant predictions.
    """

    def __init__(
        self,
        num_classes,
        lambda_cls=1.0,
        lambda_giou=2.0,
        lambda_center=1.0,
        eps=0.1,
        sinkhorn_iters=50,
        match_score_thre=0.0,
        bg_col_cost=0.0,
        candidate_topk=-1,
    ):
        self.num_classes = num_classes
        self.lambda_cls = lambda_cls
        self.lambda_giou = lambda_giou
        self.lambda_center = lambda_center
        self.eps = eps
        self.sinkhorn_iters = sinkhorn_iters
        self.match_score_thre = match_score_thre
        self.bg_col_cost = bg_col_cost
        self.candidate_topk = candidate_topk

    # ------------------------------------------------------------------
    # Cost components
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _cls_cost(self, cls_logits, gt_labels):
        """Focal-loss-style classification cost.

        Args:
            cls_logits (Tensor): (M, C) raw (pre-sigmoid) logits.
            gt_labels  (Tensor): (N,) integer GT class indices in [0, C-1].

        Returns:
            Tensor: (M, N) cost matrix.
        """
        M = cls_logits.size(0)
        N = gt_labels.size(0)
        device = cls_logits.device

        probs = cls_logits.sigmoid()          # (M, C)
        alpha, gamma = 0.25, 2.0

        # Positive focal loss for each (pred, gt) pair
        # target class probability
        tgt_prob = probs[:, gt_labels]        # (M, N)
        pos_cost = -(alpha
                     * (1 - tgt_prob) ** gamma
                     * torch.log(tgt_prob + 1e-8))

        # Negative focal for ALL other classes accumulated per pred
        neg_cost_per_pred = -(
            (1 - alpha) * (probs ** gamma) * torch.log(1 - probs + 1e-8)
        ).sum(dim=1, keepdim=True)                  # (M, 1)

        # Subtract the neg-cost of the true class (which is included above)
        true_cls_neg = -(
            (1 - alpha) * (tgt_prob ** gamma) * torch.log(1 - tgt_prob + 1e-8)
        )                                            # (M, N)

        cost = pos_cost + neg_cost_per_pred - true_cls_neg  # (M, N)
        return cost

    @torch.no_grad()
    def _giou_cost(self, pred_bboxes, gt_bboxes):
        """GIoU cost = 1 − GIoU.

        Args:
            pred_bboxes (Tensor): (M, 7+) decoded prediction boxes.
            gt_bboxes   (Tensor): (N, 7+) GT boxes.

        Returns:
            Tensor: (M, N) cost matrix.
        """
        # bbox_overlaps_3d returns IoU/GIoU matrix of shape (M, N)
        try:
            giou = bbox_overlaps_3d(
                pred_bboxes[:, :7],
                gt_bboxes[:, :7],
                mode='giou',
                coordinate='lidar',
            )
        except Exception:
            # Fallback to IoU if GIoU not supported
            giou = bbox_overlaps_3d(
                pred_bboxes[:, :7],
                gt_bboxes[:, :7],
                mode='iou',
                coordinate='lidar',
            )
        return (1.0 - giou).clamp(min=0.0)  # (M, N)

    @torch.no_grad()
    def _center_cost(self, pred_xyz, gt_centers):
        """L1 BEV-centre distance cost.

        Args:
            pred_xyz   (Tensor): (M, 3) predicted cluster centroids.
            gt_centers (Tensor): (N, 3) GT gravity centres.

        Returns:
            Tensor: (M, N) cost matrix.
        """
        pred_bev = pred_xyz[:, :2]                       # (M, 2)
        gt_bev   = gt_centers[:, :2]                     # (N, 2)
        # L1 sum over xy
        cost = (pred_bev[:, None, :] - gt_bev[None, :, :]).abs().sum(dim=-1)  # (M, N)
        return cost

    # ------------------------------------------------------------------
    # Sinkhorn-Knopp in log-domain
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _sinkhorn_log(self, log_alpha, mu, nu):
        """Solve entropy-regularised OT via Sinkhorn in log-domain.

        Args:
            log_alpha (Tensor): (M, N) initial log-kernel = −C / ε.
            mu        (Tensor): (M,) source marginals (sum to 1).
            nu        (Tensor): (N,) target marginals (sum to 1).

        Returns:
            Tensor: (M, N) transport plan π.
        """
        log_mu = mu.log()
        log_nu = nu.log()

        # Initialise dual variables u, v to zero
        log_u = torch.zeros_like(mu)
        log_v = torch.zeros_like(nu)

        for _ in range(self.sinkhorn_iters):
            # u-update: u_i = μ_i / Σ_j K_{ij} v_j
            log_u = log_mu - torch.logsumexp(log_alpha + log_v[None, :], dim=1)
            # v-update: v_j = ν_j / Σ_i K_{ij} u_i
            log_v = log_nu - torch.logsumexp(log_alpha + log_u[:, None], dim=0)

        # π_{ij} = u_i * K_{ij} * v_j
        log_pi = log_alpha + log_u[:, None] + log_v[None, :]
        pi = log_pi.exp()
        return pi

    # ------------------------------------------------------------------
    # Main assign
    # ------------------------------------------------------------------

    @torch.no_grad()
    def assign(
        self,
        pred_xyz,
        pred_cls_logits,
        pred_bboxes_decoded,
        gt_bboxes_tensor,
        gt_labels,
    ):
        """Assign GT to predictions via Optimal Transport.

        Args:
            pred_xyz           (Tensor): (M, 3) cluster centroids.
            pred_cls_logits    (Tensor): (M, C) raw class logits (pre-sigmoid).
            pred_bboxes_decoded(Tensor): (M, 7+) decoded bounding boxes.
            gt_bboxes_tensor   (Tensor): (N, 7+) GT boxes (LiDAR coord).
            gt_labels          (Tensor): (N,) GT class indices (task-local).

        Returns:
            AssignResult: mmdet-style assign result.
        """
        device = pred_xyz.device
        M = pred_xyz.size(0)
        N = gt_bboxes_tensor.size(0)

        # Trivial cases ─────────────────────────────────────────────────
        assigned_gt_inds = pred_xyz.new_zeros((M,), dtype=torch.long)
        assigned_labels  = pred_xyz.new_full((M,), -1, dtype=torch.long)

        if N == 0:
            assigned_gt_inds[:] = 0          # all background
            return AssignResult(N, assigned_gt_inds, None, labels=assigned_labels)
        if M == 0:
            return AssignResult(N, assigned_gt_inds, None, labels=assigned_labels)

        # ── Build cost matrix ──────────────────────────────────────────
        cost_cls = self._cls_cost(pred_cls_logits, gt_labels)         # (M, N)

        cost_giou = self._giou_cost(pred_bboxes_decoded, gt_bboxes_tensor)  # (M, N)

        # GT gravity centres (first 3 dims are xyz centre in nuScenes GT)
        gt_centers = gt_bboxes_tensor[:, :3]                          # (N, 3)
        cost_center = self._center_cost(pred_xyz, gt_centers)         # (M, N)

        # Normalise each cost term to [0,1] range for stability
        def _normalize(x):
            mn, mx = x.min(), x.max()
            return (x - mn) / (mx - mn + 1e-8)

        cost = (
            self.lambda_cls    * _normalize(cost_cls)
            + self.lambda_giou   * _normalize(cost_giou)
            + self.lambda_center * _normalize(cost_center)
        )                                                              # (M, N)

        # ── Optional: candidate pool per GT ───────────────────────────
        if self.candidate_topk > 0 and M > self.candidate_topk:
            # For each GT, mask out all but top-k cheapest predictions
            topk_val, topk_inds = cost.topk(
                min(self.candidate_topk, M), dim=0, largest=False)
            inf_mask = torch.ones_like(cost) * 1e9
            inf_mask.scatter_(0, topk_inds, 0.0)
            cost = cost + inf_mask

        # ── Append background column ───────────────────────────────────
        bg_col = cost.new_full((M, 1), self.bg_col_cost)
        cost_aug = torch.cat([cost, bg_col], dim=1)                   # (M, N+1)

        # ── Marginals (uniform): μ ∈ ℝ^M, ν ∈ ℝ^{N+1} ────────────────
        mu = cost_aug.new_full((M,), 1.0 / M)
        # Give N/(N+1) mass to foreground GTs and 1/(N+1) to background
        nu_fg = cost_aug.new_full((N,), 1.0 / (N + 1))
        nu_bg = cost_aug.new_full((1,), 1.0 / (N + 1))
        nu = torch.cat([nu_fg, nu_bg], dim=0)                         # (N+1,)

        # ── Sinkhorn ───────────────────────────────────────────────────
        log_alpha = -cost_aug / (self.eps + 1e-8)                     # (M, N+1)
        pi = self._sinkhorn_log(log_alpha, mu, nu)                    # (M, N+1)

        # ── Hard label extraction ──────────────────────────────────────
        # Score per prediction towards each foreground GT
        pi_fg  = pi[:, :N]                                            # (M, N)
        max_pi, best_gt = pi_fg.max(dim=1)                            # (M,)

        # A prediction is positive iff:
        #   1) its foreground score > background score, AND
        #   2) its foreground score > match_score_thre
        pi_bg    = pi[:, N]                                            # (M,)
        pos_mask = (max_pi > pi_bg) & (max_pi > self.match_score_thre)

        # Build AssignResult (1-indexed; 0 = background, -1 = ignore)
        assigned_gt_inds[~pos_mask] = 0                               # background
        assigned_gt_inds[pos_mask]  = best_gt[pos_mask] + 1          # 1-indexed
        assigned_labels[pos_mask]   = gt_labels[best_gt[pos_mask]]

        return AssignResult(N, assigned_gt_inds, None, labels=assigned_labels)
