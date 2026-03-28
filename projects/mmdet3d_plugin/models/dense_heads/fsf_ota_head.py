"""
FSFOTAHead: FullySparseFusion detection head with Optimal Transport Assignment.

Inherits from FrustumClusterHead and replaces the heuristic 3-D/2-D hybrid
assignment with an entropy-regularised Optimal Transport (OTA) assigner
(Sinkhorn-Knopp in log-domain).

Key changes vs. FrustumClusterHead
───────────────────────────────────
• `get_targets_single` is overridden.
  – Decodes current reg_preds to full 3-D boxes.
  – Calls OTAAssigner with (cluster_xyz, cls_logits, decoded_boxes, gt_boxes,
    gt_labels) → AssignResult.
  – Falls back to the parent hybrid-assigner when `use_ota=False`.
• Everything else (loss computation, NMS inference, multi-task heads) is
  inherited unchanged from FrustumClusterHead / SparseClusterHeadV2.
"""
import torch
from mmdet.core import build_assigner, multi_apply, reduce_mean
from mmdet.models import HEADS

from .frustum_cluster_head import FrustumClusterHead
from mmdet3d.core import AssignResult, PseudoSampler


@HEADS.register_module()
class FSFOTAHead(FrustumClusterHead):
    """FrustumClusterHead variant that uses OTAAssigner for label assignment.

    Extra constructor args (all others forwarded to FrustumClusterHead):
        ota_assigner (dict): Config for :class:`OTAAssigner`.
        use_ota (bool): If False, fall back to parent hybrid assigner
            (useful for warm-up first N epochs). Default: True.
    """

    def __init__(
        self,
        *args,
        ota_assigner=None,
        use_ota=True,
        **kwargs,
    ):
        # Pop ota-specific kwargs before forwarding to parent
        super().__init__(*args, **kwargs)

        self.use_ota = use_ota
        if ota_assigner is not None:
            self.ota_assigner = build_assigner(ota_assigner)
        else:
            self.use_ota = False
            self.ota_assigner = None

    # ------------------------------------------------------------------
    # Override get_targets_single with OTA
    # ------------------------------------------------------------------

    def get_targets_single(
        self,
        num_task_classes,
        preds_2d,
        cluster_xyz,
        no_aug_gt_bboxes_3d,
        no_aug_gt_labels_3d,
        gt_bboxes_3d,
        gt_labels_3d,
        reg_preds=None,
        new_cls_logits=None,
        old_cls_logits=None,
        old_reg_preds=None,
        task_id=None,
        img_metas=None,
    ):
        """Override: use OTA assigner instead of heuristic hybrid assign."""

        if not self.use_ota or reg_preds is None or new_cls_logits is None:
            # Fall back to parent implementation
            return super().get_targets_single(
                num_task_classes,
                preds_2d,
                cluster_xyz,
                no_aug_gt_bboxes_3d,
                no_aug_gt_labels_3d,
                gt_bboxes_3d,
                gt_labels_3d,
                reg_preds=reg_preds,
                new_cls_logits=new_cls_logits,
                old_cls_logits=old_cls_logits,
                old_reg_preds=old_reg_preds,
                task_id=task_id,
                img_metas=img_metas,
            )

        # ── Initialise output tensors ──────────────────────────────────
        num_cluster = cluster_xyz.shape[0]
        labels      = gt_labels_3d.new_full(
            (num_cluster,), num_task_classes, dtype=torch.long)
        label_weights = preds_2d.new_ones(num_cluster)
        bbox_targets  = preds_2d.new_zeros((num_cluster, self.box_code_size))
        bbox_weights  = preds_2d.new_zeros((num_cluster, self.box_code_size))

        if num_cluster == 0:
            self.task_info[str(task_id)] = dict(
                num_preds=torch.tensor(0, dtype=torch.float32, device=preds_2d.device),
                num_pos_preds=torch.tensor(0, dtype=torch.float32, device=preds_2d.device),
                num_gts=torch.tensor(
                    int(gt_bboxes_3d.tensor.shape[0]), dtype=torch.float32, device=preds_2d.device),
                assigned_gts=torch.tensor(0, dtype=torch.float32, device=preds_2d.device),
            )
            iou_labels = None
            if self.loss_iou is not None:
                iou_labels = preds_2d.new_zeros(0)
            return labels, label_weights, bbox_targets, bbox_weights, iou_labels

        # ── Move GT to device ──────────────────────────────────────────
        no_aug_gt_bboxes_3d = no_aug_gt_bboxes_3d.to(preds_2d.device)
        no_aug_gt_labels_3d = no_aug_gt_labels_3d.to(preds_2d.device)
        gt_bboxes_3d  = gt_bboxes_3d.to(preds_2d.device)
        gt_labels_3d  = gt_labels_3d.to(preds_2d.device)

        # ── Decode current predictions ─────────────────────────────────
        # reg_preds: (M, code_size) – task-local encoded predictions
        # cluster_xyz: (M, 3) – cluster centroids used as base points
        with torch.no_grad():
            decoded_boxes = self.bbox_coder.decode(
                reg_preds.detach(), cluster_xyz.detach())  # (M, 7 or 10)

        # ── OTA assignment ─────────────────────────────────────────────
        gt_tensor = gt_bboxes_3d.tensor            # (N, 10)
        assign_result = self.ota_assigner.assign(
            pred_xyz=cluster_xyz.detach(),
            pred_cls_logits=new_cls_logits.detach(),
            pred_bboxes_decoded=decoded_boxes,
            gt_bboxes_tensor=gt_tensor,
            gt_labels=gt_labels_3d,
        )

        # ── Sampling (pseudo: keep all assigned) ─────────────────────
        sample_result = self.sampler.sample(
            assign_result,
            gt_bboxes_3d.tensor.new_zeros((num_cluster, 9)),
            gt_bboxes_3d.tensor,
        )

        # ── Log statistics ────────────────────────────────────────────
        self.task_info[str(task_id)] = dict(
            num_preds=torch.tensor(
                cluster_xyz.shape[0], dtype=torch.float32, device=preds_2d.device),
            num_pos_preds=torch.tensor(
                sample_result.pos_inds.shape[0], dtype=torch.float32, device=preds_2d.device),
            num_gts=torch.tensor(
                sample_result.num_gts, dtype=torch.float32, device=preds_2d.device),
            assigned_gts=torch.tensor(
                sample_result.pos_assigned_gt_inds.unique().shape[0],
                dtype=torch.float32, device=preds_2d.device),
        )

        pos_inds = sample_result.pos_inds

        # ── Fill targets ──────────────────────────────────────────────
        labels[pos_inds] = gt_labels_3d[sample_result.pos_assigned_gt_inds]
        assert (labels >= 0).all()
        bbox_weights[pos_inds] = 1.0

        if len(pos_inds) > 0:
            bbox_targets[pos_inds] = self.bbox_coder.encode(
                sample_result.pos_gt_bboxes,
                cluster_xyz[pos_inds].detach(),
            )
            # Zero velocity loss weight for copy-pasted objects
            if sample_result.pos_gt_bboxes.size(1) == 10:
                assert sample_result.pos_gt_bboxes[:, 9].max().item() in (0, 1)
                assert sample_result.pos_gt_bboxes[:, 9].min().item() in (0, 1)
                assert bbox_weights.size(1) == 10
                bbox_weights[pos_inds, -2:] = sample_result.pos_gt_bboxes[:, [9]]

        if self.loss_iou is not None:
            iou_labels = self.get_dist_labels(cluster_xyz, sample_result)
        else:
            iou_labels = None

        return labels, label_weights, bbox_targets, bbox_weights, iou_labels
