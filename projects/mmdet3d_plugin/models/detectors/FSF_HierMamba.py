import torch
from mmdet.models import DETECTORS

from .FSF import FSF
from projects.mmdet3d_plugin.models.utils.hierarchical_hmamba import (
    HierarchicalHMambaInteraction,
)


@DETECTORS.register_module()
class FSF_HierMamba(FSF):
    """FSF detector with hierarchical Hilbert-serialized H-Mamba interaction."""

    def __init__(self, hiermamba_cfg=None, hmamba_cfg=None, **kwargs):
        super().__init__(**kwargs)

        cfg = {} if hiermamba_cfg is None else dict(hiermamba_cfg)
        d_model = cfg.get("d_model", self.embed_dims)

        self.hier_mamba = HierarchicalHMambaInteraction(
            d_model=d_model,
            d_state=cfg.get("d_state", 16),
            expand_factor=cfg.get("expand_factor", 2),
            dt_rank=cfg.get("dt_rank", "auto"),
            conv_kernel=cfg.get("conv_kernel", 4),
            keep_ratio=cfg.get("keep_ratio", 0.75),
            min_tokens=cfg.get("min_tokens", 8),
            min_per_modality=cfg.get("min_per_modality", 1),
            class_groups=cfg.get("class_groups", None),
            class_group_min_tokens=cfg.get("class_group_min_tokens", None),
            num_rotations=cfg.get("num_rotations", 2),
            window_size=cfg.get("window_size", 32),
            use_fast_path=cfg.get("use_fast_path", False),
            use_extended_reliability=cfg.get("use_extended_reliability", False),
            small_object_class_indices=cfg.get("small_object_class_indices", None),
            small_object_residual_scale=cfg.get("small_object_residual_scale", 0.1),
        )

    def _flatten_batched_tensor(self, batched_tensor, batch_ids):
        if batched_tensor is None:
            return None

        out_channels = 0
        device = batch_ids.device
        dtype = torch.float32
        for tensor in batched_tensor:
            if tensor.numel() > 0:
                out_channels = tensor.shape[-1]
                device = tensor.device
                dtype = tensor.dtype
                break
        if out_channels == 0:
            return torch.zeros((batch_ids.shape[0], 0), device=device, dtype=dtype)

        flat = torch.zeros((batch_ids.shape[0], out_channels), device=device, dtype=dtype)
        for batch_idx, tensor in enumerate(batched_tensor):
            if tensor.numel() == 0:
                continue
            mask = batch_ids == batch_idx
            if mask.sum() == 0:
                continue
            indices = mask.nonzero(as_tuple=False).squeeze(-1)
            count = min(indices.numel(), tensor.shape[0])
            flat.index_copy_(0, indices[:count], tensor[:count])
        return flat

    def combine_frustum_and_fsd(
        self,
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
        obj_centers = torch.cat([frustum_obj_centers, fsd_obj_centers], dim=0)

        fsd_obj_coors_re = fsd_obj_coors.clone()
        fsd_obj_coors_re[:, 0] = fsd_obj_coors[:, 1]
        fsd_obj_coors_re[:, 1] = fsd_obj_coors[:, 0]
        fsd_obj_coors_re[:, 2] += self.fsd_begin_idx
        obj_coors = torch.cat([frustum_obj_coors, fsd_obj_coors_re], dim=0)

        obj_result = {}
        for key in frustum_obj_result.keys():
            batch_size = len(frustum_obj_result[key])
            obj_result[key] = []
            for bidx in range(batch_size):
                obj_result[key].append(
                    torch.cat([frustum_obj_result[key][bidx], fsd_obj_result[key][bidx]], dim=0)
                )

        frustum_proj = self.combine_frustum_feat_mlp(frustum_obj_feats)
        fsd_proj = self.combine_fsd_feat_mlp(fsd_obj_feats)
        obj_feats_cat = torch.cat([frustum_proj, fsd_proj], dim=0)

        fsd_preds_2d = frustum_preds_2d.new_zeros((fsd_obj_feats.shape[0], frustum_preds_2d.shape[1]))
        preds_2d = torch.cat([frustum_preds_2d, fsd_preds_2d], dim=0)

        modality_ids = torch.cat(
            [
                obj_feats_cat.new_zeros((frustum_proj.shape[0],), dtype=torch.long),
                obj_feats_cat.new_ones((fsd_proj.shape[0],), dtype=torch.long),
            ],
            dim=0,
        )
        batch_ids = obj_coors[:, 0].long()
        cls_logits = self._flatten_batched_tensor(obj_result.get("cls_logits", None), batch_ids)

        fused_obj_feats, aux = self.hier_mamba(
            obj_feats_cat,
            obj_centers,
            batch_ids,
            modality_ids,
            cls_logits=cls_logits,
            preds_2d=preds_2d,
        )
        self.runtime_info["hier_mamba_aux"] = aux

        return obj_centers, obj_coors, obj_result, fused_obj_feats, preds_2d
