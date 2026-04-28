import importlib.util
import math
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from .h_mamba import HMambaInteraction
except ImportError:
    spec = importlib.util.spec_from_file_location(
        "h_mamba_fallback",
        pathlib.Path(__file__).with_name("h_mamba.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    HMambaInteraction = module.HMambaInteraction


def _hilbert_bit_data(point_locs, bit):
    return (point_locs >> bit) & 1


def hilbert_encode(coords, num_bits):
    """Encode integer 3D coordinates into Hilbert indices."""
    if coords.numel() == 0:
        return coords.new_zeros((0,), dtype=torch.long)

    if coords.size(-1) != 3:
        raise ValueError("hilbert_encode expects [..., 3] coordinates")

    if 3 * num_bits >= 63:
        raise ValueError("3D Hilbert encoding must fit in signed int64")

    point_locs = coords.long().clone()
    fig_size = 1 << num_bits
    bit_pow = fig_size >> 1

    while bit_pow > 1:
        mask = bit_pow - 1
        for dim in range(point_locs.size(-1)):
            judge_invert = (point_locs[:, dim] & bit_pow) > 0
            point_locs[judge_invert, 0] ^= mask

            judge_exchange = ~judge_invert
            to_flip = (point_locs[:, 0] ^ point_locs[:, dim]) & mask
            point_locs[judge_exchange, 0] ^= to_flip[judge_exchange]
            point_locs[judge_exchange, dim] ^= to_flip[judge_exchange]
        bit_pow >>= 1

    gray_code = point_locs.new_zeros((point_locs.size(0),), dtype=torch.long)
    for bit_current in range(num_bits):
        bit_data = _hilbert_bit_data(point_locs, bit_current)
        for dim in range(point_locs.size(-1)):
            dim_shift = point_locs.size(-1) - 1 - dim
            gray_code += bit_data[:, dim].long() << (bit_current * point_locs.size(-1) + dim_shift)

    shift = 1
    total_bits = point_locs.size(-1) * num_bits
    while shift < total_bits:
        gray_code ^= gray_code >> shift
        shift <<= 1
    return gray_code


def rotate_centers_z(centers, rotation_id):
    rotation_id = int(rotation_id) % 4
    if rotation_id == 0:
        return centers

    rotated = centers.clone()
    x = centers[:, 0].clone()
    y = centers[:, 1].clone()
    if rotation_id == 1:
        rotated[:, 0] = -y
        rotated[:, 1] = x
    elif rotation_id == 2:
        rotated[:, 0] = -x
        rotated[:, 1] = -y
    else:
        rotated[:, 0] = y
        rotated[:, 1] = -x
    return rotated


def quantize_centers(centers, num_bits):
    if centers.numel() == 0:
        return centers.new_zeros((0, 3), dtype=torch.long)

    mins = centers.min(dim=0)[0]
    maxs = centers.max(dim=0)[0]
    spans = (maxs - mins).clamp(min=1e-3)
    max_value = float((1 << num_bits) - 1)
    quantized = ((centers - mins) / spans * max_value).round()
    quantized = quantized.clamp(min=0, max=max_value)
    return quantized.long()


def serialize_hilbert(centers, batch_ids, num_rotations=2, num_bits=10):
    """Serialize tokens with batched 3D Hilbert ordering."""
    if centers.numel() == 0:
        empty = centers.new_zeros((num_rotations, 0), dtype=torch.long)
        return {"orders": empty, "inverse": empty, "codes": empty}

    orders = []
    inverse = []
    codes = []
    bit_shift = num_bits * 3

    for rotation_id in range(num_rotations):
        rotated = rotate_centers_z(centers, rotation_id)
        quantized = quantize_centers(rotated, num_bits)
        code = hilbert_encode(quantized, num_bits)
        code = (batch_ids.long() << bit_shift) | code
        order = torch.argsort(code)
        inv = torch.empty_like(order)
        inv[order] = torch.arange(order.numel(), device=order.device, dtype=order.dtype)

        orders.append(order)
        inverse.append(inv)
        codes.append(code)

    return {
        "orders": torch.stack(orders, dim=0),
        "inverse": torch.stack(inverse, dim=0),
        "codes": torch.stack(codes, dim=0),
    }


class ForegroundTokenSelector(nn.Module):
    """Deterministic foreground scoring plus per-batch top-k selection."""

    def __init__(self, d_model, keep_ratio=0.75, min_tokens=8, min_per_modality=1):
        super().__init__()
        hidden = max(16, d_model // 4)
        self.keep_ratio = keep_ratio
        self.min_tokens = min_tokens
        self.min_per_modality = min_per_modality

        self.score_refine = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.score_refine[-1].weight)
        nn.init.zeros_(self.score_refine[-1].bias)

    def _base_scores(self, features, cls_logits=None, preds_2d=None):
        feature_score = torch.sigmoid(features.pow(2).mean(dim=-1).sqrt())

        if cls_logits is None or cls_logits.numel() == 0:
            cls_score = feature_score
        else:
            cls_score = torch.sigmoid(cls_logits).amax(dim=-1)

        img_score = features.new_zeros((features.size(0),))
        valid_flag = features.new_zeros((features.size(0),))
        if preds_2d is not None and preds_2d.numel() > 0:
            if preds_2d.size(-1) > 4:
                raw_score = preds_2d[:, 4]
                if raw_score.max() <= 1.0 and raw_score.min() >= 0.0:
                    img_score = raw_score
                else:
                    img_score = torch.sigmoid(raw_score)
            elif preds_2d.size(-1) > 0:
                raw_score = preds_2d[:, 0]
                if raw_score.max() <= 1.0 and raw_score.min() >= 0.0:
                    img_score = raw_score
                else:
                    img_score = torch.sigmoid(raw_score)
            if preds_2d.size(-1) > 8:
                valid_flag = preds_2d[:, 8].clamp(min=0.0, max=1.0)

        refined = 0.05 * torch.tanh(self.score_refine(features).squeeze(-1))
        return 0.60 * cls_score + 0.10 * feature_score + 0.25 * img_score + 0.05 * valid_flag + refined

    def forward(self, features, batch_ids, modality_ids, cls_logits=None, preds_2d=None):
        if features.numel() == 0:
            mask = batch_ids.new_zeros((0,), dtype=torch.bool)
            scores = features.new_zeros((0,))
            return {"mask": mask, "scores": scores, "selected_indices": batch_ids.new_zeros((0,), dtype=torch.long)}

        scores = self._base_scores(features, cls_logits=cls_logits, preds_2d=preds_2d)
        selected_mask = torch.zeros_like(scores, dtype=torch.bool)

        for batch_id in batch_ids.unique(sorted=True):
            batch_mask = batch_ids == batch_id
            batch_indices = batch_mask.nonzero(as_tuple=False).squeeze(-1)
            batch_scores = scores.index_select(0, batch_indices)
            batch_modalities = modality_ids.index_select(0, batch_indices)

            target = max(self.min_tokens, int(math.ceil(batch_indices.numel() * self.keep_ratio)))
            target = min(target, batch_indices.numel())
            keep = torch.zeros_like(batch_scores, dtype=torch.bool)

            if self.min_per_modality > 0:
                for modality in batch_modalities.unique(sorted=True):
                    modality_mask = batch_modalities == modality
                    modality_indices = modality_mask.nonzero(as_tuple=False).squeeze(-1)
                    if modality_indices.numel() == 0:
                        continue
                    topk = min(self.min_per_modality, modality_indices.numel())
                    _, local_idx = torch.topk(batch_scores.index_select(0, modality_indices), k=topk, sorted=False)
                    keep[modality_indices.index_select(0, local_idx)] = True

            remaining = max(0, target - int(keep.sum().item()))
            if remaining > 0:
                available_indices = (~keep).nonzero(as_tuple=False).squeeze(-1)
                _, local_idx = torch.topk(batch_scores.index_select(0, available_indices), k=min(remaining, available_indices.numel()), sorted=False)
                keep[available_indices.index_select(0, local_idx)] = True

            selected_mask[batch_indices] = keep

        selected_indices = selected_mask.nonzero(as_tuple=False).squeeze(-1)
        return {
            "mask": selected_mask,
            "scores": scores,
            "selected_indices": selected_indices,
        }


class ReliabilityGatedFusion(nn.Module):
    """Fuse interacted features back into the residual stream with uncertainty-aware gates."""

    def __init__(self, d_model, descriptor_dim):
        super().__init__()
        hidden = max(32, d_model)
        input_dim = d_model * 2 + descriptor_dim

        self.gate_head = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.var_head = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.zeros_(self.gate_head[-1].bias)
        nn.init.zeros_(self.var_head[-1].weight)
        nn.init.zeros_(self.var_head[-1].bias)

    def forward(self, residual, interacted, descriptors):
        context = torch.cat([residual, interacted, descriptors], dim=-1)
        gate = torch.sigmoid(self.gate_head(context))
        log_var = self.var_head(context).clamp(min=-6.0, max=6.0)
        inv_var = torch.exp(-log_var)
        mix = gate * inv_var / (1.0 + inv_var)
        fused = residual * (1.0 - mix) + interacted * mix
        return fused, {"gate": gate, "log_var": log_var}


class HierarchicalHMambaInteraction(nn.Module):
    """Research-content-1 implementation for FSF instance interaction."""

    def __init__(
        self,
        d_model=256,
        d_state=16,
        expand_factor=2,
        dt_rank="auto",
        conv_kernel=4,
        keep_ratio=0.75,
        min_tokens=8,
        min_per_modality=1,
        num_rotations=2,
        window_size=32,
        use_fast_path=False,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_rotations = num_rotations
        self.window_size = window_size

        self.selector = ForegroundTokenSelector(
            d_model=d_model,
            keep_ratio=keep_ratio,
            min_tokens=min_tokens,
            min_per_modality=min_per_modality,
        )

        self.local_mamba = HMambaInteraction(
            d_model=d_model,
            d_state=d_state,
            expand_factor=expand_factor,
            dt_rank=dt_rank,
            conv_kernel=conv_kernel,
            use_fast_path=use_fast_path,
        )
        self.global_mamba = HMambaInteraction(
            d_model=d_model,
            d_state=d_state,
            expand_factor=expand_factor,
            dt_rank=dt_rank,
            conv_kernel=conv_kernel,
            use_fast_path=use_fast_path,
        )

        self.spatial_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=1, bias=True)
        self.alignment_mlp = nn.Sequential(
            nn.Linear(d_model * 2 + 4, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.semantic_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.reliability_fuser = ReliabilityGatedFusion(d_model=d_model, descriptor_dim=6)
        self.output_norm = nn.LayerNorm(d_model)

        nn.init.zeros_(self.alignment_mlp[-1].weight)
        nn.init.zeros_(self.alignment_mlp[-1].bias)
        nn.init.zeros_(self.semantic_proj[-1].weight)
        nn.init.zeros_(self.semantic_proj[-1].bias)

    def _run_mamba(self, module, seq_feat):
        if seq_feat.numel() == 0:
            return seq_feat
        return seq_feat + module(seq_feat.unsqueeze(0)).squeeze(0)

    def _window_pass(self, seq_feat, shift):
        if seq_feat.size(0) <= self.window_size:
            return self._run_mamba(self.local_mamba, seq_feat)

        output = torch.zeros_like(seq_feat)
        counts = seq_feat.new_zeros((seq_feat.size(0), 1))
        cursor = shift

        if shift > 0:
            prefix = self._run_mamba(self.local_mamba, seq_feat[:shift])
            output[:shift] += prefix
            counts[:shift] += 1

        while cursor < seq_feat.size(0):
            end = min(cursor + self.window_size, seq_feat.size(0))
            window = self._run_mamba(self.local_mamba, seq_feat[cursor:end])
            output[cursor:end] += window
            counts[cursor:end] += 1
            cursor += self.window_size

        return output / counts.clamp(min=1.0)

    def _regional_to_global(self, seq_feat):
        shift = self.window_size // 2
        local = self._window_pass(seq_feat, shift=0)
        if shift > 0 and seq_feat.size(0) > shift:
            local = 0.5 * (local + self._window_pass(seq_feat, shift=shift))
        local = local + F.gelu(
            self.spatial_conv(local.transpose(0, 1).unsqueeze(0)).squeeze(0).transpose(0, 1)
        )
        return self._run_mamba(self.global_mamba, local)

    def _cross_modal_align(self, features, centers, batch_ids, modality_ids, cls_logits=None):
        if features.size(0) == 0:
            return features

        aligned = features.clone()
        for batch_id in batch_ids.unique(sorted=True):
            batch_mask = batch_ids == batch_id
            batch_indices = batch_mask.nonzero(as_tuple=False).squeeze(-1)
            batch_modalities = modality_ids.index_select(0, batch_indices)

            cam_local = (batch_modalities == 0).nonzero(as_tuple=False).squeeze(-1)
            lid_local = (batch_modalities == 1).nonzero(as_tuple=False).squeeze(-1)
            if cam_local.numel() == 0 or lid_local.numel() == 0:
                continue

            batch_centers = centers.index_select(0, batch_indices)
            cost = torch.cdist(
                batch_centers.index_select(0, cam_local),
                batch_centers.index_select(0, lid_local),
            )

            if cls_logits is not None and cls_logits.numel() > 0:
                batch_probs = torch.sigmoid(cls_logits.index_select(0, batch_indices))
                cam_probs = batch_probs.index_select(0, cam_local)
                lid_probs = batch_probs.index_select(0, lid_local)
                semantic_similarity = cam_probs.matmul(lid_probs.t()) / max(1, cam_probs.size(-1))
                cost = cost / (0.1 + semantic_similarity)

            cam_partner = lid_local.index_select(0, cost.argmin(dim=1))
            lid_partner = cam_local.index_select(0, cost.argmin(dim=0))

            partner_local = torch.arange(batch_indices.numel(), device=batch_indices.device)
            partner_local[cam_local] = cam_partner
            partner_local[lid_local] = lid_partner

            partner_global = batch_indices.index_select(0, partner_local)
            self_feat = features.index_select(0, batch_indices)
            partner_feat = features.index_select(0, partner_global)
            rel_center = centers.index_select(0, batch_indices) - centers.index_select(0, partner_global)

            if cls_logits is not None and cls_logits.numel() > 0:
                probs = torch.sigmoid(cls_logits.index_select(0, batch_indices))
                partner_probs = torch.sigmoid(cls_logits.index_select(0, partner_global))
                semantic_score = (probs * partner_probs).mean(dim=-1, keepdim=True)
            else:
                semantic_score = rel_center.new_zeros((rel_center.size(0), 1))

            align_input = torch.cat([self_feat, partner_feat, rel_center, semantic_score], dim=-1)
            align_delta = self.alignment_mlp(align_input)
            align_weight = torch.exp(-rel_center.norm(dim=-1, keepdim=True))
            aligned.index_copy_(0, batch_indices, self_feat + align_weight * align_delta)
        return aligned

    def _reliability_descriptors(self, centers, modality_ids, cls_logits=None, preds_2d=None):
        if centers.numel() == 0:
            return centers.new_zeros((0, 6))

        distance = centers.norm(dim=-1, keepdim=True)
        distance = distance / distance.max().clamp(min=1.0)
        modality = modality_ids.float().unsqueeze(-1)

        if cls_logits is not None and cls_logits.numel() > 0:
            probs = torch.sigmoid(cls_logits).clamp(min=1e-4, max=1 - 1e-4)
            cls_conf = probs.max(dim=-1, keepdim=True)[0]
            entropy = -(probs * probs.log() + (1 - probs) * (1 - probs).log()).mean(dim=-1, keepdim=True)
        else:
            cls_conf = distance.new_zeros((distance.size(0), 1))
            entropy = distance.new_zeros((distance.size(0), 1))

        img_score = distance.new_zeros((distance.size(0), 1))
        valid_2d = distance.new_zeros((distance.size(0), 1))
        if preds_2d is not None and preds_2d.numel() > 0:
            if preds_2d.size(-1) > 4:
                raw_score = preds_2d[:, 4:5]
                if raw_score.max() <= 1.0 and raw_score.min() >= 0.0:
                    img_score = raw_score
                else:
                    img_score = torch.sigmoid(raw_score)
            if preds_2d.size(-1) > 8:
                valid_2d = preds_2d[:, 8:9].clamp(min=0.0, max=1.0)

        return torch.cat([distance, modality, cls_conf, entropy, img_score, valid_2d], dim=-1)

    def forward(
        self,
        features,
        centers,
        batch_ids,
        modality_ids,
        cls_logits=None,
        preds_2d=None,
    ):
        if features.numel() == 0:
            empty_mask = batch_ids.new_zeros((0,), dtype=torch.bool)
            aux = {
                "selected_mask": empty_mask,
                "scores": features.new_zeros((0,)),
                "gate": features.new_zeros((0, 1)),
                "log_var": features.new_zeros((0, 1)),
                "restore_indices": batch_ids.new_zeros((0,), dtype=torch.long),
            }
            return features, aux

        selection = self.selector(
            features,
            batch_ids,
            modality_ids,
            cls_logits=cls_logits,
            preds_2d=preds_2d,
        )
        selected_mask = selection["mask"]
        selected_indices = selection["selected_indices"]

        interacted = features.clone()
        gate = features.new_zeros((features.size(0), 1))
        log_var = features.new_zeros((features.size(0), 1))

        if selected_indices.numel() > 0:
            sel_features = features.index_select(0, selected_indices)
            sel_centers = centers.index_select(0, selected_indices)
            sel_batches = batch_ids.index_select(0, selected_indices)
            sel_modalities = modality_ids.index_select(0, selected_indices)
            sel_logits = None if cls_logits is None or cls_logits.numel() == 0 else cls_logits.index_select(0, selected_indices)
            sel_preds_2d = None if preds_2d is None or preds_2d.numel() == 0 else preds_2d.index_select(0, selected_indices)

            serialization = serialize_hilbert(
                sel_centers,
                sel_batches,
                num_rotations=self.num_rotations,
            )

            rotated_outputs = []
            for order, inverse in zip(serialization["orders"], serialization["inverse"]):
                ordered_feat = sel_features.index_select(0, order)
                ordered_centers = sel_centers.index_select(0, order)
                ordered_batches = sel_batches.index_select(0, order)
                ordered_modalities = sel_modalities.index_select(0, order)
                ordered_logits = None if sel_logits is None else sel_logits.index_select(0, order)

                mixed = self._regional_to_global(ordered_feat)
                mixed = self._cross_modal_align(
                    mixed,
                    ordered_centers,
                    ordered_batches,
                    ordered_modalities,
                    cls_logits=ordered_logits,
                )

                if ordered_logits is not None:
                    semantic_conf = torch.sigmoid(ordered_logits).amax(dim=-1, keepdim=True)
                else:
                    semantic_conf = mixed.new_zeros((mixed.size(0), 1))
                mixed = mixed + semantic_conf * self.semantic_proj(mixed)
                rotated_outputs.append(mixed.index_select(0, inverse))

            aggregated = torch.stack(rotated_outputs, dim=0).mean(dim=0)
            descriptors = self._reliability_descriptors(
                sel_centers,
                sel_modalities,
                cls_logits=sel_logits,
                preds_2d=sel_preds_2d,
            )
            fused_selected, gate_aux = self.reliability_fuser(sel_features, aggregated, descriptors)
            fused_selected = self.output_norm(fused_selected)

            interacted.index_copy_(0, selected_indices, fused_selected)
            gate.index_copy_(0, selected_indices, gate_aux["gate"])
            log_var.index_copy_(0, selected_indices, gate_aux["log_var"])

        aux = {
            "selected_mask": selected_mask,
            "scores": selection["scores"],
            "gate": gate,
            "log_var": log_var,
            "restore_indices": torch.arange(features.size(0), device=features.device, dtype=torch.long),
        }
        return interacted, aux
