"""Contrastive term of the Hyper3-CLIP objective (paper Sec. 3.3, ``L_con``).

This is the *inherited* UNCHA contrastive computation, unchanged by this paper:
three cross-entropy groups over Lorentz distances,

* **global** whole image vs. whole caption,
* **local** box crop vs. box phrase,
* **global-local** box crop vs. whole caption (and box phrase vs. whole image),
  whose logits are modulated per row by a detached, radius-derived uncertainty
  temperature.

Logits are ``-d_L(x, y)`` scaled by ``exp(logit_scale).clamp(max=100)``; each
group is the symmetric average of the two row-wise cross-entropies.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from hyper3_clip.geometry.lorentz import pairwise_dist

__all__ = ["contrastive_losses", "embedding_uncertainty"]


def embedding_uncertainty(x: Tensor) -> Tensor:
    """Radius-derived log-uncertainty ``u(x) = softplus(-||x_bar||)``.

    Small radius (a point close to the origin of the hyperboloid, i.e. a
    generic concept) means high uncertainty.  This single quantity plays two
    roles in the objective: a detached logit temperature in the global-local
    contrastive group, and the ``log sigma`` of the entailment calibration term
    (see :mod:`hyper3_clip.models.entailment`).
    """
    norm = torch.linalg.norm(x[..., 1:].float(), dim=-1)
    if norm.dim() > 1:
        norm = norm.mean(dim=-1)
    return F.softplus(-norm)


def _cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    return F.cross_entropy(logits, targets)


def contrastive_losses(
    image_feats: Tensor,
    text_feats: Tensor,
    part_image_flat: Tensor,
    part_text_flat: Tensor,
    image_for_parts: Tensor,
    text_for_parts: Tensor,
    kappa: Tensor,
    *,
    global_logit_scale: Tensor,
    local_logit_scale: Tensor,
    global_local_logit_scale: Tensor,
    all_image_feats: Tensor | None = None,
    all_text_feats: Tensor | None = None,
    all_part_image_feats: Tensor | None = None,
    all_part_text_feats: Tensor | None = None,
    all_image_for_parts: Tensor | None = None,
    all_text_for_parts: Tensor | None = None,
    global_targets: Tensor | None = None,
    part_targets: Tensor | None = None,
    row_loss_scale: Tensor | None = None,
) -> dict[str, Tensor]:
    """Return ``{"contrastive", "global", "local", "global_local"}`` losses.

    ``all_*`` arguments carry the DDP-gathered negative pools; they default to
    the rank-local tensors, which is the correct single-process behaviour.  The
    global-local negatives are the ``P`` owner-repeated whole-image / whole-
    caption rows, scored against the part rows with part-index targets.
    """
    all_image_feats = image_feats if all_image_feats is None else all_image_feats
    all_text_feats = text_feats if all_text_feats is None else all_text_feats
    all_part_image_feats = part_image_flat if all_part_image_feats is None else all_part_image_feats
    all_part_text_feats = part_text_flat if all_part_text_feats is None else all_part_text_feats
    all_image_for_parts = image_for_parts if all_image_for_parts is None else all_image_for_parts
    all_text_for_parts = text_for_parts if all_text_for_parts is None else all_text_for_parts
    if global_targets is None:
        global_targets = torch.arange(image_feats.size(0), device=image_feats.device)
    if part_targets is None:
        part_targets = torch.arange(part_image_flat.size(0), device=part_image_flat.device)
    if row_loss_scale is None:
        row_loss_scale = image_feats.new_ones(())

    global_scale = global_logit_scale.exp().clamp(max=100.0)
    local_scale = local_logit_scale.exp().clamp(max=100.0)
    global_local_scale = global_local_logit_scale.exp().clamp(max=100.0)

    image_logits = -pairwise_dist(image_feats, all_text_feats, kappa) * global_scale
    text_logits = -pairwise_dist(text_feats, all_image_feats, kappa) * global_scale
    global_contrastive = 0.5 * (
        _cross_entropy(image_logits, global_targets) + _cross_entropy(text_logits, global_targets)
    )

    if part_image_flat.numel() == 0:
        # Keep every parameter that would otherwise be unused in the autograd
        # graph so the DDP reduction invariant still holds on part-free batches.
        zero = (
            part_image_flat.sum()
            + part_text_flat.sum()
            + all_part_image_feats.sum() * 0.0
            + all_part_text_feats.sum() * 0.0
            + local_scale * 0.0
            + global_local_scale * 0.0
            + all_image_for_parts.sum() * 0.0
            + all_text_for_parts.sum() * 0.0
        )
        return {
            "contrastive": global_contrastive + zero,
            "contrastive_global": global_contrastive,
            "contrastive_local": zero,
            "contrastive_global_local": zero,
        }

    part_image_logits = -pairwise_dist(part_image_flat, all_part_text_feats, kappa) * local_scale
    part_text_logits = -pairwise_dist(part_text_flat, all_part_image_feats, kappa) * local_scale
    local_contrastive = (
        row_loss_scale
        * 0.5
        * (_cross_entropy(part_image_logits, part_targets) + _cross_entropy(part_text_logits, part_targets))
    )

    # Global-local: each part row against the owner-repeated whole-image and
    # whole-caption rows, with the row's distances first scaled by a detached,
    # radius-derived uncertainty temperature.
    part_image_to_whole_text = -pairwise_dist(part_image_flat, all_text_for_parts, kappa)
    part_text_to_whole_image = -pairwise_dist(part_text_flat, all_image_for_parts, kappa)
    image_temp = torch.exp(-0.5 * embedding_uncertainty(part_image_flat).detach()).clamp(min=0.1, max=10.0)
    text_temp = torch.exp(-0.5 * embedding_uncertainty(part_text_flat).detach()).clamp(min=0.1, max=10.0)
    part_image_to_whole_text = part_image_to_whole_text * image_temp[:, None] * global_local_scale
    part_text_to_whole_image = part_text_to_whole_image * text_temp[:, None] * global_local_scale
    global_local_contrastive = (
        row_loss_scale
        * 0.5
        * (
            _cross_entropy(part_image_to_whole_text, part_targets)
            + _cross_entropy(part_text_to_whole_image, part_targets)
        )
    )

    contrastive = global_contrastive + local_contrastive + global_local_contrastive
    return {
        "contrastive": contrastive,
        "contrastive_global": global_contrastive,
        "contrastive_local": local_contrastive,
        "contrastive_global_local": global_local_contrastive,
    }
