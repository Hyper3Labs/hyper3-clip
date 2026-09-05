"""Entailment-cone terms of the Hyper3-CLIP objective (paper Sec. 3.3).

This module implements the objective **as the paper states it**: every relation
``a <= b`` is scored by the MERU/HyCoCLIP entailment-cone violation

``E(a <= b) = clamp( phi(a, b) - s * Theta(b), min=0 )``

with ``phi`` the exterior angle of ``a`` w.r.t. the cone rooted at the *general*
node ``b``, ``Theta(b)`` that cone's half-aperture, and ``s`` the aperture scale
(``inter_aperture_scale`` for cross-modal relations, ``intra_aperture_scale``
for within-modality part-whole / hierarchy relations).  Relations are summed
with unit weights.

Two groups of relations are computed:

* the four **inherited** relations ``I <= T``, ``I^box <= T^box``,
  ``I <= I^box``, ``T <= T^box`` (:func:`base_entailment`);
* for every **non-root** query ``q`` with ``w(q) > 0``, the three **query**
  relations ``I^q <= T^q``, ``I <= I^q`` and ``T^pi(q) <= T^q``
  (:func:`query_entailment`).

Both groups carry the inherited UNCHA *uncertainty calibration*, which is
always on: a radius-derived log-uncertainty ``u = softplus(-||x_bar||)`` of the
relation's **general** node turns the residual into a Gaussian negative
log-likelihood ``alpha * (0.5 * r_hat / sigma + 0.5 * u)``, with
``sigma = exp(u)`` and ``r_hat`` detached.  The paper does not say which node's
radius defines the uncertainty or on which relations it acts; both follow the
inherited UNCHA objective, which calibrates exactly the within-modality
relations (whose general node is the part or query node).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from hyper3_clip.geometry.lorentz import cone_violation
from hyper3_clip.models.contrastive import embedding_uncertainty

__all__ = [
    "QueryHierarchy",
    "base_entailment",
    "calibrated_entailment",
    "calibration_rows",
    "query_entailment",
    "resolve_query_hierarchy",
]


def calibration_rows(
    residual: Tensor,
    log_uncertainty: Tensor,
    *,
    alpha: float = 10.0,
    stop_grad: bool = True,
) -> Tensor:
    """Per-row calibration loss ``alpha * (0.5 * r_hat / sigma + 0.5 * u)``.

    A Gaussian negative log-likelihood that trains the embedding radius to
    predict how badly the relation is violated (``sigma = exp(u)``).  With
    ``stop_grad=True`` (the default and the paper setting) ``r_hat`` is
    detached, so calibration shapes the radii only and never rescales the
    entailment gradient.

    The inherited UNCHA implementation also adds the batch-level entropy of
    ``softmax(u)`` to every row.  That constant has no counterpart in the paper
    and makes the loss depend on batch composition, so it is dropped here.
    """
    sigma = torch.exp(log_uncertainty).clamp(min=1e-6, max=1e6)
    detached = residual.detach() if stop_grad else residual
    return alpha * (0.5 * detached / (sigma + 1e-6) + 0.5 * log_uncertainty)


def calibrated_entailment(
    residual: Tensor,
    log_uncertainty: Tensor,
    *,
    alpha: float = 10.0,
    stop_grad: bool = True,
) -> tuple[Tensor, Tensor]:
    """``(mean(residual), mean(calibration_rows(...)))``."""
    calibration = calibration_rows(residual, log_uncertainty, alpha=alpha, stop_grad=stop_grad)
    return residual.mean(), calibration.mean()


def base_entailment(
    image_feats: Tensor,
    text_feats: Tensor,
    part_image_flat: Tensor,
    part_text_flat: Tensor,
    image_for_parts: Tensor,
    text_for_parts: Tensor,
    kappa: Tensor,
    *,
    inter_aperture_scale: float,
    intra_aperture_scale: float,
    calibration_alpha: float = 10.0,
    stop_grad_calibration: bool = True,
) -> dict[str, Tensor]:
    """The four inherited relations, summed with unit weights.

    ``E(I <= T)`` and ``E(I^box <= T^box)`` are cross-modal and use
    ``inter_aperture_scale``; ``E(I <= I^box)`` and ``E(T <= T^box)`` are
    within-modality and use ``intra_aperture_scale``.  The two within-modality
    relations are calibrated by the uncertainty of their general node (the box
    crop / box phrase).
    """
    zero = image_feats.new_zeros(())
    image_text = cone_violation(image_feats, text_feats, kappa, inter_aperture_scale).mean()

    if part_image_flat.numel() == 0:
        return {
            "entailment_base": image_text,
            "entailment_image_text": image_text,
            "entailment_part_image_text": zero,
            "entailment_image_part_image": zero,
            "entailment_text_part_text": zero,
            "calibration_image_part_image": zero,
            "calibration_text_part_text": zero,
        }

    part_image_text = cone_violation(part_image_flat, part_text_flat, kappa, inter_aperture_scale)
    image_part_image = cone_violation(image_for_parts, part_image_flat, kappa, intra_aperture_scale)
    text_part_text = cone_violation(text_for_parts, part_text_flat, kappa, intra_aperture_scale)

    part_image_text_loss = part_image_text.mean()
    image_part_loss, image_part_calibration = calibrated_entailment(
        image_part_image,
        embedding_uncertainty(part_image_flat),
        alpha=calibration_alpha,
        stop_grad=stop_grad_calibration,
    )
    text_part_loss, text_part_calibration = calibrated_entailment(
        text_part_text,
        embedding_uncertainty(part_text_flat),
        alpha=calibration_alpha,
        stop_grad=stop_grad_calibration,
    )

    total = (
        image_text
        + part_image_text_loss
        + image_part_loss
        + text_part_loss
        + image_part_calibration
        + text_part_calibration
    )
    return {
        "entailment_base": total,
        "entailment_image_text": image_text,
        "entailment_part_image_text": part_image_text_loss,
        "entailment_image_part_image": image_part_loss,
        "entailment_text_part_text": text_part_loss,
        "calibration_image_part_image": image_part_calibration,
        "calibration_text_part_text": text_part_calibration,
    }


@dataclass(frozen=True)
class QueryHierarchy:
    """The subset of queries that contribute to the query entailment term.

    ``mask`` selects the **non-root** queries (``parent >= 0``) with a valid
    parent index and ``w(q) > 0`` — exactly the population the paper's Sec. 3.3
    sum runs over.  ``weight`` holds ``w(q)`` for the selected rows and
    ``count`` their number, which normalizes the term.
    """

    mask: Tensor
    parent: Tensor
    owner: Tensor
    weight: Tensor
    count: int


def resolve_query_hierarchy(
    query_parent: Tensor,
    query_owner: Tensor,
    query_weight: Tensor,
    num_queries: int,
    device: torch.device,
) -> QueryHierarchy:
    """Select non-root, positively weighted queries with in-range parents."""
    parent = query_parent.to(device=device, dtype=torch.long)
    owner = query_owner.to(device=device, dtype=torch.long)
    weight = query_weight.to(device=device, dtype=torch.float32).clamp_min(0.0)
    if weight.numel() != num_queries:
        raise ValueError("query_weight must have one value per query")
    mask = (parent >= 0) & (parent < num_queries) & (weight > 0.0)
    return QueryHierarchy(
        mask=mask,
        parent=parent,
        owner=owner,
        weight=weight,
        count=int(mask.sum().item()),
    )


def _query_reduce(residual: Tensor, weight: Tensor, count: int) -> Tensor:
    """``sum_q w(q) * residual_q / count`` — the paper's weighted sum."""
    return (residual * weight).sum() / max(count, 1)


def query_entailment(
    image_feats: Tensor,
    query_image_feats: Tensor,
    query_text_feats: Tensor,
    hierarchy: QueryHierarchy,
    kappa: Tensor,
    *,
    inter_aperture_scale: float,
    intra_aperture_scale: float,
    calibration_alpha: float = 10.0,
    stop_grad_calibration: bool = True,
    query_image_text_weight: float = 1.0,
    visual_hierarchy_weight: float = 1.0,
    text_hierarchy_weight: float = 1.0,
    reverse_visual_order: bool = False,
) -> dict[str, Tensor]:
    """The three query relations, summed over non-root queries with ``w(q) > 0``.

    Relation (1) ``E(I^q <= T^q)`` is cross-modal (``inter_aperture_scale``);
    relations (2) ``E(I <= I^q)`` and (3) ``E(T^pi(q) <= T^q)`` are
    within-modality (``intra_aperture_scale``) and are calibrated by the
    uncertainty of their general node.  ``reverse_visual_order`` swaps relation
    (2) to ``E(I^q <= I)`` — the paper's direction control — in which case the
    calibrating node becomes the whole image, still the general node.

    Each term is ``sum_q w(q) * E_q`` divided by the number of contributing
    queries.
    """
    zero = image_feats.new_zeros(())
    keys = (
        "entailment_query",
        "entailment_query_image_text",
        "entailment_query_visual",
        "entailment_query_text",
        "calibration_query_visual",
        "calibration_query_text",
    )
    if hierarchy.count == 0:
        # Stay connected to the query graph so DDP sees the pooling parameters.
        empty = query_image_feats.sum() * 0.0 + query_text_feats.sum() * 0.0
        return dict.fromkeys(keys, zero + empty)

    mask = hierarchy.mask
    weight = hierarchy.weight[mask]
    count = hierarchy.count

    child_image = query_image_feats[mask]
    child_text = query_text_feats[mask]
    owner_image = image_feats.index_select(0, hierarchy.owner[mask])
    parent_text = query_text_feats[hierarchy.parent[mask]]

    image_text = _query_reduce(
        cone_violation(child_image, child_text, kappa, inter_aperture_scale), weight, count
    )

    if reverse_visual_order:
        visual_specific, visual_general = child_image, owner_image
    else:
        visual_specific, visual_general = owner_image, child_image
    visual_residual = cone_violation(visual_specific, visual_general, kappa, intra_aperture_scale)
    text_residual = cone_violation(parent_text, child_text, kappa, intra_aperture_scale)

    visual_loss, visual_calibration = _calibrated_query_term(
        visual_residual, visual_general, weight, count, calibration_alpha, stop_grad_calibration
    )
    text_loss, text_calibration = _calibrated_query_term(
        text_residual, child_text, weight, count, calibration_alpha, stop_grad_calibration
    )

    image_text = float(query_image_text_weight) * image_text
    visual_loss = float(visual_hierarchy_weight) * visual_loss
    text_loss = float(text_hierarchy_weight) * text_loss
    visual_calibration = float(visual_hierarchy_weight) * visual_calibration
    text_calibration = float(text_hierarchy_weight) * text_calibration

    return {
        "entailment_query": image_text + visual_loss + text_loss + visual_calibration + text_calibration,
        "entailment_query_image_text": image_text,
        "entailment_query_visual": visual_loss,
        "entailment_query_text": text_loss,
        "calibration_query_visual": visual_calibration,
        "calibration_query_text": text_calibration,
    }


def _calibrated_query_term(
    residual: Tensor,
    general: Tensor,
    weight: Tensor,
    count: int,
    alpha: float,
    stop_grad: bool,
) -> tuple[Tensor, Tensor]:
    calibration = calibration_rows(
        residual, embedding_uncertainty(general), alpha=alpha, stop_grad=stop_grad
    )
    return _query_reduce(residual, weight, count), _query_reduce(calibration, weight, count)
