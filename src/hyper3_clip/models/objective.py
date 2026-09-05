"""The Hyper3-CLIP training objective (paper Sec. 3.3).

``L = L_con + lambda_ent * (L_ent + L_query)``

* ``L_con`` is the inherited UNCHA contrastive term
  (:mod:`hyper3_clip.models.contrastive`);
* ``L_ent`` sums the four inherited entailment relations ``I <= T``,
  ``I^box <= T^box``, ``I <= I^box``, ``T <= T^box``;
* ``L_query`` sums, over every non-root query ``q`` with ``w(q) > 0``, the three
  query relations ``I^q <= T^q``, ``I <= I^q`` and ``T^pi(q) <= T^q``, weighted
  by ``w(q)``.

Each ``E(a <= b)`` is the entailment-cone violation
``clamp(phi(a, b) - s * Theta(b), 0)`` (:mod:`hyper3_clip.models.entailment`).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from hyper3_clip.models.contrastive import contrastive_losses
from hyper3_clip.models.entailment import base_entailment, query_entailment, resolve_query_hierarchy
from hyper3_clip.models.reduction import global_mean_scale

__all__ = ["ObjectiveConfig", "compute_objective", "Hyper3CLIPObjective"]


@dataclass(frozen=True)
class ObjectiveConfig:
    """Configuration of the training objective; the defaults are the paper's.

    ============================  ============================================
    field                         paper symbol / role
    ============================  ============================================
    ``entail_weight``             ``lambda_ent``
    ``inter_aperture_scale``      ``s`` for cross-modal relations
    ``intra_aperture_scale``      ``s`` for within-modality relations
    ``calibration_alpha``         ``alpha`` of the uncertainty calibration
    ``stop_grad_calibration``     detach ``r`` inside the calibration term
    ``query_image_text_weight``   weight of query relation (1) ``I^q <= T^q``
    ``visual_hierarchy_weight``   weight of query relation (2) ``I <= I^q``
    ``text_hierarchy_weight``     weight of query relation (3)
                                  ``T^pi(q) <= T^q``
    ``reverse_visual_order``      the ``I^q <= I`` direction control
    ============================  ============================================

    ``calibration_alpha = 10.0`` and ``stop_grad_calibration = True`` are the
    inherited UNCHA constants; the paper does not vary them.  The three query
    weights and ``reverse_visual_order`` are the controls of the objective
    ablation (Table 5).
    """

    entail_weight: float = 0.2
    inter_aperture_scale: float = 0.7
    intra_aperture_scale: float = 1.2
    calibration_alpha: float = 10.0
    stop_grad_calibration: bool = True
    query_image_text_weight: float = 1.0
    visual_hierarchy_weight: float = 1.0
    text_hierarchy_weight: float = 1.0
    reverse_visual_order: bool = False


_COMPONENT_KEYS = (
    "entailment_image_text",
    "entailment_part_image_text",
    "entailment_image_part_image",
    "entailment_text_part_text",
    "calibration_image_part_image",
    "calibration_text_part_text",
    "entailment_query_image_text",
    "entailment_query_visual",
    "entailment_query_text",
    "calibration_query_visual",
    "calibration_query_text",
)


def _require(nodes: Mapping[str, Tensor], key: str) -> Tensor:
    value = nodes.get(key)
    if value is None:
        raise ValueError(f"compute_objective requires node {key!r}")
    return value


def compute_objective(nodes: Mapping[str, Tensor], config: ObjectiveConfig | None = None) -> dict[str, Tensor]:
    """Evaluate the objective on one batch of Lorentz nodes.

    Required ``nodes`` entries (all manifold tensors are ``[N, d + 1]`` points
    on the hyperboloid, i.e. ``exp_map0`` outputs):

    ``image_feats`` ``[B, d+1]``, ``text_feats`` ``[B, d+1]``,
    ``part_image_flat`` ``[P, d+1]``, ``part_text_flat`` ``[P, d+1]``,
    ``part_owner`` ``[P]``, ``query_image_feats`` ``[Q, d+1]``,
    ``query_text_feats`` ``[Q, d+1]``, ``query_owner`` ``[Q]``,
    ``query_parent`` ``[Q]`` (``-1`` = root), ``query_weight`` ``[Q]``,
    ``kappa`` (scalar curvature magnitude; ``curv`` is accepted as an alias) and
    ``logit_scales`` = ``{"global", "local", "global_local"}``.

    Optional entries: ``image_for_parts`` / ``text_for_parts`` (derived from
    ``part_owner`` when absent), ``targets`` ``[B]``, ``part_targets`` ``[P]``,
    ``entail_weight_scale`` (entailment warm-up ramp), and the DDP-gathered
    negative pools ``all_image_feats``, ``all_text_feats``,
    ``all_part_image_feats``, ``all_part_text_feats``, ``all_image_for_parts``,
    ``all_text_for_parts``.  ``query_source_part`` is carried through the batch
    but never read here.

    Returns ``{"loss", "contrastive", "entailment", "entailment_base",
    "entailment_query", "part_count", "query_count",
    ...}`` plus one scalar per relation (and per calibration term).
    """
    config = ObjectiveConfig() if config is None else config

    image_feats = _require(nodes, "image_feats")
    text_feats = _require(nodes, "text_feats")
    part_image_flat = _require(nodes, "part_image_flat")
    part_text_flat = _require(nodes, "part_text_flat")
    part_owner = _require(nodes, "part_owner").to(device=image_feats.device, dtype=torch.long)
    query_image_feats = _require(nodes, "query_image_feats")
    query_text_feats = _require(nodes, "query_text_feats")
    query_owner = _require(nodes, "query_owner")
    query_parent = _require(nodes, "query_parent")
    query_weight = _require(nodes, "query_weight")
    kappa = nodes.get("kappa")
    if kappa is None:
        kappa = _require(nodes, "curv")
    logit_scales = nodes.get("logit_scales")
    if logit_scales is None:
        raise ValueError("compute_objective requires node 'logit_scales'")

    image_for_parts = nodes.get("image_for_parts")
    text_for_parts = nodes.get("text_for_parts")
    if image_for_parts is None or text_for_parts is None:
        if part_owner.numel() == 0:
            image_for_parts = image_feats.narrow(0, 0, 0)
            text_for_parts = text_feats.narrow(0, 0, 0)
        else:
            image_for_parts = image_feats[part_owner]
            text_for_parts = text_feats[part_owner]

    global_targets = nodes.get("targets")
    part_targets = nodes.get("part_targets")
    row_loss_scale = global_mean_scale(part_image_flat.new_tensor(float(part_image_flat.size(0))))

    contrastive = contrastive_losses(
        image_feats,
        text_feats,
        part_image_flat,
        part_text_flat,
        image_for_parts,
        text_for_parts,
        kappa,
        global_logit_scale=logit_scales["global"],
        local_logit_scale=logit_scales["local"],
        global_local_logit_scale=logit_scales["global_local"],
        all_image_feats=nodes.get("all_image_feats"),
        all_text_feats=nodes.get("all_text_feats"),
        all_part_image_feats=nodes.get("all_part_image_feats"),
        all_part_text_feats=nodes.get("all_part_text_feats"),
        all_image_for_parts=nodes.get("all_image_for_parts"),
        all_text_for_parts=nodes.get("all_text_for_parts"),
        global_targets=global_targets,
        part_targets=part_targets,
        row_loss_scale=row_loss_scale,
    )

    parts = base_entailment(
        image_feats,
        text_feats,
        part_image_flat,
        part_text_flat,
        image_for_parts,
        text_for_parts,
        kappa,
        inter_aperture_scale=config.inter_aperture_scale,
        intra_aperture_scale=config.intra_aperture_scale,
        calibration_alpha=config.calibration_alpha,
        stop_grad_calibration=config.stop_grad_calibration,
    )
    hierarchy = resolve_query_hierarchy(
        query_parent, query_owner, query_weight, query_image_feats.size(0), image_feats.device
    )
    queries = query_entailment(
        image_feats,
        query_image_feats,
        query_text_feats,
        hierarchy,
        kappa,
        inter_aperture_scale=config.inter_aperture_scale,
        intra_aperture_scale=config.intra_aperture_scale,
        calibration_alpha=config.calibration_alpha,
        stop_grad_calibration=config.stop_grad_calibration,
        query_image_text_weight=config.query_image_text_weight,
        visual_hierarchy_weight=config.visual_hierarchy_weight,
        text_hierarchy_weight=config.text_hierarchy_weight,
        reverse_visual_order=config.reverse_visual_order,
    )
    query_count = query_owner.new_tensor(hierarchy.count)

    entail_weight_scale = nodes.get("entail_weight_scale")
    if entail_weight_scale is None:
        entail_weight_scale = image_feats.new_ones(())

    entailment = parts["entailment_base"] + queries["entailment_query"]
    loss = contrastive["contrastive"] + config.entail_weight * entail_weight_scale * entailment

    out: dict[str, Tensor] = {
        "loss": loss,
        **contrastive,
        "entailment": entailment,
        "entailment_base": parts["entailment_base"],
        "entailment_query": queries["entailment_query"],
        "part_count": part_owner.new_tensor(part_owner.numel()),
        "query_count": query_count,
        "entail_weight_scale": entail_weight_scale.detach(),
    }
    for key in _COMPONENT_KEYS:
        out[key] = parts.get(key, queries.get(key, image_feats.new_zeros(())))
    return out


class Hyper3CLIPObjective:
    """Thin callable wrapper binding an :class:`ObjectiveConfig`.

    Deliberately not an ``nn.Module``: the objective holds no parameters (the
    logit scales, curvature and per-modality alphas live on the model and are
    handed in through ``nodes``).
    """

    def __init__(self, config: ObjectiveConfig | None = None) -> None:
        self.config = ObjectiveConfig() if config is None else config

    def __call__(self, nodes: Mapping[str, Tensor]) -> dict[str, Tensor]:
        return compute_objective(nodes, self.config)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.config!r})"
