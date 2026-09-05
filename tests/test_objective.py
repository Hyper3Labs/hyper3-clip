"""Unit tests for the paper objective."""

from __future__ import annotations

import pytest
import torch

from hyper3_clip.geometry.lorentz import cone_violation, exp_map0
from hyper3_clip.models.contrastive import embedding_uncertainty
from hyper3_clip.models.entailment import calibration_rows
from hyper3_clip.models.objective import Hyper3CLIPObjective, ObjectiveConfig, compute_objective

from ._inputs import BATCH, PARTS, QUERIES, build_nodes

CONE = ObjectiveConfig()

# queries 1, 3, 5 are non-root with a positive weight; query 7 is non-root but
# carries weight 0, and 0/2/4/6 are roots.  See ``tests/_inputs.py``.
CONTRIBUTING_QUERIES = 3


def _loss(config: ObjectiveConfig, seed: int = 0, **overrides) -> dict:
    return compute_objective({**build_nodes(seed), **overrides}, config)


def test_defaults_are_the_paper_recipe() -> None:
    assert CONE.entail_weight == pytest.approx(0.2)
    assert CONE.inter_aperture_scale == pytest.approx(0.7)
    assert CONE.intra_aperture_scale == pytest.approx(1.2)
    assert CONE.calibration_alpha == pytest.approx(10.0)
    assert CONE.stop_grad_calibration is True
    assert CONE.reverse_visual_order is False


def test_total_is_contrastive_plus_weighted_entailment() -> None:
    out = _loss(CONE)
    expected = out["contrastive"] + CONE.entail_weight * (out["entailment_base"] + out["entailment_query"])
    assert torch.allclose(out["loss"], expected, atol=1e-6)
    assert torch.allclose(out["entailment"], out["entailment_base"] + out["entailment_query"], atol=1e-6)
    # the paper objective has no norm regularizer


def test_reported_counts() -> None:
    out = _loss(CONE)
    assert int(out["part_count"]) == PARTS
    assert int(out["query_count"]) == CONTRIBUTING_QUERIES


def test_base_is_the_plain_sum_of_its_four_relations_plus_calibration() -> None:
    out = _loss(CONE)
    total = (
        out["entailment_image_text"]
        + out["entailment_part_image_text"]
        + out["entailment_image_part_image"]
        + out["entailment_text_part_text"]
        + out["calibration_image_part_image"]
        + out["calibration_text_part_text"]
    )
    assert torch.allclose(out["entailment_base"], total, atol=1e-6)


def test_query_is_the_plain_sum_of_its_three_relations_plus_calibration() -> None:
    out = _loss(CONE)
    total = (
        out["entailment_query_image_text"]
        + out["entailment_query_visual"]
        + out["entailment_query_text"]
        + out["calibration_query_visual"]
        + out["calibration_query_text"]
    )
    assert torch.allclose(out["entailment_query"], total, atol=1e-6)


def test_gradients_are_finite_for_every_input() -> None:
    nodes = build_nodes(0)
    tracked = {
        key: nodes[key].clone().requires_grad_(True)
        for key in ("image_feats", "text_feats", "part_image_flat", "part_text_flat",
                    "query_image_feats", "query_text_feats")
    }
    out = compute_objective({**nodes, **tracked}, CONE)
    out["loss"].backward()
    for key, tensor in tracked.items():
        assert tensor.grad is not None, key
        assert torch.isfinite(tensor.grad).all(), key
        assert float(tensor.grad.abs().sum()) > 0.0, key


def test_relation_is_zero_inside_the_cone_and_positive_outside() -> None:
    """``E(a <= b) = 0`` when ``a`` sits inside ``b``'s cone, ``> 0`` outside."""
    kappa = torch.tensor(1.0)
    general = exp_map0(torch.tensor([[1.0, 0.0]]), kappa)
    inside = exp_map0(torch.tensor([[4.0, 0.0]]), kappa)
    outside = exp_map0(torch.tensor([[0.0, 4.0]]), kappa)
    assert float(cone_violation(inside, general, kappa, CONE.inter_aperture_scale)) == 0.0
    assert float(cone_violation(outside, general, kappa, CONE.inter_aperture_scale)) > 0.0


def test_aperture_scales_change_the_loss() -> None:
    """The cone apertures are actually used."""
    base = _loss(CONE)
    wider_inter = _loss(ObjectiveConfig(inter_aperture_scale=2.0))
    wider_intra = _loss(ObjectiveConfig(intra_aperture_scale=3.0))
    assert not torch.allclose(base["entailment_base"], wider_inter["entailment_base"], atol=1e-6)
    assert not torch.allclose(base["entailment_base"], wider_intra["entailment_base"], atol=1e-6)
    assert not torch.allclose(base["entailment_query"], wider_inter["entailment_query"], atol=1e-6)
    # a wider cone can only shrink the violations
    assert float(wider_inter["entailment_image_text"]) <= float(base["entailment_image_text"]) + 1e-6


@pytest.mark.parametrize(
    ("field", "component"),
    [
        ("query_image_text_weight", "entailment_query_image_text"),
        ("visual_hierarchy_weight", "entailment_query_visual"),
        ("text_hierarchy_weight", "entailment_query_text"),
    ],
)
def test_query_switches_zero_their_terms(field: str, component: str) -> None:
    enabled = _loss(CONE)
    disabled = _loss(ObjectiveConfig(**{field: 0.0}))
    assert float(enabled[component]) != 0.0
    assert float(disabled[component]) == 0.0
    assert float(disabled["entailment_query"]) < float(enabled["entailment_query"])


def test_disabling_a_hierarchy_switch_also_zeros_its_calibration() -> None:
    disabled = _loss(ObjectiveConfig(visual_hierarchy_weight=0.0, text_hierarchy_weight=0.0))
    assert float(disabled["calibration_query_visual"]) == 0.0
    assert float(disabled["calibration_query_text"]) == 0.0


def test_reverse_visual_order_swaps_the_relation_arguments() -> None:
    """``reverse_visual_order`` turns ``E(I <= I^q)`` into ``E(I^q <= I)``."""
    forward = _loss(CONE)
    reverse = _loss(ObjectiveConfig(reverse_visual_order=True))
    assert not torch.allclose(forward["entailment_query_visual"], reverse["entailment_query_visual"], atol=1e-6)
    # only the visual relation moves
    assert torch.allclose(forward["entailment_query_text"], reverse["entailment_query_text"], atol=1e-6)
    assert torch.allclose(forward["entailment_query_image_text"], reverse["entailment_query_image_text"], atol=1e-6)
    assert torch.allclose(forward["entailment_base"], reverse["entailment_base"], atol=1e-6)


def test_root_queries_are_excluded() -> None:
    """Making every query a root empties the query term."""
    nodes = build_nodes(0)
    roots = torch.full((QUERIES,), -1, dtype=torch.long)
    out = compute_objective({**nodes, "query_parent": roots}, CONE)
    assert int(out["query_count"]) == 0
    assert float(out["entailment_query"]) == 0.0


def test_zero_weight_queries_are_excluded() -> None:
    """Query 7 is non-root but weightless; giving it a weight adds a contributor."""
    nodes = build_nodes(0)
    base = compute_objective(nodes, CONE)
    weight = nodes["query_weight"].clone()
    weight[7] = 1.0
    boosted = compute_objective({**nodes, "query_weight": weight}, CONE)
    assert int(base["query_count"]) == CONTRIBUTING_QUERIES
    assert int(boosted["query_count"]) == CONTRIBUTING_QUERIES + 1
    assert not torch.allclose(base["entailment_query"], boosted["entailment_query"], atol=1e-6)

    # changing a weightless query's features leaves the loss untouched
    query_image = nodes["query_image_feats"].clone()
    query_image[7] = query_image[0]
    unchanged = compute_objective({**nodes, "query_image_feats": query_image}, CONE)
    assert torch.allclose(base["entailment_query"], unchanged["entailment_query"], atol=1e-6)


def test_query_weights_scale_their_relations() -> None:
    """A contributing query's ``w(q)`` multiplies its three relations."""
    nodes = build_nodes(0)
    base = compute_objective(nodes, CONE)
    weight = nodes["query_weight"].clone()
    weight[1] *= 2.0
    scaled = compute_objective({**nodes, "query_weight": weight}, CONE)
    assert float(scaled["entailment_query_image_text"]) > float(base["entailment_query_image_text"])
    assert int(scaled["query_count"]) == int(base["query_count"])


def test_calibration_is_always_on_for_the_within_modality_relations() -> None:
    out = _loss(CONE)
    for key in (
        "calibration_image_part_image",
        "calibration_text_part_text",
        "calibration_query_visual",
        "calibration_query_text",
    ):
        assert float(out[key]) != 0.0


def test_stop_grad_calibration_does_not_change_the_entailment_gradient() -> None:
    """With the default stop-grad, calibration trains radii, not the residual."""
    nodes = build_nodes(0)
    grads = {}
    for stop_grad in (True, False):
        probe = nodes["query_image_feats"].clone().requires_grad_(True)
        out = compute_objective(
            {**nodes, "query_image_feats": probe},
            ObjectiveConfig(stop_grad_calibration=stop_grad),
        )
        out["entailment_query"].backward()
        grads[stop_grad] = probe.grad.clone()
    assert not torch.allclose(grads[True], grads[False], atol=1e-6)


def test_contrastive_is_the_sum_of_its_three_groups() -> None:
    out = _loss(CONE)
    total = out["contrastive_global"] + out["contrastive_local"] + out["contrastive_global_local"]
    assert torch.allclose(out["contrastive"], total, atol=1e-6)


def test_missing_nodes_raise() -> None:
    nodes = build_nodes(0)
    for key in ("image_feats", "query_parent", "kappa", "logit_scales"):
        broken = {k: v for k, v in nodes.items() if k != key}
        with pytest.raises(ValueError, match=key if key != "kappa" else "curv"):
            compute_objective(broken, CONE)


def test_empty_parts_and_queries_still_produce_a_finite_loss() -> None:
    nodes = build_nodes(0)
    empty_parts = nodes["part_image_flat"].narrow(0, 0, 0)
    empty_queries = nodes["query_image_feats"].narrow(0, 0, 0)
    stripped = {
        **nodes,
        "part_image_flat": empty_parts,
        "part_text_flat": empty_parts,
        "part_owner": nodes["part_owner"].narrow(0, 0, 0),
        "query_image_feats": empty_queries,
        "query_text_feats": empty_queries,
        "query_owner": nodes["query_owner"].narrow(0, 0, 0),
        "query_parent": nodes["query_parent"].narrow(0, 0, 0),
        "query_weight": nodes["query_weight"].narrow(0, 0, 0),
    }
    for config in (CONE,):
        out = compute_objective(stripped, config)
        assert torch.isfinite(out["loss"])
        assert int(out["part_count"]) == 0
        assert int(out["query_count"]) == 0


def test_curv_is_accepted_as_an_alias_for_kappa() -> None:
    nodes = build_nodes(0)
    renamed = {k: v for k, v in nodes.items() if k != "kappa"}
    renamed["curv"] = nodes["kappa"]
    assert torch.allclose(compute_objective(renamed, CONE)["loss"], _loss(CONE)["loss"], atol=1e-6)


def test_callable_wrapper_binds_its_config() -> None:
    objective = Hyper3CLIPObjective(ObjectiveConfig(entail_weight=0.0))
    out = objective(build_nodes(0))
    assert torch.allclose(out["loss"], out["contrastive"], atol=1e-6)
    assert "ObjectiveConfig" in repr(objective)


def test_batch_shapes_are_respected() -> None:
    nodes = build_nodes(0)
    assert nodes["image_feats"].shape[0] == BATCH
    assert nodes["query_image_feats"].shape[0] == QUERIES
    out = compute_objective(nodes, CONE)
    for value in out.values():
        assert value.ndim == 0


def test_calibration_drops_the_inherited_entropy_constant() -> None:
    """The calibration is the inherited UNCHA one minus its entropy term.

    The inherited implementation adds a batch-level entropy
    ``H = -sum softmax(u) log softmax(u)`` to every row inside
    ``alpha * (term + H)``.  That constant has no counterpart in the paper and
    makes the loss depend on batch composition, so it is dropped here;
    everything else about the mechanism is identical.
    """
    nodes = build_nodes(0)
    residual = torch.rand(nodes["part_image_flat"].size(0))
    log_uncertainty = embedding_uncertainty(nodes["part_image_flat"])
    alpha = 10.0

    ours = calibration_rows(residual, log_uncertainty, alpha=alpha, stop_grad=True)

    sigma = torch.exp(log_uncertainty).clamp(min=1e-6, max=1e6)
    term = 0.5 * residual.detach() / (sigma + 1e-6) + 0.5 * log_uncertainty
    probability = torch.softmax(log_uncertainty.flatten(), dim=0)
    entropy = -(probability * torch.log(probability + 1e-8)).sum()
    reference = alpha * (term + entropy)

    assert torch.allclose(reference, ours + alpha * entropy, rtol=0.0, atol=1e-5)
    assert float(entropy) > 0.0
    assert not torch.allclose(reference, ours, rtol=0.0, atol=1e-6)
