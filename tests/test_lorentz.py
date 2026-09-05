"""Lorentz-model geometry: manifold invariants and entailment-cone behaviour."""

from __future__ import annotations

import math

import pytest
import torch

from hyper3_clip.geometry.lorentz import (
    cone_violation,
    entailment_score,
    exp_map0,
    exterior_angle,
    half_aperture,
    lorentz_distance,
    lorentz_inner,
    metric_similarity,
    pairwise_dist,
    space_norm,
    time_component,
)

KAPPA = torch.tensor(1.0)


def _points(n: int, dim: int = 8, seed: int = 0, scale: float = 1.0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return exp_map0(torch.randn(n, dim, generator=gen) * scale * dim**-0.5, KAPPA)


def test_exp_map0_lands_on_the_hyperboloid() -> None:
    """``<x, x>_L = -1/kappa`` for every point produced by the exponential map."""
    for kappa in (torch.tensor(0.5), torch.tensor(1.0), torch.tensor(2.0)):
        tangent = torch.randn(32, 16, generator=torch.Generator().manual_seed(1)) * 0.25
        points = exp_map0(tangent, kappa)
        assert points.shape == (32, 17)
        inner = lorentz_inner(points, points)
        assert torch.allclose(inner, torch.full_like(inner, -1.0 / float(kappa)), atol=1e-4)


def test_exp_map0_preserves_direction_and_origin() -> None:
    tangent = torch.randn(8, 6, generator=torch.Generator().manual_seed(2))
    points = exp_map0(tangent, KAPPA)
    cosine = torch.nn.functional.cosine_similarity(points[:, 1:], tangent, dim=-1)
    assert torch.allclose(cosine, torch.ones_like(cosine), atol=1e-5)
    origin = exp_map0(torch.zeros(1, 6), KAPPA)
    assert torch.allclose(origin[:, 1:], torch.zeros(1, 6), atol=1e-6)
    assert torch.allclose(origin[:, 0], torch.ones(1), atol=1e-6)


def test_distance_is_a_metric_on_sampled_points() -> None:
    points = _points(6)
    zero = lorentz_distance(points, points, KAPPA)
    # the acosh argument is clamped at 1 + 16*eps, so d(x, x) floors
    # at acosh(1 + 16*eps) ~ 2e-3 rather than at exactly 0
    assert torch.allclose(zero, torch.zeros_like(zero), atol=3e-3)
    forward = lorentz_distance(points[:3], points[3:], KAPPA)
    backward = lorentz_distance(points[3:], points[:3], KAPPA)
    assert torch.allclose(forward, backward, atol=1e-6)
    assert bool((forward > 0).all())


def test_paired_distance_is_the_diagonal_of_the_pairwise_one() -> None:
    left, right = _points(5, seed=3), _points(5, seed=4)
    assert torch.allclose(lorentz_distance(left, right, KAPPA), pairwise_dist(left, right, KAPPA).diagonal(), atol=1e-5)


def test_space_norm_and_time_component() -> None:
    points = _points(4)
    assert torch.allclose(space_norm(points), torch.linalg.norm(points[:, 1:], dim=-1), atol=1e-6)
    assert torch.allclose(time_component(points), points[:, 0])


def test_similarity_is_the_lorentz_inner_product() -> None:
    """The paper's hierarchy evaluation ranks by ``<x, y>_L`` (not by ``-d_L``)."""
    left, right = _points(3, seed=5), _points(4, seed=6)
    similarity = metric_similarity(left, right, KAPPA)
    assert similarity.shape == (3, 4)
    assert torch.allclose(similarity[0], lorentz_inner(left[:1].expand(4, -1), right), atol=1e-5)


def test_half_aperture_shrinks_with_radius() -> None:
    """A specific concept (large radius) roots a narrow cone; a generic one a wide cone."""
    small = exp_map0(torch.tensor([[0.05, 0.0, 0.0]]), KAPPA)
    large = exp_map0(torch.tensor([[3.0, 0.0, 0.0]]), KAPPA)
    assert float(half_aperture(small, KAPPA)) > float(half_aperture(large, KAPPA))
    assert float(half_aperture(small, KAPPA)) <= math.pi / 2 + 1e-6
    # a wider min_radius widens the cone
    assert float(half_aperture(large, KAPPA, min_radius=0.5)) > float(half_aperture(large, KAPPA, min_radius=0.1))


def test_exterior_angle_is_zero_along_the_cone_axis() -> None:
    """A point directly "outward" from ``b`` sits on the cone axis, angle ~ 0."""
    general = exp_map0(torch.tensor([[1.0, 0.0]]), KAPPA)
    axial = exp_map0(torch.tensor([[3.0, 0.0]]), KAPPA)
    opposite = exp_map0(torch.tensor([[-3.0, 0.0]]), KAPPA)
    # floored at acos(1 - 16*eps) ~ 2e-3 by that clamp, not exactly 0
    assert float(exterior_angle(axial, general, KAPPA)) < 3e-3
    assert float(exterior_angle(opposite, general, KAPPA)) > math.pi / 2


def test_cone_violation_is_zero_inside_and_positive_outside() -> None:
    general = exp_map0(torch.tensor([[1.0, 0.0]]), KAPPA)
    inside = exp_map0(torch.tensor([[3.0, 0.0]]), KAPPA)
    outside = exp_map0(torch.tensor([[0.0, 3.0]]), KAPPA)
    assert float(cone_violation(inside, general, KAPPA, 1.0)) == 0.0
    assert float(cone_violation(outside, general, KAPPA, 1.0)) > 0.0


def test_cone_violation_is_monotone_in_the_aperture_scale() -> None:
    """Widening the cone can only reduce the violation."""
    specific, general = _points(8, seed=7), _points(8, seed=8)
    narrow = cone_violation(specific, general, KAPPA, 0.5)
    wide = cone_violation(specific, general, KAPPA, 2.0)
    assert bool((wide <= narrow + 1e-6).all())
    assert float(narrow.sum()) > float(wide.sum())


def test_cone_violation_direction_matters() -> None:
    """``b`` roots the cone: swapping the arguments changes the relation."""
    specific, general = _points(6, seed=9), _points(6, seed=10)
    forward = cone_violation(specific, general, KAPPA, 1.0)
    reverse = cone_violation(general, specific, KAPPA, 1.0)
    assert not torch.allclose(forward, reverse, atol=1e-4)


def test_entailment_score_ranges_and_ordering() -> None:
    specific, general = _points(16, seed=11), _points(16, seed=12)
    score = entailment_score(specific, general, KAPPA)
    angle = exterior_angle(specific, general, KAPPA)
    assert bool(((score >= 0.0) & (score <= 1.0)).all())
    assert torch.allclose(score, torch.clamp(1.0 - 2.0 * angle / math.pi, min=0.0, max=1.0), atol=1e-6)
    # smaller angle => higher score
    order_by_angle = torch.argsort(angle)
    assert bool((score[order_by_angle].diff() <= 1e-6).all())


@pytest.mark.parametrize("function", [exterior_angle, half_aperture])
def test_gradients_are_finite(function) -> None:
    specific = _points(8, seed=13).requires_grad_(True)
    general = _points(8, seed=14).requires_grad_(True)
    value = function(specific, general, KAPPA) if function is exterior_angle else function(general, KAPPA)
    value.sum().backward()
    grads = [general.grad] + ([specific.grad] if specific.grad is not None else [])
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_cone_violation_gradients_are_finite() -> None:
    specific = _points(8, seed=15).requires_grad_(True)
    general = _points(8, seed=16).requires_grad_(True)
    cone_violation(specific, general, KAPPA, 0.7).sum().backward()
    assert torch.isfinite(specific.grad).all()
    assert torch.isfinite(general.grad).all()
