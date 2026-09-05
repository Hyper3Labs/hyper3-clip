"""Hyperbolic Lorentz-model utilities for Hyper3-CLIP.

Shapes are ``[N, d+1] = (x_0, x_bar)`` with ``x_bar``
in ``R^d``; the hyperboloid is embedded in Minkowski space with inner product
``<x,y>_L = -x_0 y_0 + <x_bar, y_bar>`` and curvature ``-c`` where ``c = kappa``.

Direction convention
--------------------
Every entailment helper distinguishes a *specific* node ``a`` and a *general*
node ``b``.  In the relation ``E(a <= b)`` the general node ``b`` roots the
cone and ``a`` is penalized when it falls outside that cone: the first
argument ``a`` is the specific candidate, the second argument ``b`` is the
general, cone-rooting node.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

__all__ = [
    "lorentz_inner",
    "pairwise_lorentz_inner",
    "exp_map0",
    "pairwise_dist",
    "paired_dist",
    "lorentz_distance",
    "metric_pairwise_dist",
    "metric_similarity",
    "space_norm",
    "time_component",
    "half_aperture",
    "exterior_angle",
    "cone_violation",
    "entailment_score",
]


def _f32(x: Tensor, y: Tensor | None = None) -> Tensor | tuple[Tensor, Tensor]:
    xf = x.float()
    if y is None:
        return xf
    return xf, y.float()


def lorentz_inner(x: Tensor, y: Tensor) -> Tensor:
    """Batched Lorentzian inner product ``<x,y>_L = -x_0 y_0 + <x_bar,y_bar>``.

    Computed for matching rows; see :func:`pairwise_lorentz_inner` for all-pairs.
    """
    with torch.autocast(device_type=x.device.type, enabled=False):
        x, y = _f32(x, y)
        return -x[..., 0] * y[..., 0] + (x[..., 1:] * y[..., 1:]).sum(dim=-1)


def pairwise_lorentz_inner(x: Tensor, y: Tensor) -> Tensor:
    """All-pairs Lorentzian inner products ``x @_L y.T``."""
    with torch.autocast(device_type=x.device.type, enabled=False):
        x, y = _f32(x, y)
        time = -x[:, :1] @ y[:, :1].T
        space = x[:, 1:] @ y[:, 1:].T
        return time + space


def exp_map0(u: Tensor, kappa: Tensor, eps: float = 1e-8) -> Tensor:
    """Exponential map at the origin from tangent space to the hyperboloid.

    ``exp_map0`` lifts a Euclidean tangent vector ``u in R^d`` onto the
    hyperboloid shell with curvature ``-kappa``.  This is the lift used to
    produce all manifold nodes from CLIP-style Euclidean embeddings in the
    training forward pass.
    """
    with torch.autocast(device_type=u.device.type, enabled=False):
        u, kappa = _f32(u, kappa)
        sqrt_k = torch.sqrt(kappa)
        norm_u = torch.linalg.norm(u, dim=-1, keepdim=True).clamp_min(eps)
        scaled = sqrt_k * norm_u
        clipped_scaled = scaled.clamp_max(math.asinh(2**15))
        time = torch.cosh(clipped_scaled) / sqrt_k
        space = torch.sinh(clipped_scaled) * u / scaled.clamp_min(eps)
        return torch.cat([time, space], dim=-1)


def space_norm(x: Tensor, eps: float = 1e-8) -> Tensor:
    """Space-component norm ``||x_bar||`` (the radius on the hyperboloid)."""
    with torch.autocast(device_type=x.device.type, enabled=False):
        x = x.float()
        return torch.linalg.norm(x[..., 1:], dim=-1).clamp_min(eps)


def time_component(x: Tensor) -> Tensor:
    """Time component ``x_0`` of ``[N, d+1]`` nodes."""
    with torch.autocast(device_type=x.device.type, enabled=False):
        return x.float()[..., 0]


def pairwise_dist(x: Tensor, y: Tensor, kappa: Tensor, eps: float = 1e-8) -> Tensor:
    """Pairwise geodesic distance on the Lorentz model."""
    with torch.autocast(device_type=x.device.type, enabled=False):
        x, y, kappa = x.float(), y.float(), kappa.float()
        dist_eps = max(eps, 16.0 * torch.finfo(x.dtype).eps)
        prod = (-kappa) * pairwise_lorentz_inner(x, y)
        prod = prod.clamp_min(1.0 + dist_eps)
        return torch.acosh(prod) / torch.sqrt(kappa)


def paired_dist(x: Tensor, y: Tensor, kappa: Tensor, eps: float = 1e-8) -> Tensor:
    """Paired (row-wise) geodesic distance; ``x`` and ``y`` must share shape."""
    with torch.autocast(device_type=x.device.type, enabled=False):
        x, y, kappa = x.float(), y.float(), kappa.float()
        dist_eps = max(eps, 16.0 * torch.finfo(x.dtype).eps)
        prod = (-kappa) * lorentz_inner(x, y)
        prod = prod.clamp_min(1.0 + dist_eps)
        return torch.acosh(prod) / torch.sqrt(kappa)


def lorentz_distance(x: Tensor, y: Tensor, kappa: Tensor, eps: float = 1e-8) -> Tensor:
    """Alias for :func:`paired_dist` — row-wise Lorentz geodesic distance.

    Provided under the name used across the objective module for
    ``d_L(x, y)``.
    """
    return paired_dist(x, y, kappa, eps=eps)


def metric_pairwise_dist(x: Tensor, y: Tensor, kappa: Tensor) -> Tensor:
    """Convenience alias for :func:`pairwise_dist`."""
    return pairwise_dist(x, y, kappa)


def metric_similarity(x: Tensor, y: Tensor, kappa: Tensor) -> Tensor:
    """Retrieval similarity, the Lorentz inner product (the paper's eval metric)."""
    return pairwise_lorentz_inner(x, y)


def half_aperture(general: Tensor, kappa: Tensor, min_radius: float = 0.1, eps: float = 1e-8) -> Tensor:
    """Cone half-aperture ``Theta(b)`` for the entailment cone at ``b``.

    ``b`` is the general / cone-rooting node.  A larger ``min_radius`` widens
    the aperture (the cone tolerates more deviation); it is fixed at ``0.1``.
    """
    with torch.autocast(device_type=general.device.type, enabled=False):
        general, kappa = _f32(general, kappa)
        aperture_eps = max(eps, 16.0 * torch.finfo(general.dtype).eps)
        general_norm = torch.linalg.norm(general[:, 1:], dim=-1)
        ratio = (2.0 * min_radius) / (general_norm * torch.sqrt(kappa) + aperture_eps)
        ratio = ratio.clamp(max=1.0 - aperture_eps)
        return torch.asin(ratio)


def exterior_angle(a: Tensor, b: Tensor, kappa: Tensor, eps: float = 1e-8) -> Tensor:
    """Exterior angle ``O(a, b)`` between ``a`` and the cone rooted at ``b``.

    ``b`` roots the cone (the general node); ``a`` is the specific node being
    scored against it.  The angle measures how far ``a`` lies outside the cone:
    ``O = 0`` when ``a`` sits on the cone boundary inside/on it.
    """
    with torch.autocast(device_type=a.device.type, enabled=False):
        a, b, kappa = a.float(), b.float(), kappa.float()
        angle_eps = max(eps, 16.0 * torch.finfo(a.dtype).eps)
        inner = lorentz_inner(a, b)
        numerator = a[:, 0] + kappa * inner * b[:, 0]
        general_norm = torch.linalg.norm(b[:, 1:], dim=-1).clamp_min(angle_eps)
        denom_term = (kappa * inner).pow(2) - 1.0
        denom = general_norm * torch.sqrt(denom_term.clamp_min(angle_eps))
        cosine = (numerator / denom).clamp(min=-1.0 + angle_eps, max=1.0 - angle_eps)
        return torch.acos(cosine)


def cone_violation(a: Tensor, b: Tensor, kappa: Tensor, aperture_scale: float, *, eps: float = 1e-8) -> Tensor:
    """Cone violation ``E_hinge(a <= b) = clamp(O(a,b) - s*Theta(b), min=0)``.

    ``b`` roots the cone; ``a`` is penalized (positive) only when its exterior
    angle exceeds ``aperture_scale`` times the cone half-aperture at ``b``.
    This is the hinge used by the paper's entailment objective.
    """
    angle = exterior_angle(a, b, kappa, eps=eps)
    aperture = aperture_scale * half_aperture(b, kappa, eps=eps)
    return torch.clamp(angle - aperture, min=0.0)


def entailment_score(a: Tensor, b: Tensor, kappa: Tensor, *, eps: float = 1e-8) -> Tensor:
    """Entailment score ``clamp(1 - 2*O(a,b)/pi, min=0)``.

    Used by the hierarchy evaluation: ``a`` is deemed to entail ``b`` the more
    it lies inside ``b``'s cone.  ``b`` is the general / cone-rooting node.
    """
    angle = exterior_angle(a, b, kappa, eps=eps)
    return torch.clamp(1.0 - (2.0 * angle / math.pi), min=0.0, max=1.0)
