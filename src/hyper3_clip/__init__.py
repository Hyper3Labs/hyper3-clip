"""Hyper3-CLIP: hierarchy-conditioned hyperbolic vision-language training.

Code release for the ECCV 2026 workshop paper
*Hyper3-CLIP: Hierarchy-Conditioned Hyperbolic Vision-Language Training*.

Sub-packages
------------
``hyper3_clip.geometry``
    Lorentz-model primitives: exponential map, geodesic distance, entailment
    cones (half-aperture, exterior angle, cone violation, entailment score).
``hyper3_clip.data``
    Rule-based query-hierarchy construction (paper Sec. 3.1).
``hyper3_clip.models``
    The towers, query-conditioned pooling (Sec. 3.2), the training objective
    (Sec. 3.3) and the :class:`~hyper3_clip.models.hyper3_clip.Hyper3CLIP`
    model itself, with checkpoint interchange.
``hyper3_clip.training``
    Run configuration, optimizer groups, LR schedule, DDP and the step loop.
``hyper3_clip.evaluation``
    The paper's tables: retrieval, ImageNet zero-shot and WordNet hierarchy,
    ImageFolder classification, multi-label mAP, hierarchy entailment, the
    GRIT parts scan, the suite driver and the reporters (see
    ``docs/evaluation.md``).
``hyper3_clip.distributed``
    Rank-aware gathers shared by the model forward and the training loop.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
