"""Shared encoding helpers for the evaluation tables.

Two conventions live here because every table depends on them.

**Similarity.** Ranking uses :meth:`Hyper3CLIP.similarity`, the negative
Lorentz distance ``-d_L``.  The evaluator that produced the paper numbers
ranked by the Lorentz inner product ``<x,y>_L`` instead; the two give the
*identical ordering*, since ``-d_L = -acosh(-kappa <x,y>_L)/sqrt(kappa)`` is a
strictly decreasing function of ``-<x,y>_L`` at fixed curvature.  Every
argmax, top-k and per-class ranking is therefore unchanged.  The one place
where the choice is visible is a mean of scores across prompt templates
(:mod:`hyper3_clip.evaluation.multilabel` with more than one prompt), because
averaging is not invariant under a monotone reparametrisation; the paper's
multi-label runs use a single prompt, so the reported numbers are unaffected.

**Prompt ensembling.** :func:`build_text_classifier` averages the 512-d
*tangent* projections over the prompt templates and lifts the single average
onto the hyperboloid.  It is not a mean of Lorentz points and there is no L2
renormalisation of the averaged tangent vector.  Getting this wrong is the
easiest way to fail to reproduce Tables 3 and 6.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from hyper3_clip.models.hyper3_clip import Hyper3CLIP

__all__ = [
    "build_text_classifier",
    "encode_texts",
    "resolve_tokenizer",
    "similarity_matrix",
]


def resolve_tokenizer(model: Hyper3CLIP, tokenizer: Any | None = None) -> Any:
    """Return ``tokenizer`` or build the one named by ``model.config``."""
    if tokenizer is not None:
        return tokenizer
    from hyper3_clip.models.preprocessing import build_tokenizer

    return build_tokenizer(model.config)


def _tokenize(
    tokenizer: Any, texts: Sequence[str], max_text_length: int, device: torch.device
) -> tuple[Tensor, Tensor]:
    encoded = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=int(max_text_length),
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    attention_mask = torch.ones_like(input_ids) if attention_mask is None else attention_mask.to(device)
    return input_ids, attention_mask


@torch.inference_mode()
def encode_texts(
    model: Hyper3CLIP,
    tokenizer: Any,
    texts: Sequence[str],
    device: torch.device,
    *,
    max_text_length: int = 77,
    batch_size: int = 128,
) -> Tensor:
    """Encode ``texts`` as Lorentz points on the CPU, in ``batch_size`` chunks."""
    features: list[Tensor] = []
    for start in range(0, len(texts), batch_size):
        input_ids, attention_mask = _tokenize(tokenizer, texts[start : start + batch_size], max_text_length, device)
        features.append(model.encode_text(input_ids, attention_mask).cpu())
    if not features:
        raise ValueError("encode_texts requires at least one text")
    return torch.cat(features, dim=0)


@torch.inference_mode()
def build_text_classifier(
    model: Hyper3CLIP,
    tokenizer: Any,
    class_names: Sequence[str],
    prompts: Sequence[str],
    device: torch.device,
    *,
    max_text_length: int = 77,
) -> Tensor:
    """Build the ``[num_classes, embed_dim + 1]`` zero-shot classifier.

    Per class: format every template with the class name (underscores turned
    into spaces), encode them, average the **tangent** projections, then lift
    the average with :meth:`Hyper3CLIP.project_text_features`.  See the module
    docstring for why the order of those two steps is part of the protocol.
    """
    if not class_names:
        raise ValueError("build_text_classifier requires at least one class name")
    if not prompts:
        raise ValueError("build_text_classifier requires at least one prompt template")
    embeddings: list[Tensor] = []
    for class_name in class_names:
        readable = str(class_name).replace("_", " ")
        texts = [prompt.format(readable) for prompt in prompts]
        input_ids, attention_mask = _tokenize(tokenizer, texts, max_text_length, device)
        tangent = model.encode_text_tangent(input_ids, attention_mask).float().mean(dim=0, keepdim=True)
        embeddings.append(model.project_text_features(tangent).squeeze(0))
    return torch.stack(embeddings, dim=0)


def similarity_matrix(model: Hyper3CLIP, image_features: Tensor, text_features: Tensor) -> Tensor:
    """All-pairs ``-d_L`` scores, ``[num_images, num_texts]``, higher is better."""
    return model.similarity(image_features, text_features)
