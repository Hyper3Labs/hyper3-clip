"""Hierarchy entailment on a HierarCaps-style file (paper Table 5).

Each sample is one image plus a caption hierarchy ordered coarse to fine, so
``positive_captions[-1]`` is the finest caption.  The task is binary: rank
(image, caption) pairs so that an image's own hierarchy captions come above
captions belonging to other images.

Pair construction (1,000 images x 4 hierarchy levels = 4,000 positives, 1,000 x
100 = 100,000 negatives in the reported run):

* **Positives** — every caption of the image's own hierarchy, paired with that
  image.
* **Negatives** — the finest caption of every *other* sample, in dataset order,
  with any caption that is also a positive for this image removed and the pool
  then truncated to ``max_negatives_per_image``.  Negative construction is
  deterministic: the same roughly 100 captions serve as negatives for every
  image, which is what the reported run did.  The paper's wording says
  "sampled"; this is the one documented difference here.

Scoring direction follows the paper: the **image is the specific node** and the
caption the **general node** that roots the entailment cone.

* ``score="entailment_score"`` (default) is the paper's
  ``p(a <= b) = max(1 - 2*phi/pi, 0)``, i.e.
  :meth:`Hyper3CLIP.entailment_score`.
* ``score="signed_margin"`` is the unclamped signed cone margin
  ``-(phi - Theta(caption))``.  It is the setting that produced the published
  Table 5 numbers, because the clamped score saturates once a pair is inside
  the cone; the returned tie diagnostics say whether that is happening.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import torch
from torch import Tensor

from hyper3_clip.data.transforms import build_eval_transform
from hyper3_clip.evaluation.encoding import encode_texts, resolve_tokenizer
from hyper3_clip.evaluation.metrics import average_precision, roc_auc, tie_diagnostics
from hyper3_clip.geometry.lorentz import exterior_angle, half_aperture
from hyper3_clip.models.hyper3_clip import Hyper3CLIP

__all__ = [
    "HierarchyEntailmentSample",
    "SCORE_FUNCTIONS",
    "build_entailment_pairs",
    "evaluate_hierarchy_entailment",
    "load_hierarchy_entailment_samples",
]

#: Accepted ``score`` values; the first is the paper's formula.
SCORE_FUNCTIONS: tuple[str, ...] = ("entailment_score", "signed_margin")

_IMAGE_KEYS = ("image_path", "path", "file_name", "filename")
_URL_KEYS = ("image_url", "url")
_IMAGE_ID_KEYS = ("image_id", "id", "image_path", "path", "file_name", "filename", "image_url", "url")
_POSITIVE_KEYS = ("positive_captions", "hierarchical_captions", "caption_hierarchy", "captions")
_NEGATIVE_KEYS = ("negative_captions", "negatives", "negative_pool")
_TRUTHY = {"1", "true", "yes", "positive", "pos"}


@dataclass(frozen=True)
class HierarchyEntailmentSample:
    """One image and its coarse-to-fine caption hierarchy."""

    image_id: str
    image_path: Path
    positive_captions: tuple[str, ...]
    negative_captions: tuple[str, ...] = ()


def load_hierarchy_entailment_samples(
    annotations_path: str | Path,
    image_root: str | Path | None = None,
    max_items: int | None = None,
) -> list[HierarchyEntailmentSample]:
    """Read a HierarCaps-style ``.csv`` / ``.tsv`` / ``.json`` / ``.jsonl`` file.

    A CSV with both a ``caption`` and a ``label`` column is one row per
    (image, caption) pair and is regrouped by image key into positives (label
    in ``1/true/yes/positive/pos``) and negatives, preserving row order.
    Otherwise each row is one image, and the caption hierarchy is the first
    present of ``positive_captions|hierarchical_captions|caption_hierarchy|captions``
    — a JSON list, or a string split on ``=>`` (then ``||``).  **Caption order
    is meaningful**: the last entry is the finest node.
    """
    path = Path(annotations_path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = list(payload.get("samples", payload.get("data", []))) if isinstance(payload, dict) else list(payload)
    elif suffix in {".csv", ".tsv"}:
        rows = _rows_from_table(path)
    else:
        raise ValueError(f"Unsupported annotations format {path.suffix!r}; expected .json, .jsonl, .csv or .tsv")

    samples = [_sample_from_row(row, image_root) for row in rows]
    return samples[:max_items] if max_items is not None else samples


def _rows_from_table(path: Path) -> list[dict[str, Any]]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter=delimiter))
    if not rows:
        return []
    if "caption" not in rows[0] or "label" not in rows[0]:
        return rows
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(_first_present(row, _IMAGE_ID_KEYS))
        item = grouped.setdefault(key, {**row, "positive_captions": [], "negative_captions": []})
        target = "positive_captions" if str(row["label"]).strip().lower() in _TRUTHY else "negative_captions"
        item[target].append(row["caption"])
    return list(grouped.values())


def _sample_from_row(row: dict[str, Any], image_root: str | Path | None) -> HierarchyEntailmentSample:
    positives = _caption_tuple(_first_present(row, _POSITIVE_KEYS))
    if not positives:
        caption = row.get("caption")
        positives = (str(caption).strip(),) if caption else ()
    if not positives:
        raise ValueError(f"Hierarchy-entailment sample has no positive captions: {row}")
    raw_path = _first_present(row, _IMAGE_KEYS)
    if raw_path is None:
        # The public HierarCaps release names images by URL only
        # (``image_url``); the file name resolves against ``image_root``.
        raw_url = _first_present(row, _URL_KEYS)
        if raw_url is None:
            raise ValueError(f"Hierarchy-entailment sample has no image path: {row}")
        raw_path = urlsplit(str(raw_url)).path.rsplit("/", 1)[-1]
        if not raw_path:
            raise ValueError(f"Hierarchy-entailment sample has an image URL without a file name: {row}")
    image_path = Path(str(raw_path))
    if not image_path.is_absolute():
        if image_root is None:
            raise ValueError(f"Relative image path requires image_root: {image_path}")
        image_path = Path(image_root) / image_path
    return HierarchyEntailmentSample(
        image_id=str(_first_present(row, _IMAGE_ID_KEYS) or image_path),
        image_path=image_path,
        positive_captions=positives,
        negative_captions=_caption_tuple(_first_present(row, _NEGATIVE_KEYS)),
    )


def _first_present(row: dict[str, Any], keys: Sequence[str]) -> Any | None:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def _caption_tuple(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return ()
        if text.startswith("["):
            try:
                return _caption_tuple(json.loads(text))
            except json.JSONDecodeError:
                pass
        separator = "=>" if "=>" in text else "||" if "||" in text else None
        values = text.split(separator) if separator else [text]
        return tuple(value.strip() for value in values if value.strip())
    if isinstance(raw, list | tuple):
        return tuple(str(value).strip() for value in raw if str(value).strip())
    return (str(raw).strip(),)


def build_entailment_pairs(
    samples: Sequence[HierarchyEntailmentSample],
    *,
    max_negatives_per_image: int | None = 100,
) -> tuple[list[int], list[str], list[int]]:
    """Return ``(image_indices, captions, labels)`` for every scored pair.

    Negatives come from an image's own ``negative_captions`` when the file
    supplies them, otherwise from the finest caption of every other sample.
    The pool keeps its first ``max_negatives_per_image`` entries in dataset
    order, so the construction is deterministic and needs no seed.
    """
    fine_captions = tuple(sample.positive_captions[-1] for sample in samples)
    image_indices: list[int] = []
    captions: list[str] = []
    labels: list[int] = []
    for index, sample in enumerate(samples):
        positives = set(sample.positive_captions)
        for caption in sample.positive_captions:
            image_indices.append(index)
            captions.append(caption)
            labels.append(1)
        pool = sample.negative_captions or tuple(
            caption for other, caption in enumerate(fine_captions) if other != index
        )
        pool = tuple(caption for caption in pool if caption not in positives)
        if max_negatives_per_image is not None:
            pool = pool[:max_negatives_per_image]
        for caption in pool:
            image_indices.append(index)
            captions.append(caption)
            labels.append(0)
    return image_indices, captions, labels


def _score_pairs(
    model: Hyper3CLIP,
    image_features: Tensor,
    text_features: Tensor,
    image_indices: Sequence[int],
    text_indices: Sequence[int],
    device: torch.device,
    score: str,
    pair_batch_size: int,
) -> list[float]:
    curvature = model.curvature.detach().to(device)
    values: list[Tensor] = []
    for start in range(0, len(image_indices), pair_batch_size):
        rows = torch.tensor(image_indices[start : start + pair_batch_size], dtype=torch.long)
        columns = torch.tensor(text_indices[start : start + pair_batch_size], dtype=torch.long)
        specific = image_features.index_select(0, rows).to(device)
        general = text_features.index_select(0, columns).to(device)
        if score == "entailment_score":
            values.append(model.entailment_score(general, specific).cpu())
        else:
            angle = exterior_angle(specific, general, curvature)
            values.append((-(angle - half_aperture(general, curvature))).cpu())
    return torch.cat(values).double().tolist()


@torch.inference_mode()
def evaluate_hierarchy_entailment(
    model: Hyper3CLIP,
    annotations_path: str | Path,
    device: torch.device,
    *,
    image_root: str | Path | None = None,
    tokenizer: Any | None = None,
    score: str = "entailment_score",
    max_negatives_per_image: int | None = 100,
    batch_size: int = 128,
    pair_batch_size: int = 8192,
    image_size: int = 224,
    image_normalization: str = "imagenet",
    max_text_length: int = 77,
    max_items: int | None = None,
) -> dict[str, float]:
    """Hierarchy AP and AUROC over image/caption entailment pairs (paper Table 5).

    Returns ``average_precision_pct`` (the paper's "Hierarchy AP") and
    ``auc_roc_pct`` (its AUROC), the raw fractions, the pair counts, the mean
    positive and negative scores, and the tie diagnostics that reveal a
    saturated score.
    """
    from PIL import Image

    if score not in SCORE_FUNCTIONS:
        raise ValueError(f"score must be one of {list(SCORE_FUNCTIONS)}, got {score!r}")

    model.eval()
    tokenizer = resolve_tokenizer(model, tokenizer)
    samples = load_hierarchy_entailment_samples(annotations_path, image_root=image_root, max_items=max_items)
    if not samples:
        raise ValueError("hierarchy entailment requires at least one sample")

    transform = build_eval_transform(image_size, image_normalization)
    image_features: list[Tensor] = []
    batch: list[Tensor] = []
    for index, sample in enumerate(samples):
        with Image.open(sample.image_path) as image:
            batch.append(transform(image.convert("RGB")))
        if len(batch) == batch_size or index == len(samples) - 1:
            image_features.append(model.encode_image(torch.stack(batch).to(device)).cpu())
            batch = []
    images = torch.cat(image_features)

    image_indices, pair_captions, labels = build_entailment_pairs(
        samples, max_negatives_per_image=max_negatives_per_image
    )
    if not any(labels) or all(labels):
        raise ValueError("hierarchy entailment requires both positive and negative pairs")

    unique_captions = sorted(set(pair_captions))
    caption_to_index = {caption: index for index, caption in enumerate(unique_captions)}
    texts = encode_texts(
        model, tokenizer, unique_captions, device, max_text_length=max_text_length, batch_size=batch_size
    )
    scores = _score_pairs(
        model,
        images,
        texts,
        image_indices,
        [caption_to_index[caption] for caption in pair_captions],
        device,
        score,
        pair_batch_size,
    )

    positives = [value for value, label in zip(scores, labels, strict=True) if label == 1]
    negatives = [value for value, label in zip(scores, labels, strict=True) if label == 0]
    auc = roc_auc(scores, labels)
    ap = average_precision(scores, labels)
    return {
        "average_precision": ap,
        "average_precision_pct": 100.0 * ap,
        "auc_roc": auc,
        "auc_roc_pct": 100.0 * auc,
        "num_samples": float(len(samples)),
        "num_pairs": float(len(labels)),
        "num_positive_pairs": float(len(positives)),
        "num_negative_pairs": float(len(negatives)),
        "mean_positive_score": sum(positives) / len(positives),
        "mean_negative_score": sum(negatives) / len(negatives),
        **tie_diagnostics(scores),
    }
