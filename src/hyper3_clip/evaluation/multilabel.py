"""Zero-shot multi-label mAP on VOC and COCO (paper Table 4).

Every image is scored independently against every class prompt and each class
is ranked on its own; the reported number is the mean over classes of the
per-class average precision, ×100.  Table 4's "Avg." column is the unweighted
mean of the VOC and COCO mAPs and is computed by the summarizer, not here.

Two protocol details are inherited from the run that produced the paper's
numbers and are documented in ``docs/evaluation.md``:

* **Prompt ensembling is a mean of similarity scores**, not of embeddings —
  ``(batch, classes, prompts).mean(dim=2)``.  This differs from the
  single-label evaluator, which averages tangent features before lifting
  Both paper runs use one prompt, so the two coincide there.
* **Images with no annotated object are absent from the manifest**, because the
  builders key off the annotations.

Images use the square-resize retrieval transform, as the reported run did.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from hyper3_clip.data.transforms import build_retrieval_transform
from hyper3_clip.evaluation.encoding import encode_texts, resolve_tokenizer, similarity_matrix
from hyper3_clip.evaluation.metrics import mean_average_precision
from hyper3_clip.models.hyper3_clip import Hyper3CLIP

__all__ = [
    "MultiLabelSample",
    "evaluate_multilabel_zero_shot",
    "load_multilabel_samples",
]

_IMAGE_KEYS = ("image_path", "image", "filename", "file_name", "filepath")
_LABEL_KEYS = ("labels", "positive_labels", "objects", "classes", "class_names")
_SUBSET_KEYS = ("subset", "category", "task", "split")
_ID_KEYS = ("id", "source_id", "uid", "image_id")


@dataclass(frozen=True)
class MultiLabelSample:
    """One manifest row: an image and the set of classes present in it."""

    image_path: Path
    labels: tuple[str, ...]
    subset: str
    source_id: str


def load_multilabel_samples(
    manifest_path: str | Path,
    image_root: str | Path | None = None,
    max_items: int | None = None,
) -> list[MultiLabelSample]:
    """Read a ``.jsonl`` / ``.json`` / ``.csv`` / ``.tsv`` multi-label manifest.

    Per row the image key is the first present of
    ``image_path|image|filename|file_name|filepath`` and the labels the first
    present of ``labels|positive_labels|objects|classes|class_names`` (a list,
    a JSON string, or a ``|`` / ``;`` / ``,``-delimited string).  ``subset``
    defaults to ``"all"``; relative image paths need ``image_root``.
    """
    path = Path(manifest_path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("samples"), list):
            rows = list(payload["samples"])
        elif isinstance(payload, dict):
            rows = [{"id": key, **value} for key, value in payload.items() if isinstance(value, dict)]
        elif isinstance(payload, list):
            rows = list(payload)
        else:
            raise ValueError(f"Unsupported JSON payload in {path}")
    elif suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter=delimiter))
    else:
        raise ValueError(f"Unsupported multi-label manifest format {path.suffix!r}: {path}")

    samples = [_sample_from_row(row, image_root) for row in rows]
    return samples[:max_items] if max_items is not None else samples


def _sample_from_row(row: dict[str, Any], image_root: str | Path | None) -> MultiLabelSample:
    image_value = _first_present(row, _IMAGE_KEYS)
    labels_value = _first_present(row, _LABEL_KEYS)
    if image_value is None or labels_value is None:
        raise ValueError(f"Multi-label sample is missing image/labels fields: {row}")
    image_path = Path(str(image_value))
    if not image_path.is_absolute():
        if image_root is None:
            raise ValueError(f"Relative image path requires image_root: {image_path}")
        image_path = Path(image_root) / image_path
    return MultiLabelSample(
        image_path=image_path,
        labels=tuple(_parse_labels(labels_value)),
        subset=str(_first_present(row, _SUBSET_KEYS) or "all"),
        source_id=str(_first_present(row, _ID_KEYS) or image_value),
    )


def _first_present(row: dict[str, Any], keys: Sequence[str]) -> Any | None:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def _parse_labels(value: Any) -> list[str]:
    if isinstance(value, list | tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            parsed = json.loads(text)
            if not isinstance(parsed, list):
                raise ValueError(f"Expected list-valued labels JSON, got: {value}")
            return [str(item).strip() for item in parsed if str(item).strip()]
        delimiter = "|" if "|" in text else ";" if ";" in text else ","
        return [item.strip() for item in text.split(delimiter) if item.strip()]
    return [str(value).strip()] if str(value).strip() else []


@torch.inference_mode()
def evaluate_multilabel_zero_shot(
    model: Hyper3CLIP,
    samples: Sequence[MultiLabelSample],
    device: torch.device,
    *,
    class_names: Sequence[str] | None = None,
    tokenizer: Any | None = None,
    prompts: Sequence[str] = ("a photo of a {}.",),
    batch_size: int = 128,
    text_batch_size: int | None = None,
    image_size: int = 224,
    image_normalization: str = "imagenet",
    max_text_length: int = 77,
) -> dict[str, float]:
    """Mean average precision over classes (paper Table 4).

    ``class_names`` fixes the class order and therefore the score-matrix
    columns; labels in the manifest that are not in the list are dropped.  With
    no list the classes are the sorted union of the manifest's labels.
    """
    from PIL import Image

    if not samples:
        raise ValueError("Multi-label zero-shot evaluation requires at least one sample")
    if not prompts:
        raise ValueError("Multi-label zero-shot evaluation requires at least one prompt")

    model.eval()
    tokenizer = resolve_tokenizer(model, tokenizer)
    classes = list(class_names) if class_names else sorted({label for sample in samples for label in sample.labels})
    if not classes:
        raise ValueError("Multi-label zero-shot evaluation requires at least one class label")
    class_to_index = {name: index for index, name in enumerate(classes)}

    prompted = [prompt.format(name) for name in classes for prompt in prompts]
    text_features = encode_texts(
        model,
        tokenizer,
        prompted,
        device,
        max_text_length=max_text_length,
        batch_size=text_batch_size or batch_size,
    ).to(device)

    transform = build_retrieval_transform(image_size, normalization=image_normalization)
    scores_by_class: list[list[float]] = [[] for _ in classes]
    targets_by_class: list[list[int]] = [[] for _ in classes]

    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        tensors: list[Tensor] = []
        for sample in batch:
            with Image.open(sample.image_path) as image:
                tensors.append(transform(image.convert("RGB")))
        image_features = model.encode_image(torch.stack(tensors).to(device))
        matrix = similarity_matrix(model, image_features, text_features)
        matrix = matrix.detach().cpu().reshape(len(batch), len(classes), len(prompts)).mean(dim=2)
        for row, sample in enumerate(batch):
            present = {class_to_index[label] for label in sample.labels if label in class_to_index}
            for index in range(len(classes)):
                scores_by_class[index].append(float(matrix[row, index]))
                targets_by_class[index].append(1 if index in present else 0)

    map_value, valid_classes = mean_average_precision(targets_by_class, scores_by_class)
    return {
        "mean_average_precision": map_value,
        "mean_average_precision_pct": 100.0 * map_value,
        "num_samples": float(len(samples)),
        "num_classes": float(len(classes)),
        "valid_classes": float(valid_classes),
        "num_prompts": float(len(prompts)),
    }
