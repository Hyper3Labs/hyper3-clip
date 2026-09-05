"""ImageFolder zero-shot classification (paper Tables 3 and 6).

The 15 non-ImageNet datasets of Table 3 are evaluated from materialized
``ImageFolder`` trees whose class directories are named ``f"{index:04d}_{slug}"``,
so sorted folder order equals the order of the vendored class-name list a task
selects with ``class_names_key``.

The reported metric is **mean-per-class accuracy**, averaged over classes with
at least one image; plain top-1 is emitted next to it but is not the paper's
number.  Prompts come from :mod:`hyper3_clip.evaluation.prompts`; the prompt
regime (``"official"`` vs ``"photo"``) is what Table 6 varies, and Table 3
already reports Food-101, CUB and Flowers-102 in the Photo regime.

Prompt ensembling averages tangent-space text features and lifts once — see
:func:`hyper3_clip.evaluation.encoding.build_text_classifier`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from hyper3_clip.data.transforms import build_eval_transform
from hyper3_clip.evaluation.class_names import resolve_class_names
from hyper3_clip.evaluation.encoding import build_text_classifier, resolve_tokenizer, similarity_matrix
from hyper3_clip.evaluation.metrics import mean_per_class_accuracy
from hyper3_clip.evaluation.prompts import resolve_prompts
from hyper3_clip.models.hyper3_clip import Hyper3CLIP

__all__ = ["evaluate_imagefolder_zero_shot"]


@torch.inference_mode()
def evaluate_imagefolder_zero_shot(
    model: Hyper3CLIP,
    root: str | Path,
    device: torch.device,
    *,
    tokenizer: Any | None = None,
    prompts: Sequence[str] | None = None,
    prompt_set: str | None = None,
    prompt_regime: str | None = None,
    class_names: Sequence[str] | None = None,
    class_names_key: str | None = None,
    batch_size: int = 128,
    image_size: int = 224,
    image_normalization: str = "imagenet",
    max_text_length: int = 77,
    max_items: int | None = None,
    num_workers: int = 4,
) -> dict[str, float]:
    """Zero-shot classify an ``ImageFolder`` and return mean-per-class accuracy.

    ``prompt_regime="photo"`` overrides ``prompt_set`` with the single
    ``"a photo of a {}."`` template; ``"official"`` keeps ``prompt_set``.  An
    explicit ``prompts`` list wins over both.
    """
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets

    model.eval()
    tokenizer = resolve_tokenizer(model, tokenizer)
    dataset = datasets.ImageFolder(str(Path(root)), transform=build_eval_transform(image_size, image_normalization))
    resolved_names = resolve_class_names(
        dataset.classes,
        class_names=class_names,
        class_names_key=class_names_key,
    )
    resolved_prompts = resolve_prompts(prompts=prompts, prompt_set=prompt_set, prompt_regime=prompt_regime)
    classifier = build_text_classifier(
        model, tokenizer, resolved_names, resolved_prompts, device, max_text_length=max_text_length
    )

    evaluated = dataset if max_items is None else Subset(dataset, range(min(int(max_items), len(dataset))))
    loader = DataLoader(
        evaluated,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    num_classes = len(dataset.classes)
    per_class_correct = torch.zeros(num_classes, dtype=torch.float64)
    per_class_total = torch.zeros(num_classes, dtype=torch.float64)
    correct = 0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        predictions = similarity_matrix(model, model.encode_image(images), classifier).argmax(dim=1)
        matches = predictions == targets
        correct += int(matches.sum().item())
        total += int(targets.numel())
        cpu_targets = targets.cpu()
        per_class_correct.scatter_add_(0, cpu_targets, matches.cpu().double())
        per_class_total.scatter_add_(0, cpu_targets, torch.ones_like(cpu_targets, dtype=torch.float64))

    if total == 0:
        raise ValueError(f"No images found under {root}")
    top1 = correct / total
    return {
        "top1": top1,
        "top1_pct": 100.0 * top1,
        "mean_per_class_acc_pct": 100.0 * mean_per_class_accuracy(per_class_correct, per_class_total),
        "num_images": float(total),
        "num_classes": float(num_classes),
        "num_prompts": float(len(resolved_prompts)),
    }
