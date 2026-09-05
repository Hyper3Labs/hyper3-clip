"""ImageNet zero-shot accuracy and WordNet hierarchy metrics (paper Tables 2 and 3).

One prediction pass produces every ImageNet number the paper reports: the
Table 3 "IN" column (mean-per-class accuracy, top-1 alongside it) and the
Table 2 hierarchy block (TIE, LCA, Jaccard, hierarchical precision,
hierarchical recall).  Fusing them is not an optimisation detail — the two
tables are read off the *same* forward pass, so they can never disagree
(see ``docs/evaluation.md``).

Hierarchy definitions, with ``A_p`` / ``A_t`` the ancestor-index sets of the
predicted and true official labels, ``I = A_p & A_t`` and ``U = A_p | A_t``:

===================  ===========================================================
TIE                  undirected shortest-path length between the predicted and
                     true synset in the WordNet is-a graph
LCA                  ``|A_p| - |I| + 1``
Jaccard              ``|I| / |U|``
hierarchical prec.   ``|I| / |A_p|``
hierarchical recall  ``|I| / |A_t|``
===================  ===========================================================

Each is summed over images and divided by the image count.  The three assets
that define the hierarchy ship inside the package (``assets/PROVENANCE.md``),
so ``imagenet_hierarchy_assets_root`` only has to be set to override them.
"""

from __future__ import annotations

import csv
import pickle
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from hyper3_clip.data.transforms import build_eval_transform
from hyper3_clip.evaluation.class_names import (
    ASSETS_DIR,
    imagenet_class_names,
    looks_like_wnid,
    read_label_to_wnid,
)
from hyper3_clip.evaluation.encoding import build_text_classifier, resolve_tokenizer, similarity_matrix
from hyper3_clip.evaluation.metrics import mean_per_class_accuracy
from hyper3_clip.evaluation.prompts import IMAGENET_PROMPTS
from hyper3_clip.models.hyper3_clip import Hyper3CLIP

__all__ = [
    "WordNetHierarchy",
    "evaluate_imagenet",
    "load_wordnet_hierarchy",
]


@dataclass(frozen=True)
class WordNetHierarchy:
    """The ImageNet is-a graph plus the official label order and ancestor sets.

    ``synsets[i]`` is the WordNet id of official label ``i`` and
    ``ancestors[i]`` its ancestor label indices.  ``adjacency`` is the is-a
    graph made undirected, which is what the tree-induced error walks.
    """

    synsets: tuple[str, ...]
    ancestors: tuple[frozenset[int], ...]
    adjacency: dict[str, tuple[str, ...]]

    @classmethod
    def from_parts(
        cls,
        synsets: Sequence[str],
        ancestors: Sequence[Sequence[int]],
        edges: Iterable[tuple[str, str]],
    ) -> WordNetHierarchy:
        """Build a hierarchy from in-memory parts (used by the tests' toy tree)."""
        neighbours: dict[str, set[str]] = {}
        for parent, child in edges:
            neighbours.setdefault(parent, set()).add(child)
            neighbours.setdefault(child, set()).add(parent)
        return cls(
            synsets=tuple(str(name) for name in synsets),
            ancestors=tuple(frozenset(int(index) for index in group) for group in ancestors),
            adjacency={node: tuple(sorted(values)) for node, values in neighbours.items()},
        )

    def path_length(self, source: str, target: str) -> int:
        """Undirected shortest-path length between two synsets (breadth-first)."""
        if source == target:
            return 0
        seen = {source}
        frontier = deque([(source, 0)])
        while frontier:
            node, distance = frontier.popleft()
            for neighbour in self.adjacency.get(node, ()):
                if neighbour == target:
                    return distance + 1
                if neighbour not in seen:
                    seen.add(neighbour)
                    frontier.append((neighbour, distance + 1))
        raise ValueError(f"No path between synsets {source!r} and {target!r}")

    def totals(self, predicted_labels: Sequence[int], true_labels: Sequence[int]) -> tuple[float, ...]:
        """Unnormalised (TIE, LCA, Jaccard, H-P, H-R) sums over a batch."""
        tie = lca = jaccard = precision = recall = 0.0
        for predicted, true in zip(predicted_labels, true_labels, strict=True):
            predicted_ancestors = self.ancestors[predicted]
            true_ancestors = self.ancestors[true]
            intersection = predicted_ancestors & true_ancestors
            union = predicted_ancestors | true_ancestors
            tie += self.path_length(self.synsets[predicted], self.synsets[true])
            lca += len(predicted_ancestors) - len(intersection) + 1
            jaccard += len(intersection) / len(union)
            precision += len(intersection) / len(predicted_ancestors)
            recall += len(intersection) / len(true_ancestors)
        return tie, lca, jaccard, precision, recall

    def metrics(self, predicted_labels: Sequence[int], true_labels: Sequence[int]) -> dict[str, float]:
        """The five hierarchy metrics of paper Table 2, averaged over images."""
        count = len(true_labels)
        if count == 0:
            raise ValueError("hierarchy metrics require at least one image")
        tie, lca, jaccard, precision, recall = self.totals(predicted_labels, true_labels)
        return {
            "tie": tie / count,
            "lca": lca / count,
            "jaccard": jaccard / count,
            "hierarchical_precision": precision / count,
            "hierarchical_recall": recall / count,
            "num_images": float(count),
        }


def load_wordnet_hierarchy(assets_root: str | Path | None = None) -> WordNetHierarchy:
    """Load ``all_synsets.pkl`` / ``all_ancestors_indices.pkl`` / ``imagenet_isa.txt``.

    ``assets_root`` defaults to the copies shipped inside the package.
    """
    root = ASSETS_DIR if assets_root is None else Path(assets_root)
    with (root / "all_synsets.pkl").open("rb") as handle:
        synsets = pickle.load(handle)
    with (root / "all_ancestors_indices.pkl").open("rb") as handle:
        ancestors = pickle.load(handle)
    with (root / "imagenet_isa.txt").open("r", encoding="utf-8") as handle:
        edges = [(row[0], row[1]) for row in csv.reader(handle, delimiter=" ") if len(row) >= 2]
    return WordNetHierarchy.from_parts(synsets, ancestors, edges)


def _dataset_to_official_indices(
    folder_classes: Sequence[str],
    imagenet_val_root: str | Path,
    synsets: Sequence[str],
) -> Tensor:
    """Map ``ImageFolder`` class indices onto official ImageNet label indices."""
    wnid_to_label = read_label_to_wnid(imagenet_val_root)
    if wnid_to_label is not None:
        return torch.tensor([wnid_to_label[name] for name in folder_classes], dtype=torch.long)
    if all(looks_like_wnid(name) for name in folder_classes):
        synset_to_label = {synset: label for label, synset in enumerate(synsets)}
        return torch.tensor([synset_to_label[name] for name in folder_classes], dtype=torch.long)
    return torch.arange(len(folder_classes), dtype=torch.long)


@torch.inference_mode()
def evaluate_imagenet(
    model: Hyper3CLIP,
    imagenet_val_root: str | Path,
    device: torch.device,
    *,
    assets_root: str | Path | None = None,
    hierarchy: bool = True,
    tokenizer: Any | None = None,
    prompts: Sequence[str] = IMAGENET_PROMPTS,
    batch_size: int = 128,
    image_size: int = 224,
    image_normalization: str = "imagenet",
    max_text_length: int = 77,
    max_items: int | None = None,
    num_workers: int = 4,
) -> tuple[dict[str, float], dict[str, float]]:
    """One pass over ImageNet val; returns ``(zero_shot_metrics, hierarchy_metrics)``.

    With ``hierarchy=False`` the second dict is empty and the assets are not
    read.
    """
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets

    model.eval()
    tokenizer = resolve_tokenizer(model, tokenizer)
    root = Path(imagenet_val_root)
    dataset = datasets.ImageFolder(str(root), transform=build_eval_transform(image_size, image_normalization))
    class_names = imagenet_class_names(dataset.classes, root)
    classifier = build_text_classifier(
        model, tokenizer, class_names, prompts, device, max_text_length=max_text_length
    )

    evaluated = dataset if max_items is None else Subset(dataset, range(min(int(max_items), len(dataset))))
    loader = DataLoader(
        evaluated,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    tree: WordNetHierarchy | None = None
    dataset_to_official: Tensor | None = None
    if hierarchy:
        tree = load_wordnet_hierarchy(assets_root)
        dataset_to_official = _dataset_to_official_indices(dataset.classes, root, tree.synsets).to(device)

    num_classes = len(dataset.classes)
    per_class_correct = torch.zeros(num_classes, dtype=torch.float64)
    per_class_total = torch.zeros(num_classes, dtype=torch.float64)
    hierarchy_totals = torch.zeros(5, dtype=torch.float64)
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
        if tree is not None and dataset_to_official is not None:
            batch_totals = tree.totals(
                dataset_to_official[predictions].cpu().tolist(),
                dataset_to_official[targets].cpu().tolist(),
            )
            hierarchy_totals += torch.tensor(batch_totals, dtype=torch.float64)

    if total == 0:
        raise ValueError(f"No images found under {root}")
    top1 = correct / total
    zero_shot = {
        "top1": top1,
        "top1_pct": 100.0 * top1,
        "mean_per_class_acc_pct": 100.0 * mean_per_class_accuracy(per_class_correct, per_class_total),
        "num_images": float(total),
        "num_classes": float(num_classes),
    }
    if tree is None:
        return zero_shot, {}
    averages = hierarchy_totals / total
    hierarchy_metrics = {
        "tie": float(averages[0].item()),
        "lca": float(averages[1].item()),
        "jaccard": float(averages[2].item()),
        "hierarchical_precision": float(averages[3].item()),
        "hierarchical_recall": float(averages[4].item()),
        "num_images": float(total),
    }
    return zero_shot, hierarchy_metrics
