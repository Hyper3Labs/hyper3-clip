"""Dependency-free metric implementations shared by the evaluation tables.

Every metric here is written out rather than delegated to scikit-learn, both to
keep the evaluation importable with no extra dependency and because the exact
tie convention is part of the protocol:

* :func:`recall_at_k` is the multi-positive recall of paper Table 2's *text
  retrieval* column — an image counts as a hit when **any** of its captions
  lands in the top ``k``.
* :func:`single_target_recall_at_k` is the one-correct-image recall of the
  *image retrieval* column.
* :func:`ranked_average_precision` is the Table 4 mAP flavour: rank by score
  descending with ties broken by insertion order, average ``hits(r)/r`` over
  the positives, no interpolation and no tie averaging.
* :func:`average_precision` is the Table 5 flavour: a step-wise AP that groups
  equal scores into a single step.
* :func:`roc_auc` is the Table 5 AUROC: the rank-sum statistic with average
  ranks over ties.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

__all__ = [
    "average_precision",
    "mean_average_precision",
    "mean_per_class_accuracy",
    "ranked_average_precision",
    "recall_at_k",
    "roc_auc",
    "single_target_recall_at_k",
    "tie_diagnostics",
]


def recall_at_k(scores: Tensor, targets: Sequence[Sequence[int]], k: int) -> float:
    """Fraction of rows whose top-``k`` columns contain at least one target.

    ``scores`` is ``[rows, columns]`` and ``targets[i]`` lists every column
    index that counts as correct for row ``i``.  ``k`` is clipped to the number
    of columns, so a fixture with fewer than ``k`` candidates still scores.
    """
    if scores.ndim != 2:
        raise ValueError(f"scores must be a 2-D matrix, got shape {tuple(scores.shape)}")
    if len(targets) != scores.shape[0]:
        raise ValueError(f"targets has {len(targets)} rows for a score matrix with {scores.shape[0]}")
    topk = scores.topk(k=min(int(k), scores.shape[1]), dim=1).indices
    hits = 0
    for row, row_targets in enumerate(targets):
        target_tensor = torch.as_tensor(list(row_targets), dtype=torch.long, device=topk.device)
        if target_tensor.numel() and bool(torch.isin(target_tensor, topk[row]).any().item()):
            hits += 1
    return hits / scores.shape[0]


def single_target_recall_at_k(scores: Tensor, targets: Tensor, k: int) -> float:
    """Fraction of rows whose single correct column is in the top ``k``."""
    if scores.ndim != 2:
        raise ValueError(f"scores must be a 2-D matrix, got shape {tuple(scores.shape)}")
    topk = scores.topk(k=min(int(k), scores.shape[1]), dim=1).indices
    targets = targets.to(topk.device).view(-1, 1)
    return float((topk == targets).any(dim=1).float().mean().item())


def mean_per_class_accuracy(per_class_correct: Tensor, per_class_total: Tensor) -> float:
    """Mean of the per-class accuracies over classes with at least one image.

    This is the number the paper reports for every zero-shot classification
    column (Tables 3 and 6); plain top-1 is reported alongside it but is not the
    headline metric.
    """
    observed = per_class_total > 0
    if not bool(observed.any().item()):
        raise ValueError("mean-per-class accuracy needs at least one observed class")
    return float((per_class_correct[observed] / per_class_total[observed]).mean().item())


def ranked_average_precision(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    """Average precision for one class of the multi-label table (paper Table 4).

    Ranks descending by score with ties broken by insertion order, then
    averages the precision at every positive's rank.  Returns ``None`` for a
    class with no positive, which the mean then excludes.
    """
    positives = sum(1 for label in labels if label)
    if positives == 0:
        return None
    hits = 0
    precision_sum = 0.0
    ordered = sorted(zip(scores, labels, strict=True), key=lambda pair: pair[0], reverse=True)
    for rank, (_, label) in enumerate(ordered, start=1):
        if label:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / positives


def mean_average_precision(
    targets_by_class: Sequence[Sequence[int]],
    scores_by_class: Sequence[Sequence[float]],
) -> tuple[float, int]:
    """Mean of :func:`ranked_average_precision` over classes that have a positive.

    Returns ``(mAP, number_of_valid_classes)``.
    """
    values = [
        value
        for targets, scores in zip(targets_by_class, scores_by_class, strict=True)
        if (value := ranked_average_precision(scores, targets)) is not None
    ]
    if not values:
        raise ValueError("Mean AP is undefined because no class has a positive example")
    return sum(values) / len(values), len(values)


def average_precision(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Step-wise average precision with equal scores collapsed into one step.

    This is the "Hierarchy AP" of paper Table 5.  Grouping ties matters there:
    the clamped entailment score saturates, and without grouping the value
    would report the sort's tie-breaking rather than the model.
    """
    positives = sum(1 for label in labels if label)
    if positives == 0:
        raise ValueError("Average precision requires at least one positive label")
    ordered = sorted(zip(scores, labels, strict=True), key=lambda pair: pair[0], reverse=True)
    true_positives = 0
    false_positives = 0
    previous_recall = 0.0
    result = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        group_positives = sum(label for _, label in ordered[index:end])
        true_positives += group_positives
        false_positives += (end - index) - group_positives
        recall = true_positives / positives
        precision = true_positives / (true_positives + false_positives)
        result += (recall - previous_recall) * precision
        previous_recall = recall
        index = end
    return result


def roc_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Rank-sum AUROC with average ranks over ties (paper Table 5)."""
    positives = sum(1 for label in labels if label)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("ROC AUC requires both positive and negative labels")
    ordered = sorted(zip(scores, labels, strict=True), key=lambda pair: pair[0])
    rank_sum_positive = 0.0
    rank = 1
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = (rank + rank + (end - index) - 1) / 2.0
        rank_sum_positive += average_rank * sum(label for _, label in ordered[index:end])
        rank += end - index
        index = end
    return (rank_sum_positive - positives * (positives + 1) / 2.0) / (positives * negatives)


def tie_diagnostics(scores: Sequence[float]) -> dict[str, float]:
    """Score-concentration statistics reported next to every entailment number.

    A clamped score can saturate and make AUROC report the tie-breaking
    convention instead of the model, so the fraction of scores pinned at the
    modal value (and at exactly zero) is emitted alongside AP and AUROC.
    """
    total = len(scores)
    if total == 0:
        raise ValueError("tie diagnostics require at least one score")
    counts: dict[float, int] = {}
    for value in scores:
        counts[value] = counts.get(value, 0) + 1
    modal_value, modal_count = max(counts.items(), key=lambda item: item[1])
    return {
        "num_scores": float(total),
        "distinct_scores": float(len(counts)),
        "modal_score": float(modal_value),
        "modal_score_fraction": modal_count / total,
        "exactly_zero_fraction": sum(1 for value in scores if value == 0.0) / total,
    }
