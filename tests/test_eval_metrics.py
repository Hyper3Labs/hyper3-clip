"""Metric implementations: recall@k, mean-per-class accuracy, mAP, AP and AUROC."""

from __future__ import annotations

import itertools
import math

import pytest
import torch

from hyper3_clip.evaluation.metrics import (
    average_precision,
    mean_average_precision,
    mean_per_class_accuracy,
    ranked_average_precision,
    recall_at_k,
    roc_auc,
    single_target_recall_at_k,
    tie_diagnostics,
)


def test_recall_at_k_counts_any_of_several_positives() -> None:
    # Row 0's captions are columns 0 and 1, row 1's are 2 and 3.
    scores = torch.tensor(
        [
            [0.9, 0.1, 0.8, 0.7],  # top-1 is a hit, top-2 still a hit
            [0.9, 0.8, 0.1, 0.2],  # top-1 and top-2 both miss, top-3 hits
        ]
    )
    targets = [[0, 1], [2, 3]]
    assert recall_at_k(scores, targets, 1) == pytest.approx(0.5)
    assert recall_at_k(scores, targets, 2) == pytest.approx(0.5)
    assert recall_at_k(scores, targets, 3) == pytest.approx(1.0)


def test_recall_at_k_clips_k_to_the_number_of_columns() -> None:
    scores = torch.tensor([[0.2, 0.4]])
    assert recall_at_k(scores, [[1]], 10) == pytest.approx(1.0)


def test_recall_at_k_rejects_a_target_row_mismatch() -> None:
    with pytest.raises(ValueError, match="targets has 1 rows"):
        recall_at_k(torch.zeros(2, 2), [[0]], 1)


def test_single_target_recall_at_k() -> None:
    scores = torch.tensor(
        [
            [0.1, 0.9, 0.5],
            [0.9, 0.1, 0.5],
            [0.1, 0.5, 0.9],
        ]
    )
    targets = torch.tensor([1, 1, 2])
    # Row 1 ranks columns 0 and 2 above the correct column 1.
    assert single_target_recall_at_k(scores, targets, 1) == pytest.approx(2 / 3)
    assert single_target_recall_at_k(scores, targets, 2) == pytest.approx(2 / 3)
    assert single_target_recall_at_k(scores, targets, 3) == pytest.approx(1.0)


def test_mean_per_class_accuracy_ignores_unobserved_classes() -> None:
    correct = torch.tensor([2.0, 1.0, 0.0], dtype=torch.float64)
    total = torch.tensor([2.0, 4.0, 0.0], dtype=torch.float64)
    # Class 2 has no images and must not drag the mean towards zero.
    assert mean_per_class_accuracy(correct, total) == pytest.approx((1.0 + 0.25) / 2)


def test_mean_per_class_accuracy_needs_an_observed_class() -> None:
    with pytest.raises(ValueError, match="at least one observed class"):
        mean_per_class_accuracy(torch.zeros(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64))


def test_ranked_average_precision_matches_a_hand_computed_value() -> None:
    # Ranking: 1, 0, 1, 0 -> (1/1 + 2/3) / 2.
    value = ranked_average_precision([0.9, 0.8, 0.7, 0.6], [1, 0, 1, 0])
    assert value == pytest.approx((1.0 + 2 / 3) / 2)


def test_ranked_average_precision_is_one_for_a_perfect_ranking() -> None:
    assert ranked_average_precision([0.9, 0.8, 0.1], [1, 1, 0]) == pytest.approx(1.0)


def test_ranked_average_precision_is_none_without_positives() -> None:
    assert ranked_average_precision([0.9, 0.1], [0, 0]) is None


def test_mean_average_precision_excludes_classes_without_positives() -> None:
    value, valid = mean_average_precision([[1, 0], [0, 0]], [[0.9, 0.1], [0.5, 0.4]])
    assert valid == 1
    assert value == pytest.approx(1.0)


def test_mean_average_precision_needs_one_positive_class() -> None:
    with pytest.raises(ValueError, match="no class has a positive"):
        mean_average_precision([[0, 0]], [[0.1, 0.2]])


def _brute_force_auc(scores: list[float], labels: list[int]) -> float:
    """Definition of AUROC: P(score_pos > score_neg) + 0.5 P(tie)."""
    positives = [score for score, label in zip(scores, labels, strict=True) if label]
    negatives = [score for score, label in zip(scores, labels, strict=True) if not label]
    total = 0.0
    for positive, negative in itertools.product(positives, negatives):
        total += 1.0 if positive > negative else 0.5 if positive == negative else 0.0
    return total / (len(positives) * len(negatives))


def _brute_force_ap(scores: list[float], labels: list[int]) -> float:
    """Definition of the tie-grouped step AP: sum over score thresholds."""
    positives = sum(labels)
    previous_recall = 0.0
    result = 0.0
    for threshold in sorted(set(scores), reverse=True):
        selected = [label for score, label in zip(scores, labels, strict=True) if score >= threshold]
        true_positives = sum(selected)
        recall = true_positives / positives
        precision = true_positives / len(selected)
        result += (recall - previous_recall) * precision
        previous_recall = recall
    return result


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_average_precision_and_roc_auc_match_brute_force(seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    # Quantised scores so that ties actually occur and the tie handling is exercised.
    scores = (torch.randint(0, 5, (40,), generator=generator).double() / 4.0).tolist()
    labels = torch.randint(0, 2, (40,), generator=generator).tolist()
    if not any(labels) or all(labels):
        pytest.skip("degenerate label draw")
    assert roc_auc(scores, labels) == pytest.approx(_brute_force_auc(scores, labels))
    assert average_precision(scores, labels) == pytest.approx(_brute_force_ap(scores, labels))


def test_roc_auc_is_half_when_every_score_ties() -> None:
    assert roc_auc([0.5] * 6, [1, 1, 0, 0, 1, 0]) == pytest.approx(0.5)


def test_roc_auc_requires_both_classes() -> None:
    with pytest.raises(ValueError, match="both positive and negative"):
        roc_auc([0.1, 0.2], [1, 1])


def test_average_precision_requires_a_positive() -> None:
    with pytest.raises(ValueError, match="at least one positive"):
        average_precision([0.1, 0.2], [0, 0])


def test_tie_diagnostics_reports_saturation() -> None:
    diagnostics = tie_diagnostics([0.0, 0.0, 0.0, 0.5])
    assert diagnostics["distinct_scores"] == 2
    assert diagnostics["modal_score"] == 0.0
    assert diagnostics["modal_score_fraction"] == pytest.approx(0.75)
    assert diagnostics["exactly_zero_fraction"] == pytest.approx(0.75)
    assert math.isclose(diagnostics["num_scores"], 4.0)
