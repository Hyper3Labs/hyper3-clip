"""WordNet hierarchy metrics on a hand-checked toy tree, plus the shipped assets."""

from __future__ import annotations

import pytest

from hyper3_clip.evaluation.imagenet import WordNetHierarchy, load_wordnet_hierarchy

# A three-level toy tree:
#
#             r
#           /   \
#          a     b
#         / \     \
#       a1   a2    b1
#
# Labels are the three leaves; node ids are a1=0, a2=1, b1=2, a=3, b=4, r=5.
TOY_SYNSETS = ("a1", "a2", "b1")
TOY_ANCESTORS = ([0, 3, 5], [1, 3, 5], [2, 4, 5])
TOY_EDGES = (("r", "a"), ("r", "b"), ("a", "a1"), ("a", "a2"), ("b", "b1"))


@pytest.fixture(scope="module")
def toy_tree() -> WordNetHierarchy:
    return WordNetHierarchy.from_parts(TOY_SYNSETS, TOY_ANCESTORS, TOY_EDGES)


def test_toy_path_lengths(toy_tree: WordNetHierarchy) -> None:
    assert toy_tree.path_length("a1", "a1") == 0
    assert toy_tree.path_length("a1", "a2") == 2  # a1 - a - a2
    assert toy_tree.path_length("a1", "b1") == 4  # a1 - a - r - b - b1


def test_toy_path_length_raises_when_disconnected() -> None:
    tree = WordNetHierarchy.from_parts(("x", "y"), ([0], [1]), (("x", "x2"),))
    with pytest.raises(ValueError, match="No path between synsets"):
        tree.path_length("x", "y")


def test_sibling_confusion_matches_hand_computed_values(toy_tree: WordNetHierarchy) -> None:
    # Predict a2 where a1 is correct: siblings under a.
    # A_p = {1,3,5}, A_t = {0,3,5}, I = {3,5}, U = {0,1,3,5}.
    metrics = toy_tree.metrics([1], [0])
    assert metrics["tie"] == pytest.approx(2.0)
    assert metrics["lca"] == pytest.approx(2.0)  # |A_p| - |I| + 1 = 3 - 2 + 1
    assert metrics["jaccard"] == pytest.approx(0.5)
    assert metrics["hierarchical_precision"] == pytest.approx(2 / 3)
    assert metrics["hierarchical_recall"] == pytest.approx(2 / 3)


def test_cross_subtree_confusion_matches_hand_computed_values(toy_tree: WordNetHierarchy) -> None:
    # Predict b1 where a1 is correct: the only shared ancestor is the root.
    metrics = toy_tree.metrics([2], [0])
    assert metrics["tie"] == pytest.approx(4.0)
    assert metrics["lca"] == pytest.approx(3.0)
    assert metrics["jaccard"] == pytest.approx(0.2)
    assert metrics["hierarchical_precision"] == pytest.approx(1 / 3)
    assert metrics["hierarchical_recall"] == pytest.approx(1 / 3)


def test_correct_prediction_is_the_metric_optimum(toy_tree: WordNetHierarchy) -> None:
    metrics = toy_tree.metrics([0], [0])
    assert metrics["tie"] == pytest.approx(0.0)
    assert metrics["lca"] == pytest.approx(1.0)
    assert metrics["jaccard"] == pytest.approx(1.0)
    assert metrics["hierarchical_precision"] == pytest.approx(1.0)
    assert metrics["hierarchical_recall"] == pytest.approx(1.0)


def test_metrics_average_over_images(toy_tree: WordNetHierarchy) -> None:
    metrics = toy_tree.metrics([1, 2, 0], [0, 0, 0])
    assert metrics["num_images"] == 3
    assert metrics["tie"] == pytest.approx((2 + 4 + 0) / 3)
    assert metrics["lca"] == pytest.approx((2 + 3 + 1) / 3)
    assert metrics["jaccard"] == pytest.approx((0.5 + 0.2 + 1.0) / 3)
    assert metrics["hierarchical_precision"] == pytest.approx((2 / 3 + 1 / 3 + 1.0) / 3)
    assert metrics["hierarchical_recall"] == pytest.approx((2 / 3 + 1 / 3 + 1.0) / 3)


def test_metrics_reject_an_empty_batch(toy_tree: WordNetHierarchy) -> None:
    with pytest.raises(ValueError, match="at least one image"):
        toy_tree.metrics([], [])


def test_shipped_assets_describe_the_1000_imagenet_classes() -> None:
    tree = load_wordnet_hierarchy()
    assert len(tree.synsets) == 1000
    assert len(tree.ancestors) == 1000
    assert all(synset.startswith("n") and len(synset) == 9 for synset in tree.synsets)
    # Every class is reachable from every other one in the is-a graph.
    assert tree.path_length(tree.synsets[0], tree.synsets[999]) > 0
    metrics = tree.metrics([0], [0])
    assert metrics["jaccard"] == pytest.approx(1.0)


def test_hierarcaps_rows_resolve_image_urls_against_the_root(tmp_path) -> None:
    """The public HierarCaps CSV carries ``image_url`` only; the file name joins ``image_root``."""
    from hyper3_clip.evaluation.hierarchy_entailment import load_hierarchy_entailment_samples

    csv_path = tmp_path / "hierarcaps_test.csv"
    csv_path.write_text(
        "id,captions,image_url\n"
        '0,"table => table with plates => a table with three plates",'
        "http://images.cocodataset.org/val2014/COCO_val2014_000000301718.jpg\n",
        encoding="utf-8",
    )
    (sample,) = load_hierarchy_entailment_samples(csv_path, image_root=tmp_path / "val2014")
    assert sample.image_path == tmp_path / "val2014" / "COCO_val2014_000000301718.jpg"
    assert sample.positive_captions == ("table", "table with plates", "a table with three plates")
    assert sample.image_id == "0"
