"""End-to-end suite driver, evaluators and summarizer on the synthetic fixtures.

Everything here runs on CPU against a randomly initialised tiny model, so the
numbers are meaningless; what is checked is that each task type produces the
metric keys the paper's tables read, that the cache and the fused ImageNet path
behave, and that the summarizer lays the tables out correctly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))

from conftest import requires_tokenizer  # noqa: E402
from fixtures.make_eval_fixtures import write_eval_fixtures  # noqa: E402
from hyper3_clip.evaluation.config import load_suite, load_variables  # noqa: E402
from hyper3_clip.evaluation.hierarchy_entailment import (  # noqa: E402
    build_entailment_pairs,
    evaluate_hierarchy_entailment,
    load_hierarchy_entailment_samples,
)
from hyper3_clip.evaluation.reporting import markdown_tables, write_long_csv, write_wide_csv  # noqa: E402
from hyper3_clip.evaluation.runner import EvalRunner, ModelSpec  # noqa: E402

pytestmark = requires_tokenizer

EXPECTED_TASKS = {
    "coco_karpathy_retrieval",
    "flickr30k_retrieval",
    "coco_val2017_retrieval",
    "imagenet_zero_shot",
    "imagenet_hierarchical",
    "cifar10_zero_shot",
    "voc_multilabel",
    "hierarchy_entailment",
}


@pytest.fixture(scope="module")
def eval_fixtures(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    return write_eval_fixtures(tmp_path_factory.mktemp("eval_suite_fixtures"))


@pytest.fixture(scope="module")
def tiny_model(tiny_model_config):
    from hyper3_clip.models import Hyper3CLIP

    torch.manual_seed(0)
    return Hyper3CLIP(tiny_model_config).eval()


@pytest.fixture(scope="module")
def smoke_results(eval_fixtures: dict[str, Path], tiny_model, tmp_path_factory: pytest.TempPathFactory) -> Path:
    suite = load_suite(eval_fixtures["smoke_suite"], load_variables(eval_fixtures["paths_yaml"]))
    output_dir = tmp_path_factory.mktemp("eval_results")
    runner = EvalRunner(suite, output_dir, device="cpu", precision="fp32", batch_size=4, max_items=4)
    runner.run(ModelSpec(id="tiny", checkpoint=str(eval_fixtures["paths_yaml"]), group="fixture"), model=tiny_model)
    return output_dir


def _record(results: Path, task_id: str) -> dict:
    return json.loads((results / "tiny" / f"{task_id}.json").read_text(encoding="utf-8"))


def test_every_task_type_writes_a_record(smoke_results: Path) -> None:
    written = {path.stem for path in (smoke_results / "tiny").glob("*.json")}
    assert written == EXPECTED_TASKS


def test_record_schema_carries_the_cache_key_and_task_spec(smoke_results: Path) -> None:
    record = _record(smoke_results, "coco_karpathy_retrieval")
    assert record["suite"] == "smoke_suite"
    assert record["device"] == "cpu"
    assert record["precision"] == "fp32"
    assert record["model"]["id"] == "tiny"
    assert record["model"]["group"] == "fixture"
    assert record["task"]["name"] == "coco_karpathy_retrieval"
    assert len(record["cache_key"]) == 64


@pytest.mark.parametrize("task_id", ["coco_karpathy_retrieval", "flickr30k_retrieval", "coco_val2017_retrieval"])
def test_retrieval_metrics_are_present_and_in_range(smoke_results: Path, task_id: str) -> None:
    results = _record(smoke_results, task_id)["results"]
    for key in ("i2t_r1", "i2t_r5", "i2t_r10", "t2i_r1", "t2i_r5", "t2i_r10"):
        assert 0.0 <= results[key] <= 100.0
    assert results["i2t_r1"] == pytest.approx(100.0 * results["image_to_text_r1"])
    assert results["t2i_r1"] == pytest.approx(100.0 * results["text_to_image_r1"])
    # Every image has 2 captions and there are 4 images, so R@10 saturates.
    assert results["num_images"] == 4
    assert results["num_captions"] == 8
    assert results["i2t_r10"] == pytest.approx(100.0)


def test_imagenet_zero_shot_and_hierarchy_share_one_pass(smoke_results: Path) -> None:
    zero_shot = _record(smoke_results, "imagenet_zero_shot")
    hierarchy = _record(smoke_results, "imagenet_hierarchical")
    assert zero_shot["elapsed_seconds"] == hierarchy["elapsed_seconds"]
    assert zero_shot["results"]["num_images"] == hierarchy["results"]["num_images"] == 4
    assert 0.0 <= zero_shot["results"]["mean_per_class_acc_pct"] <= 100.0
    for key in ("tie", "lca", "jaccard", "hierarchical_precision", "hierarchical_recall"):
        assert key in hierarchy["results"]
    assert 0.0 <= hierarchy["results"]["jaccard"] <= 1.0
    assert hierarchy["results"]["tie"] >= 0.0


def test_imagefolder_zero_shot_reports_mean_per_class(smoke_results: Path) -> None:
    results = _record(smoke_results, "cifar10_zero_shot")["results"]
    assert results["num_classes"] == 3
    assert results["num_prompts"] == 1  # the Photo regime
    assert 0.0 <= results["mean_per_class_acc_pct"] <= 100.0


def test_multilabel_reports_map(smoke_results: Path) -> None:
    results = _record(smoke_results, "voc_multilabel")["results"]
    assert results["num_classes"] == 3
    assert results["num_samples"] == 4
    # --max-items keeps the manifest's first four rows, which between them carry
    # two of the three classes; a class with no positive is excluded from the mean.
    assert results["valid_classes"] == 2
    assert 0.0 <= results["mean_average_precision"] <= 1.0
    assert results["mean_average_precision_pct"] == pytest.approx(100.0 * results["mean_average_precision"])


def test_hierarchy_entailment_reports_ap_and_auroc(smoke_results: Path) -> None:
    results = _record(smoke_results, "hierarchy_entailment")["results"]
    assert results["num_samples"] == 4
    assert results["num_positive_pairs"] == 12  # 4 images x 3 hierarchy levels
    assert results["num_negative_pairs"] == 12  # capped at 3 per image
    assert 0.0 <= results["auc_roc"] <= 1.0
    assert 0.0 <= results["average_precision"] <= 1.0
    assert results["distinct_scores"] >= 1


def test_a_second_run_hits_the_cache(eval_fixtures: dict[str, Path], tiny_model, smoke_results: Path) -> None:
    before = _record(smoke_results, "voc_multilabel")["created_at"]
    suite = load_suite(eval_fixtures["smoke_suite"], load_variables(eval_fixtures["paths_yaml"]))
    runner = EvalRunner(suite, smoke_results, device="cpu", precision="fp32", batch_size=4, max_items=4)
    runner.run(ModelSpec(id="tiny", checkpoint=str(eval_fixtures["paths_yaml"]), group="fixture"), model=tiny_model)
    assert _record(smoke_results, "voc_multilabel")["created_at"] == before


def test_a_different_max_items_invalidates_the_cache(
    eval_fixtures: dict[str, Path], tiny_model, tmp_path: Path
) -> None:
    suite = load_suite(eval_fixtures["smoke_suite"], load_variables(eval_fixtures["paths_yaml"]))
    spec = ModelSpec(id="tiny", checkpoint=str(eval_fixtures["paths_yaml"]))
    first = EvalRunner(suite, tmp_path, device="cpu", batch_size=4, max_items=4)._cache_key(spec, suite.tasks[0])
    second = EvalRunner(suite, tmp_path, device="cpu", batch_size=4, max_items=2)._cache_key(spec, suite.tasks[0])
    assert first != second


def test_summarizer_writes_csvs_and_the_paper_tables(smoke_results: Path, tmp_path: Path) -> None:
    long_csv = write_long_csv(smoke_results, tmp_path / "summary_long.csv")
    wide_csv = write_wide_csv(smoke_results, tmp_path / "summary_wide.csv")
    long_text = long_csv.read_text(encoding="utf-8")
    assert long_text.splitlines()[0].startswith("suite,model_id,model_group,task_id")
    # The retrieval metric names must survive unmangled.
    assert ",i2t_r5," in long_text
    assert ",t2i_r5," in long_text
    wide_header = wide_csv.read_text(encoding="utf-8").splitlines()[0]
    assert "coco_karpathy_retrieval.i2t_r10" in wide_header
    assert "imagenet_hierarchical.tie" in wide_header

    tables = markdown_tables(smoke_results)
    assert set(tables) == {"table_2", "table_3", "table_4", "table_5"}
    assert "COCO Text retrieval R@5" in tables["table_2"]
    assert "| TIE | LCA | Jaccard | H-P | H-R |" in tables["table_2"]
    assert "Avg." in tables["table_3"]
    assert "Hierarchy AP" in tables["table_5"]
    assert "| tiny |" in tables["table_4"]


def test_prompt_regimes_change_the_classifier(eval_fixtures: dict[str, Path], tiny_model) -> None:
    from hyper3_clip.evaluation.classification import evaluate_imagefolder_zero_shot

    common = {
        "device": torch.device("cpu"),
        "batch_size": 4,
        "num_workers": 0,
        "max_items": 6,
    }
    official = evaluate_imagefolder_zero_shot(
        tiny_model, eval_fixtures["HYPER3_CIFAR10_ROOT"], prompt_set="cifar", prompt_regime="official", **common
    )
    photo = evaluate_imagefolder_zero_shot(
        tiny_model, eval_fixtures["HYPER3_CIFAR10_ROOT"], prompt_set="cifar", prompt_regime="photo", **common
    )
    assert official["num_prompts"] == 18
    assert photo["num_prompts"] == 1
    assert official["num_images"] == photo["num_images"] == 6


def test_negative_construction_is_a_deterministic_prefix(eval_fixtures: dict[str, Path]) -> None:
    samples = load_hierarchy_entailment_samples(
        eval_fixtures["HYPER3_HIERARCAPS_ANNOTATIONS_PATH"],
        image_root=eval_fixtures["HYPER3_HIERARCAPS_IMAGE_ROOT"],
    )
    assert len(samples) == 4
    assert len(samples[0].positive_captions) == 3
    pairs = build_entailment_pairs(samples, max_negatives_per_image=2)
    assert sum(pairs[2]) == 12
    assert len(pairs[2]) == 12 + 8
    # The pool is a prefix in dataset order, so re-running is stable.
    assert pairs == build_entailment_pairs(samples, max_negatives_per_image=2)


def test_unknown_score_raises(eval_fixtures: dict[str, Path], tiny_model) -> None:
    with pytest.raises(ValueError, match="score must be one of"):
        evaluate_hierarchy_entailment(
            tiny_model,
            eval_fixtures["HYPER3_HIERARCAPS_ANNOTATIONS_PATH"],
            torch.device("cpu"),
            score="cosine",
        )


def test_clamped_and_signed_margin_scores_agree_on_the_pair_counts(
    eval_fixtures: dict[str, Path], tiny_model
) -> None:
    common = {
        "image_root": eval_fixtures["HYPER3_HIERARCAPS_IMAGE_ROOT"],
        "device": torch.device("cpu"),
        "batch_size": 4,
        "pair_batch_size": 16,
        "max_negatives_per_image": 3,
    }
    clamped = evaluate_hierarchy_entailment(
        tiny_model, eval_fixtures["HYPER3_HIERARCAPS_ANNOTATIONS_PATH"], score="entailment_score", **common
    )
    margin = evaluate_hierarchy_entailment(
        tiny_model, eval_fixtures["HYPER3_HIERARCAPS_ANNOTATIONS_PATH"], score="signed_margin", **common
    )
    assert clamped["num_pairs"] == margin["num_pairs"] == 24
    # The clamped score lives in [0, 1]; the signed margin is unbounded below.
    assert 0.0 <= clamped["mean_positive_score"] <= 1.0
    assert 0.0 <= clamped["mean_negative_score"] <= 1.0
    assert margin["distinct_scores"] >= clamped["distinct_scores"]
