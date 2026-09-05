"""Prompt-regime resolution, class-name resolution and the shipped suite configs."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hyper3_clip.evaluation.class_names import (
    load_class_name_list,
    load_class_name_table,
    resolve_class_names,
)
from hyper3_clip.evaluation.config import load_suite, resolve_templates
from hyper3_clip.evaluation.prompts import (
    IMAGENET_PROMPTS,
    PHOTO_PROMPTS,
    PROMPT_SENSITIVE_DATASETS,
    PROMPT_SETS,
    resolve_prompts,
)
from hyper3_clip.evaluation.runner import TASK_NAMES

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "eval"


def test_photo_regime_overrides_the_official_prompt_set() -> None:
    assert resolve_prompts(prompt_set="food101", prompt_regime="photo") == PHOTO_PROMPTS
    assert resolve_prompts(prompt_set="food101", prompt_regime="official") == PROMPT_SETS["food101"]


def test_explicit_prompts_win_over_the_regime() -> None:
    assert resolve_prompts(prompts=["a {}."], prompt_set="cifar", prompt_regime="photo") == ("a {}.",)


def test_default_prompt_source_is_the_imagenet_ensemble() -> None:
    assert resolve_prompts() == IMAGENET_PROMPTS
    assert resolve_prompts(prompt_regime="official") == IMAGENET_PROMPTS


def test_unknown_regime_and_prompt_set_raise() -> None:
    with pytest.raises(ValueError, match="prompt_regime must be"):
        resolve_prompts(prompt_regime="Photo")
    with pytest.raises(ValueError, match="Unknown prompt_set"):
        resolve_prompts(prompt_set="nope")


def test_the_three_prompt_sensitive_datasets_have_official_sets() -> None:
    for key in PROMPT_SENSITIVE_DATASETS:
        assert key in PROMPT_SETS
        assert PROMPT_SETS[key] != PHOTO_PROMPTS


def test_class_names_resolve_from_the_vendored_table() -> None:
    names = resolve_class_names(["0000_a"] * 101, class_names_key="food101")
    assert len(names) == 101
    assert names[0] == "apple pie"


def test_class_name_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="Resolved 101 class names for 3"):
        resolve_class_names(["a", "b", "c"], class_names_key="food101")


def test_unknown_class_names_key_raises() -> None:
    with pytest.raises(ValueError, match="Unknown class_names_key"):
        resolve_class_names(["a"], class_names_key="not-a-dataset")


def test_folder_names_are_the_fallback() -> None:
    assert resolve_class_names(["cat", "dog"]) == ["cat", "dog"]


def test_vendored_table_has_the_class_counts_the_tables_expect() -> None:
    table = load_class_name_table()
    assert len(table["imagenet"]) == 1000
    for key, count in (("food101", 101), ("cub2011", 200), ("flowers102", 102), ("cifar100", 100)):
        assert len(table[key]) == count


def test_class_name_list_reads_a_txt_in_file_order(tmp_path: Path) -> None:
    path = tmp_path / "classes.txt"
    path.write_text("zebra\n\napple\n", encoding="utf-8")
    assert load_class_name_list(path) == ["zebra", "apple"]
    assert load_class_name_list(None) is None
    assert load_class_name_list(None, ["a", "b"]) == ["a", "b"]


def test_resolve_templates_raises_for_a_missing_variable() -> None:
    with pytest.raises(KeyError, match="MISSING"):
        resolve_templates({"root": "${MISSING}"}, {})


@pytest.mark.parametrize(
    "suite_name",
    [
        "paper_table_2",
        "paper_table_3",
        "paper_table_4",
        "hierarchy_entailment",
        "prompt_sensitivity",
        "all_paper_tables",
    ],
)
def test_shipped_suites_load_with_the_example_paths(suite_name: str) -> None:
    variables = yaml.safe_load((CONFIG_DIR / "local_paths.example.yaml").read_text(encoding="utf-8"))
    suite = load_suite(CONFIG_DIR / f"{suite_name}.yaml", {key: str(value) for key, value in variables.items()})
    assert suite.name == suite_name
    assert suite.tasks
    assert len({task.id for task in suite.tasks}) == len(suite.tasks)
    for task in suite.tasks:
        assert task.name in TASK_NAMES
        assert "${" not in repr(task.data)


def test_table_3_uses_the_photo_regime_for_exactly_three_datasets() -> None:
    variables = yaml.safe_load((CONFIG_DIR / "local_paths.example.yaml").read_text(encoding="utf-8"))
    suite = load_suite(CONFIG_DIR / "paper_table_3.yaml", {key: str(value) for key, value in variables.items()})
    photo = {task.data["class_names_key"] for task in suite.tasks if task.data.get("prompt_regime") == "photo"}
    assert photo == set(PROMPT_SENSITIVE_DATASETS)
    assert len(suite.tasks) == 16


def test_prompt_sensitivity_suite_runs_both_regimes() -> None:
    variables = yaml.safe_load((CONFIG_DIR / "local_paths.example.yaml").read_text(encoding="utf-8"))
    suite = load_suite(CONFIG_DIR / "prompt_sensitivity.yaml", {key: str(value) for key, value in variables.items()})
    assert len(suite.tasks) == 2 * len(PROMPT_SENSITIVE_DATASETS)
    regimes = sorted({task.data["prompt_regime"] for task in suite.tasks})
    assert regimes == ["official", "photo"]


def test_example_paths_file_holds_only_placeholders() -> None:
    text = (CONFIG_DIR / "local_paths.example.yaml").read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("HYPER3_"):
            assert line.split(": ", 1)[1].startswith("/path/to/")
