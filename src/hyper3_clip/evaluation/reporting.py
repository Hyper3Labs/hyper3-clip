"""Turn a directory of task records into CSVs and the paper's tables.

``summary_long.csv`` is one row per numeric metric
(``suite, model_id, model_group, task_id, task_name, dataset, metric, value,
elapsed_seconds, cache_key``); ``summary_wide.csv`` is one row per model with a
``{task_id}.{metric}`` column each.  Metric names are written through
unchanged — the retrieval columns to read are ``i2t_r5`` / ``i2t_r10``
(text retrieval) and ``t2i_r5`` / ``t2i_r10`` (image retrieval).

The Markdown emitters lay each paper table out in its published column order
and only render a table whose tasks are present in the results directory.
Table 4's "Avg." is the unweighted mean of the VOC and COCO mAPs, Table 3's is
the unweighted mean over the 16 datasets, and Table 6 reports both the
three-dataset average and the 16-dataset average recomputed with only those
three columns swapped.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from hyper3_clip.evaluation.prompts import PROMPT_SENSITIVE_DATASETS

__all__ = [
    "TABLE_2_HIERARCHY_COLUMNS",
    "TABLE_2_RETRIEVAL_COLUMNS",
    "TABLE_3_COLUMNS",
    "flatten_records",
    "iter_records",
    "markdown_tables",
    "write_long_csv",
    "write_wide_csv",
]

#: (task id, dataset label, direction label, metric key) for paper Table 2.
TABLE_2_RETRIEVAL_COLUMNS: tuple[tuple[str, str, str, str], ...] = tuple(
    (task_id, dataset, direction, f"{prefix}_r{k}")
    for task_id, dataset in (("coco_karpathy_retrieval", "COCO"), ("flickr30k_retrieval", "Flickr30K"))
    for direction, prefix in (("Text retrieval", "i2t"), ("Image retrieval", "t2i"))
    for k in (1, 5, 10)
)

#: (metric key, column label) for the ImageNet hierarchy block of paper Table 2.
TABLE_2_HIERARCHY_COLUMNS: tuple[tuple[str, str], ...] = (
    ("tie", "TIE"),
    ("lca", "LCA"),
    ("jaccard", "Jaccard"),
    ("hierarchical_precision", "H-P"),
    ("hierarchical_recall", "H-R"),
)

#: (task id, column label) for the 16 columns of paper Table 3, in order.
TABLE_3_COLUMNS: tuple[tuple[str, str], ...] = (
    ("imagenet_zero_shot", "IN"),
    ("cifar10_zero_shot", "C10"),
    ("cifar100_zero_shot", "C100"),
    ("sun397_zero_shot", "SUN"),
    ("caltech101_zero_shot", "Cal"),
    ("stl10_zero_shot", "STL"),
    ("food101_zero_shot", "Food"),
    ("cub_zero_shot", "CUB"),
    ("cars_zero_shot", "Cars"),
    ("aircraft_zero_shot", "Airc"),
    ("pets_zero_shot", "Pets"),
    ("flowers_zero_shot", "Flwr"),
    ("dtd_zero_shot", "DTD"),
    ("eurosat_zero_shot", "ESAT"),
    ("resisc45_zero_shot", "RES"),
    ("country211_zero_shot", "C211"),
)

_TABLE_4_COLUMNS: tuple[tuple[str, str], ...] = (("voc_multilabel", "VOC"), ("coco_multilabel", "COCO"))

#: Table 6 pairs one task id per regime with the Table 3 column it replaces.
_TABLE_6_ROWS: tuple[tuple[str, str, str, str], ...] = (
    ("Food-101", "food101_zero_shot_official", "food101_zero_shot_photo", "food101_zero_shot"),
    ("CUB", "cub_zero_shot_official", "cub_zero_shot_photo", "cub_zero_shot"),
    ("Flowers-102", "flowers_zero_shot_official", "flowers_zero_shot_photo", "flowers_zero_shot"),
)

_LONG_FIELDS = (
    "suite",
    "model_id",
    "model_group",
    "task_id",
    "task_name",
    "dataset",
    "metric",
    "value",
    "elapsed_seconds",
    "cache_key",
)


def iter_records(results_root: str | Path) -> Iterator[dict[str, Any]]:
    """Yield every ``<model>/<task>.json`` record under ``results_root``."""
    for path in sorted(Path(results_root).glob("*/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if {"results", "task", "model"} <= payload.keys():
            yield payload


def flatten_records(records: Iterable[Mapping[str, Any]]) -> Iterator[dict[str, Any]]:
    """Expand records into one row per numeric metric."""
    for record in records:
        model = record["model"]
        task = record["task"]
        for metric, value in record["results"].items():
            if isinstance(value, bool) or not isinstance(value, int | float):
                continue
            yield {
                "suite": record.get("suite", ""),
                "model_id": model["id"],
                "model_group": model.get("group") or "",
                "task_id": task["id"],
                "task_name": task["name"],
                "dataset": task.get("dataset") or task["id"],
                "metric": metric,
                "value": value,
                "elapsed_seconds": record.get("elapsed_seconds", ""),
                "cache_key": record.get("cache_key", ""),
            }


def write_long_csv(results_root: str | Path, output_path: str | Path) -> Path:
    """Write the one-row-per-metric CSV."""
    rows = list(flatten_records(iter_records(results_root)))
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_LONG_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_wide_csv(results_root: str | Path, output_path: str | Path) -> Path:
    """Write the one-row-per-model CSV with ``{task_id}.{metric}`` columns."""
    rows = list(flatten_records(iter_records(results_root)))
    model_ids = sorted({row["model_id"] for row in rows})
    metric_keys = sorted({f"{row['task_id']}.{row['metric']}" for row in rows})
    values = {(row["model_id"], f"{row['task_id']}.{row['metric']}"): row["value"] for row in rows}
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model_id", *metric_keys])
        writer.writeheader()
        for model_id in model_ids:
            writer.writerow(
                {"model_id": model_id, **{key: values.get((model_id, key), "") for key in metric_keys}}
            )
    return path


def _metric_index(results_root: str | Path) -> tuple[list[str], dict[tuple[str, str, str], float]]:
    rows = list(flatten_records(iter_records(results_root)))
    models = sorted({row["model_id"] for row in rows})
    index = {(row["model_id"], row["task_id"], row["metric"]): float(row["value"]) for row in rows}
    return models, index


def _format(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def markdown_tables(results_root: str | Path) -> dict[str, str]:
    """Render every paper table whose tasks are present, keyed ``"table_2"`` etc."""
    models, index = _metric_index(results_root)
    tables: dict[str, str] = {}
    for name, builder in (
        ("table_2", _table_2),
        ("table_3", _table_3),
        ("table_4", _table_4),
        ("table_5", _table_5),
        ("table_6", _table_6),
    ):
        rendered = builder(models, index)
        if rendered is not None:
            tables[name] = rendered
    return tables


def _table_2(models: Sequence[str], index: Mapping[tuple[str, str, str], float]) -> str | None:
    retrieval = [
        column for column in TABLE_2_RETRIEVAL_COLUMNS if any((m, column[0], column[3]) in index for m in models)
    ]
    hierarchy_present = any(
        (m, "imagenet_hierarchical", key) in index for m in models for key, _ in TABLE_2_HIERARCHY_COLUMNS
    )
    if not retrieval and not hierarchy_present:
        return None
    header = ["Model"]
    header += [f"{dataset} {direction} R@{metric.rsplit('_r', 1)[1]}" for _, dataset, direction, metric in retrieval]
    if hierarchy_present:
        header += [label for _, label in TABLE_2_HIERARCHY_COLUMNS]
    rows = []
    for model in models:
        row = [model]
        row += [_format(index.get((model, task_id, metric))) for task_id, _, _, metric in retrieval]
        if hierarchy_present:
            row += [_format(index.get((model, "imagenet_hierarchical", key))) for key, _ in TABLE_2_HIERARCHY_COLUMNS]
        rows.append(row)
    return "### Table 2 — retrieval and ImageNet hierarchy\n\n" + _table(header, rows)


def _table_3(models: Sequence[str], index: Mapping[tuple[str, str, str], float]) -> str | None:
    columns = [
        (task_id, label)
        for task_id, label in TABLE_3_COLUMNS
        if any((model, task_id, "mean_per_class_acc_pct") in index for model in models)
    ]
    if not columns:
        return None
    header = ["Model", *[label for _, label in columns], "Avg."]
    rows = []
    for model in models:
        values = [index.get((model, task_id, "mean_per_class_acc_pct")) for task_id, _ in columns]
        present = [value for value in values if value is not None]
        average = sum(present) / len(present) if present else None
        rows.append([model, *[_format(value) for value in values], _format(average)])
    return "### Table 3 — zero-shot classification (mean-per-class accuracy, %)\n\n" + _table(header, rows)


def _table_4(models: Sequence[str], index: Mapping[tuple[str, str, str], float]) -> str | None:
    columns = [
        (task_id, label)
        for task_id, label in _TABLE_4_COLUMNS
        if any((model, task_id, "mean_average_precision_pct") in index for model in models)
    ]
    if not columns:
        return None
    header = ["Model", *[label for _, label in columns], "Avg."]
    rows = []
    for model in models:
        values = [index.get((model, task_id, "mean_average_precision_pct")) for task_id, _ in columns]
        present = [value for value in values if value is not None]
        average = sum(present) / len(present) if present else None
        rows.append([model, *[_format(value) for value in values], _format(average)])
    return "### Table 4 — zero-shot multi-label mAP (%)\n\n" + _table(header, rows)


def _table_5(models: Sequence[str], index: Mapping[tuple[str, str, str], float]) -> str | None:
    if not any((model, "hierarchy_entailment", "average_precision_pct") in index for model in models):
        return None
    header = ["Model", "Hierarchy AP", "AUROC", "Avg R@10"]
    rows = []
    for model in models:
        present = _recall_at_10_values(model, index)
        rows.append(
            [
                model,
                _format(index.get((model, "hierarchy_entailment", "average_precision_pct"))),
                _format(index.get((model, "hierarchy_entailment", "auc_roc_pct"))),
                _format(sum(present) / len(present) if present else None),
            ]
        )
    return "### Table 5 — hierarchy entailment\n\n" + _table(header, rows)


#: Avg R@10 averages one COCO task and one Flickr task; val2017 is the
#: convention the reported ablation used, so it wins when both are present.
_COCO_RETRIEVAL_TASKS = ("coco_val2017_retrieval", "coco_karpathy_retrieval")
_FLICKR_RETRIEVAL_TASKS = ("flickr30k_retrieval",)


def _recall_at_10_values(model: str, index: Mapping[tuple[str, str, float], float]) -> list[float]:
    """The four R@10 values behind "Avg R@10": one COCO task and one Flickr task."""
    values: list[float] = []
    for group in (_COCO_RETRIEVAL_TASKS, _FLICKR_RETRIEVAL_TASKS):
        for task_id in group:
            found = [
                index[(model, task_id, metric)]
                for metric in ("i2t_r10", "t2i_r10")
                if (model, task_id, metric) in index
            ]
            if found:
                values.extend(found)
                break
    return values


def _table_6(models: Sequence[str], index: Mapping[tuple[str, str, str], float]) -> str | None:
    present_rows = [
        row
        for row in _TABLE_6_ROWS
        if any((model, row[1], "mean_per_class_acc_pct") in index for model in models)
        or any((model, row[2], "mean_per_class_acc_pct") in index for model in models)
    ]
    if not present_rows:
        return None
    header = ["Model", "Regime", *[label for label, _, _, _ in present_rows], "3-dataset Avg.", "16-dataset Avg."]
    other_columns = [
        (task_id, label) for task_id, label in TABLE_3_COLUMNS if task_id not in {row[3] for row in _TABLE_6_ROWS}
    ]
    rows: list[list[str]] = []
    for model in models:
        for regime, position in (("Official", 1), ("Photo", 2)):
            values = [index.get((model, row[position], "mean_per_class_acc_pct")) for row in present_rows]
            present = [value for value in values if value is not None]
            three = sum(present) / len(present) if present else None
            rest = [index.get((model, task_id, "mean_per_class_acc_pct")) for task_id, _ in other_columns]
            rest_present = [value for value in rest if value is not None]
            sixteen = None
            if present and len(rest_present) == len(other_columns):
                combined = rest_present + present
                sixteen = sum(combined) / len(combined)
            rows.append([model, regime, *[_format(value) for value in values], _format(three), _format(sixteen)])
    note = (
        f"\nThe three prompt-sensitive datasets are {', '.join(PROMPT_SENSITIVE_DATASETS)}; the 16-dataset average "
        "re-uses the Table 3 columns for the other 13 datasets and is blank unless they are in the same results "
        "directory.\n"
    )
    return "### Table 6 — prompt sensitivity (mean-per-class accuracy, %)\n\n" + _table(header, rows) + note
