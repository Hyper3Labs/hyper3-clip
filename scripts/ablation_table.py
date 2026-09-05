#!/usr/bin/env python3
"""Print the paper's Table 5 ablation table from finished results directories.

    python scripts/ablation_table.py \
        --baseline results/base_80k \
        --results results/full_objective results/no_query results/text_hierarchy_only

Each directory is one run's ``<model>/<task>.json`` tree (the output of
``scripts/evaluate.py --suite configs/eval/hierarchy_entailment.yaml``).  The
columns are the paper's:

============  =============================================================
Variant       the directory name, or ``--labels`` in the same order
Hierarchy AP  ``hierarchy_entailment.average_precision_pct``
AUROC         ``hierarchy_entailment.auc_roc_pct``
Avg R@10      mean of the four COCO / Flickr R@10 values
dAP           Hierarchy AP minus the baseline's
dR@10         Avg R@10 minus the baseline's
============  =============================================================

"Avg R@10" is the unweighted mean of ``i2t_r10`` and ``t2i_r10`` on the two
retrieval tasks in the directory.  The reported ablation used COCO **val2017**
here rather than the Table 2 Karpathy split (see ``docs/evaluation.md``);
whichever COCO task the directory holds is the one averaged.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hyper3_clip.evaluation.reporting import flatten_records, iter_records  # noqa: E402

_COCO_TASKS = ("coco_val2017_retrieval", "coco_karpathy_retrieval")
_FLICKR_TASKS = ("flickr30k_retrieval",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", nargs="+", required=True, type=Path, help="one results directory per row")
    parser.add_argument("--baseline", type=Path, default=None, help="results directory the deltas are taken against")
    parser.add_argument("--labels", nargs="*", default=None, help="row labels, in the order of --results")
    parser.add_argument("--baseline-label", default=None)
    return parser.parse_args()


def _metrics(results_dir: Path) -> dict[tuple[str, str], float]:
    return {
        (row["task_id"], row["metric"]): float(row["value"])
        for row in flatten_records(iter_records(results_dir))
    }


def _avg_recall_at_10(metrics: dict[tuple[str, str], float]) -> float | None:
    """Mean of four R@10 values: one COCO task and one Flickr task.

    ``coco_val2017_retrieval`` wins over ``coco_karpathy_retrieval`` when a
    directory holds both, because val2017 is the convention the reported
    ablation used.
    """
    values: list[float] = []
    for group in (_COCO_TASKS, _FLICKR_TASKS):
        for task_id in group:
            found = [metrics[(task_id, metric)] for metric in ("i2t_r10", "t2i_r10") if (task_id, metric) in metrics]
            if found:
                values.extend(found)
                break
    return sum(values) / len(values) if values else None


def _row(label: str, metrics: dict[tuple[str, str], float], baseline: dict[str, float | None]) -> list[str]:
    ap = metrics.get(("hierarchy_entailment", "average_precision_pct"))
    auroc = metrics.get(("hierarchy_entailment", "auc_roc_pct"))
    recall = _avg_recall_at_10(metrics)
    delta_ap = None if ap is None or baseline["ap"] is None else ap - baseline["ap"]
    delta_recall = None if recall is None or baseline["recall"] is None else recall - baseline["recall"]
    return [
        label,
        _fmt(ap),
        _fmt(auroc),
        _fmt(recall),
        _fmt(delta_ap, signed=True),
        _fmt(delta_recall, signed=True),
    ]


def _fmt(value: float | None, signed: bool = False) -> str:
    if value is None:
        return "-"
    return f"{value:+.2f}" if signed else f"{value:.2f}"


def main() -> int:
    args = parse_args()
    baseline_metrics = _metrics(args.baseline) if args.baseline is not None else {}
    baseline = {
        "ap": baseline_metrics.get(("hierarchy_entailment", "average_precision_pct")),
        "recall": _avg_recall_at_10(baseline_metrics) if baseline_metrics else None,
    }

    header = ["Variant", "Hierarchy AP", "AUROC", "Avg R@10", "dAP", "dR@10"]
    rows: list[list[str]] = []
    if args.baseline is not None:
        rows.append(_row(args.baseline_label or args.baseline.name, baseline_metrics, baseline))
    labels = args.labels or [path.name for path in args.results]
    if len(labels) != len(args.results):
        raise SystemExit("--labels must have one entry per --results directory")
    for label, path in zip(labels, args.results, strict=True):
        rows.append(_row(label, _metrics(path), baseline))

    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join("---" for _ in header) + " |")
    for row in rows:
        print("| " + " | ".join(row) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
