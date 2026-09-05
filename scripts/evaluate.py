#!/usr/bin/env python3
"""Run one evaluation suite against one checkpoint.

    python scripts/evaluate.py \
        --checkpoint hyper3labs/hyper3-clip \
        --suite configs/eval/all_paper_tables.yaml \
        --paths configs/eval/local_paths.yaml \
        --output-dir results/hyper3-clip

Writes ``<output-dir>/<model-id>/<task-id>.json``, one record per task, and
skips any task whose cache key already matches (pass ``--force`` to re-run).
``--max-items N`` truncates every dataset, which is what the repository's own
smoke test uses.  Summarize a finished directory with
``scripts/summarize_results.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hyper3_clip.evaluation.config import filter_tasks, load_suite, load_variables  # noqa: E402
from hyper3_clip.evaluation.runner import EvalRunner, ModelSpec  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="checkpoint directory, .pt file, or Hugging Face repo id")
    parser.add_argument("--suite", required=True, type=Path, help="suite YAML under configs/eval/")
    parser.add_argument("--paths", type=Path, default=None, help="YAML of ${VAR} dataset locations")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-id", default=None, help="name used in the output path and the records")
    parser.add_argument("--model-group", default=None, help="optional grouping label carried into the records")
    parser.add_argument("--tasks", nargs="*", default=None, help="restrict the run to these task ids")
    parser.add_argument("--device", default=None, help="cuda, cpu, mps; defaults to cuda when available")
    parser.add_argument("--batch-size", type=int, default=None, help="override the suite's batch size")
    parser.add_argument("--precision", default="fp32", choices=("fp32", "bf16", "fp16"))
    parser.add_argument("--max-items", type=int, default=None, help="truncate every dataset (smoke runs)")
    parser.add_argument("--force", action="store_true", help="ignore cached task records")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    suite = filter_tasks(load_suite(args.suite, load_variables(args.paths)), args.tasks)
    runner = EvalRunner(
        suite,
        args.output_dir,
        device=args.device,
        precision=args.precision,
        batch_size=args.batch_size,
        max_items=args.max_items,
        force=args.force,
    )
    model_id = args.model_id or Path(str(args.checkpoint)).name or str(args.checkpoint)
    written = runner.run(ModelSpec(id=model_id, checkpoint=str(args.checkpoint), group=args.model_group))
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
