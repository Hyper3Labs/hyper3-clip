#!/usr/bin/env python3
"""Summarize a results directory into CSVs and the paper's Markdown tables.

    python scripts/summarize_results.py --results-dir results/hyper3-clip

Writes ``summary_long.csv`` (one row per metric), ``summary_wide.csv`` (one row
per model) and ``table_2.md`` ... ``table_6.md`` for whichever tables the
directory has tasks for.  Retrieval columns are the plain ``i2t_r*`` /
``t2i_r*`` names: image-to-text is the paper's "text retrieval".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hyper3_clip.evaluation.reporting import markdown_tables, write_long_csv, write_wide_csv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", required=True, type=Path, help="directory holding <model>/<task>.json")
    parser.add_argument("--output-dir", type=Path, default=None, help="defaults to --results-dir")
    parser.add_argument("--print", dest="print_tables", action="store_true", help="also print the tables")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or args.results_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    written = [
        write_long_csv(args.results_dir, output_dir / "summary_long.csv"),
        write_wide_csv(args.results_dir, output_dir / "summary_wide.csv"),
    ]
    for name, table in markdown_tables(args.results_dir).items():
        path = output_dir / f"{name}.md"
        path.write_text(table, encoding="utf-8")
        written.append(path)
        if args.print_tables:
            print(table)
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
