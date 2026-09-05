#!/usr/bin/env python3
"""Exact parts-per-image scan over processed-GRIT shards (paper Sec. 5.3).

    python scripts/scan_grit_parts.py --tarfiles '/data/grit-processed/*.tar'

Prints a JSON report with the shard count, example count, part count, the mean
and max parts per example, the fraction of examples with exactly one and with
one or two parts, the full histogram, and — for each cap in 2..6 plus the
uncapped run — the fraction of part instances retained and of examples left
untruncated.  The cap is the training-side ``max_parts`` of
``ProcessedGritDataset``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hyper3_clip.evaluation.grit_stats import DEFAULT_PART_CAPS, scan_grit_parts  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tarfiles", nargs="+", required=True, help="shard paths or glob patterns")
    parser.add_argument("--max-shards", type=int, default=None, help="cap the number of shards scanned")
    parser.add_argument("--caps", type=int, nargs="*", default=list(DEFAULT_PART_CAPS))
    parser.add_argument("--output", type=Path, default=None, help="also write the report to this JSON file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = scan_grit_parts(args.tarfiles, max_shards=args.max_shards, caps=args.caps)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
