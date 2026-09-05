#!/usr/bin/env python
"""Launch a Hyper3-CLIP training run.

    torchrun --nproc_per_node=8 scripts/train.py \
        --config configs/paper/hyper3_clip_vitb_500k.yaml \
        --override data.tarfiles='["/data/grit/shard-{000000..000999}.tar"]'

``--override`` takes dotted paths into the config tree and YAML-typed values;
it may be repeated.  The resolved config is written to
``<output_dir>/config.yaml`` before the first step.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hyper3_clip.training import load_run_config, run_training  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse ``--config`` and any number of ``--override key.path=value``."""
    parser = argparse.ArgumentParser(description="Train Hyper3-CLIP")
    parser.add_argument("--config", required=True, help="path to a YAML run config")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="dotted config override, repeatable (e.g. training.total_steps=100000)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Load the config, apply overrides and run training."""
    args = parse_args(argv)
    config = load_run_config(args.config, args.override)
    run_training(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
