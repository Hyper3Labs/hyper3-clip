"""Parts-per-image statistics over processed-GRIT shards (paper Sec. 5.3).

The scan walks every shard's tar index, reads each sample's
``numparents.txt``, and accumulates the histogram ``H`` of localized parts per
example.  Everything Sec. 5.3 quotes is arithmetic on that histogram, with
``N = sum_k H[k]``:

======================================  ===========================================
examples                                ``N``
localized parts                         ``sum_k k*H[k]``
mean parts per example                  ``sum_k k*H[k] / N``
max parts                               ``max(k : H[k] > 0)``
fraction with exactly one part          ``H[1] / N``
fraction with one or two parts          ``(H[1] + H[2]) / N``
part instances retained at cap ``m``    ``sum_k min(k, m)*H[k] / sum_k k*H[k]``
examples untruncated at cap ``m``       ``sum_{k <= m} H[k] / N``
======================================  ===========================================

The paper's full-corpus scan reports 2,051 shards, 13,149,251 examples,
25,185,017 parts, mean 1.9153, max 20, 40.68% with one part, 79.23% with one or
two, and at cap 5 99.38% of part instances retained with 99.20% of examples
untruncated.  The training-side cap this sweep varies is
``ProcessedGritDataset(max_parts=...)``, which keeps every part when a group
has at most ``max_parts`` and otherwise draws a random subset of that size.
"""

from __future__ import annotations

import glob
import tarfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

__all__ = [
    "DEFAULT_PART_CAPS",
    "cap_statistics",
    "part_histogram_statistics",
    "scan_grit_parts",
    "shard_part_histogram",
]

#: Per-image part caps the Sec. 5.3 sweep reports, plus the uncapped run.
DEFAULT_PART_CAPS: tuple[int, ...] = (2, 3, 4, 5, 6)

_NUM_PARENTS_SUFFIX = ".numparents.txt"


def shard_part_histogram(shard_path: str | Path) -> Counter[int]:
    """Histogram of parts per example for one shard, read from ``numparents.txt``."""
    histogram: Counter[int] = Counter()
    with tarfile.open(str(shard_path), "r") as tar:
        for member in tar:
            name = member.name.removeprefix("./")
            if not name.endswith(_NUM_PARENTS_SUFFIX):
                continue
            handle = tar.extractfile(member)
            if handle is None:
                raise ValueError(f"{shard_path}: could not read {member.name}")
            text = handle.read().decode("utf-8").strip()
            try:
                histogram[int(text)] += 1
            except ValueError as error:
                raise ValueError(f"{shard_path}:{name}: invalid numparents.txt {text!r}") from error
    return histogram


def part_histogram_statistics(histogram: Mapping[int, int]) -> dict[str, float]:
    """Derive the Sec. 5.3 corpus statistics from a parts histogram."""
    counts = {int(key): int(value) for key, value in histogram.items() if int(value) > 0}
    examples = sum(counts.values())
    if examples == 0:
        raise ValueError("the parts histogram is empty")
    parts = sum(key * value for key, value in counts.items())
    return {
        "example_count": float(examples),
        "part_count": float(parts),
        "mean_parts_per_example": parts / examples,
        "max_parts_per_example": float(max(counts)),
        "fraction_exactly_one_part": counts.get(1, 0) / examples,
        "fraction_one_or_two_parts": (counts.get(1, 0) + counts.get(2, 0)) / examples,
    }


def cap_statistics(
    histogram: Mapping[int, int], caps: Sequence[int] = DEFAULT_PART_CAPS
) -> dict[str, dict[str, float]]:
    """Retention under each per-image cap, plus the uncapped ``"all"`` row.

    ``parts_retained`` is the fraction of part *instances* a cap keeps and
    ``examples_untruncated`` the fraction of examples the cap does not touch.
    """
    counts = {int(key): int(value) for key, value in histogram.items() if int(value) > 0}
    examples = sum(counts.values())
    parts = sum(key * value for key, value in counts.items())
    if examples == 0 or parts == 0:
        raise ValueError("the parts histogram is empty")
    rows: dict[str, dict[str, float]] = {}
    for cap in caps:
        retained = sum(min(key, cap) * value for key, value in counts.items())
        untruncated = sum(value for key, value in counts.items() if key <= cap)
        rows[str(cap)] = {
            "parts_retained": retained / parts,
            "parts_retained_pct": 100.0 * retained / parts,
            "examples_untruncated": untruncated / examples,
            "examples_untruncated_pct": 100.0 * untruncated / examples,
        }
    rows["all"] = {
        "parts_retained": 1.0,
        "parts_retained_pct": 100.0,
        "examples_untruncated": 1.0,
        "examples_untruncated_pct": 100.0,
    }
    return rows


def scan_grit_parts(
    tarfile_patterns: Iterable[str],
    *,
    max_shards: int | None = None,
    caps: Sequence[int] = DEFAULT_PART_CAPS,
) -> dict[str, object]:
    """Scan shards and return the full Sec. 5.3 report.

    Keys: ``shard_count``, ``example_count``, ``part_count``,
    ``mean_parts_per_example``, ``max_parts_per_example``,
    ``fraction_exactly_one_part``, ``fraction_one_or_two_parts``,
    ``parts_histogram`` and ``caps``.
    """
    paths = _expand(tarfile_patterns)
    if max_shards is not None:
        paths = paths[:max_shards]
    if not paths:
        raise FileNotFoundError(f"No tar files matched {list(tarfile_patterns)!r}")
    histogram: Counter[int] = Counter()
    for path in paths:
        histogram.update(shard_part_histogram(path))
    report: dict[str, object] = {"shard_count": len(paths)}
    report.update(part_histogram_statistics(histogram))
    report["parts_histogram"] = dict(sorted(histogram.items()))
    report["caps"] = cap_statistics(histogram, caps)
    return report


def _expand(patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(str(pattern)))
        paths.extend(Path(match) for match in matches)
    return paths
