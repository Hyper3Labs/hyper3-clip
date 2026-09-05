"""The Sec. 5.3 parts-per-image scan, on the GRIT fixture shard and on histograms."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyper3_clip.evaluation.grit_stats import (
    cap_statistics,
    part_histogram_statistics,
    scan_grit_parts,
    shard_part_histogram,
)

# The fixture shard has groups of 3, 2, 6 and 1 parts.
FIXTURE_HISTOGRAM = {1: 1, 2: 1, 3: 1, 6: 1}


def test_scan_matches_the_fixture_shard(grit_shard: Path) -> None:
    assert shard_part_histogram(grit_shard) == FIXTURE_HISTOGRAM
    report = scan_grit_parts([str(grit_shard)])
    assert report["shard_count"] == 1
    assert report["example_count"] == 4
    assert report["part_count"] == 12  # 3 + 2 + 6 + 1
    assert report["mean_parts_per_example"] == pytest.approx(3.0)
    assert report["max_parts_per_example"] == 6
    assert report["fraction_exactly_one_part"] == pytest.approx(0.25)
    assert report["fraction_one_or_two_parts"] == pytest.approx(0.5)
    assert report["parts_histogram"] == FIXTURE_HISTOGRAM


def test_cap_rows_cover_the_sweep_and_the_uncapped_run(grit_shard: Path) -> None:
    report = scan_grit_parts([str(grit_shard)])
    caps = report["caps"]
    assert sorted(caps) == ["2", "3", "4", "5", "6", "all"]
    # At cap 2: min(k,2) over {1,2,3,6} = 1+2+2+2 = 7 of 12 part instances,
    # and only the groups with 1 and 2 parts are left untruncated.
    assert caps["2"]["parts_retained"] == pytest.approx(7 / 12)
    assert caps["2"]["examples_untruncated"] == pytest.approx(0.5)
    assert caps["6"]["parts_retained"] == pytest.approx(1.0)
    assert caps["all"]["examples_untruncated_pct"] == pytest.approx(100.0)


def test_glob_patterns_expand_and_max_shards_truncates(grit_shard: Path, tmp_path: Path) -> None:
    second = tmp_path / "shard-000001.tar"
    second.write_bytes(grit_shard.read_bytes())
    pattern = str(tmp_path / "*.tar")
    assert scan_grit_parts([pattern])["shard_count"] == 1
    both = scan_grit_parts([str(grit_shard), pattern])
    assert both["shard_count"] == 2
    assert both["example_count"] == 8
    assert scan_grit_parts([str(grit_shard), pattern], max_shards=1)["shard_count"] == 1


def test_missing_shards_raise() -> None:
    with pytest.raises(FileNotFoundError, match="No tar files matched"):
        scan_grit_parts(["/nonexistent/*.tar"])


def test_paper_statistics_are_reproduced_from_a_histogram() -> None:
    # A histogram with the paper's corpus totals: 13,149,251 examples and
    # 25,185,017 parts give mean 1.9153 and the quoted retention at cap 5.
    histogram = {1: 5_349_137, 2: 5_070_663, 3: 1_600_000, 4: 900_000, 5: 200_000, 20: 29_451}
    statistics = part_histogram_statistics(histogram)
    assert statistics["max_parts_per_example"] == 20
    assert statistics["fraction_exactly_one_part"] == pytest.approx(
        histogram[1] / statistics["example_count"]
    )
    assert statistics["fraction_one_or_two_parts"] == pytest.approx(
        (histogram[1] + histogram[2]) / statistics["example_count"]
    )
    caps = cap_statistics(histogram)
    expected_retained = sum(min(k, 5) * v for k, v in histogram.items()) / sum(k * v for k, v in histogram.items())
    assert caps["5"]["parts_retained"] == pytest.approx(expected_retained)
    untruncated = sum(v for k, v in histogram.items() if k <= 5) / sum(histogram.values())
    assert caps["5"]["examples_untruncated"] == pytest.approx(untruncated)


def test_empty_histograms_raise() -> None:
    with pytest.raises(ValueError, match="histogram is empty"):
        part_histogram_statistics({})
    with pytest.raises(ValueError, match="histogram is empty"):
        cap_statistics({0: 5})
