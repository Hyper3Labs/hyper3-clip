"""The Table 4 manifest builder and the multi-label manifest reader."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from hyper3_clip.evaluation.multilabel import load_multilabel_samples

sys.path.insert(0, str(Path(__file__).parent))

from fixtures.make_eval_fixtures import write_eval_fixtures  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILDER = REPO_ROOT / "scripts" / "build_multilabel_manifest.py"


@pytest.fixture(scope="module")
def fixture_tree(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    return write_eval_fixtures(tmp_path_factory.mktemp("eval_fixtures"))


def _run(*args: str) -> dict[str, object]:
    process = subprocess.run(
        [sys.executable, str(BUILDER), *args],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
    )
    return json.loads(process.stdout.strip().splitlines()[-1])


def test_voc_builder_keeps_raw_label_spellings(fixture_tree: dict[str, Path], tmp_path: Path) -> None:
    manifest = tmp_path / "voc.jsonl"
    classes = tmp_path / "voc_classes.txt"
    report = _run(
        "voc",
        "--annotations-root",
        str(fixture_tree["voc_root"] / "Annotations"),
        "--image-root",
        str(fixture_tree["voc_root"] / "JPEGImages"),
        "--output",
        str(manifest),
        "--class-names-output",
        str(classes),
    )
    assert report["rows"] == 4
    names = classes.read_text(encoding="utf-8").split()
    # The raw VOC XML spelling, which is what produced the paper's numbers.
    assert "tvmonitor" in names
    assert "tv monitor" not in classes.read_text(encoding="utf-8")
    samples = load_multilabel_samples(manifest)
    assert len(samples) == 4
    assert all(sample.subset == "all" for sample in samples)
    assert any("tvmonitor" in sample.labels for sample in samples)


def test_coco_builder_writes_category_id_order(fixture_tree: dict[str, Path], tmp_path: Path) -> None:
    manifest = tmp_path / "coco.jsonl"
    classes = tmp_path / "coco_classes.txt"
    report = _run(
        "coco",
        "--instances-json",
        str(fixture_tree["coco_instances_json"]),
        "--image-root",
        str(fixture_tree["coco_instances_images"]),
        "--output",
        str(manifest),
        "--class-names-output",
        str(classes),
    )
    assert report["rows"] == 4
    assert classes.read_text(encoding="utf-8").splitlines() == ["bicycle", "kitchen table", "sailing boat"]
    samples = load_multilabel_samples(manifest)
    assert [sample.source_id for sample in samples] == ["0", "1", "2", "3"]
    assert all(len(sample.labels) == 1 for sample in samples)


def test_manifest_reader_accepts_delimited_labels_and_a_relative_path(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("image,labels\na.jpg,cat|dog\nb.jpg,\"[\"\"cat\"\"]\"\n", encoding="utf-8")
    samples = load_multilabel_samples(manifest, image_root=tmp_path)
    assert samples[0].labels == ("cat", "dog")
    assert samples[1].labels == ("cat",)
    assert samples[0].image_path == tmp_path / "a.jpg"


def test_manifest_reader_needs_an_image_root_for_relative_paths(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"image_path": "a.jpg", "labels": ["cat"]}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="requires image_root"):
        load_multilabel_samples(manifest)


def test_manifest_reader_rejects_an_unknown_format(tmp_path: Path) -> None:
    path = tmp_path / "manifest.parquet"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported multi-label manifest format"):
        load_multilabel_samples(path)


def test_max_items_truncates_the_manifest(fixture_tree: dict[str, Path]) -> None:
    manifest = fixture_tree["HYPER3_VOC_MULTILABEL_MANIFEST"]
    assert len(load_multilabel_samples(manifest, max_items=2)) == 2
