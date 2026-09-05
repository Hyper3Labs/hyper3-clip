#!/usr/bin/env python3
"""Build the VOC / COCO multi-label manifests for paper Table 4.

    # PASCAL VOC, from the XML annotations
    python scripts/build_multilabel_manifest.py voc \
        --annotations-root /data/VOC2007/Annotations \
        --image-root /data/VOC2007/JPEGImages \
        --output /data/manifests/voc_multilabel.jsonl \
        --class-names-output /data/manifests/voc_classes.txt

    # COCO, from instances_val2017.json
    python scripts/build_multilabel_manifest.py coco \
        --instances-json /data/coco2017/annotations/instances_val2017.json \
        --image-root /data/coco2017/val2017 \
        --output /data/manifests/coco_multilabel.jsonl \
        --class-names-output /data/manifests/coco_classes.txt

Each row is ``{"image_path", "labels", "subset": "all", "source_id"}``.  Labels
are the raw strings the source files spell — VOC's ``diningtable``,
``pottedplant``, ``tvmonitor``, ``boat`` — which is what produced the paper's
numbers.  Images with no annotated object are absent from the manifest, since
the builders key off the annotations; both points are documented in
``docs/evaluation.md``.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="source", required=True)

    voc = sub.add_parser("voc", help="build from PASCAL VOC XML annotations")
    voc.add_argument("--annotations-root", required=True, type=Path)
    voc.add_argument("--image-root", required=True, type=Path)
    voc.add_argument("--image-set-file", type=Path, default=None, help="optional ImageSets/Main/*.txt of image ids")

    coco = sub.add_parser("coco", help="build from a COCO instances_*.json")
    coco.add_argument("--instances-json", required=True, type=Path)
    coco.add_argument("--image-root", required=True, type=Path)

    for group in (voc, coco):
        group.add_argument("--output", required=True, type=Path)
        group.add_argument("--class-names-output", type=Path, default=None)
        group.add_argument("--skip-missing-images", action="store_true")
    return parser.parse_args()


def _voc_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str], int]:
    if args.image_set_file is not None:
        image_ids = [
            line.split()[0]
            for line in args.image_set_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    else:
        image_ids = sorted(path.stem for path in args.annotations_root.glob("*.xml"))

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    skipped = 0
    for image_id in image_ids:
        annotation_path = args.annotations_root / f"{image_id}.xml"
        if not annotation_path.exists():
            raise FileNotFoundError(f"Missing VOC annotation for {image_id}: {annotation_path}")
        labels = {
            name.strip()
            for obj in ET.parse(annotation_path).getroot().findall("object")
            if (name := obj.findtext("name"))
        }
        if not labels:
            continue
        seen.update(labels)
        image_path = args.image_root / f"{image_id}.jpg"
        if not image_path.exists():
            if args.skip_missing_images:
                skipped += 1
                continue
            raise FileNotFoundError(f"Missing VOC image for {image_id}: {image_path}")
        rows.append(
            {
                "image_path": str(image_path),
                "labels": sorted(labels),
                "subset": "all",
                "source_id": image_id,
            }
        )
    return rows, sorted(seen), skipped


def _coco_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str], int]:
    payload = json.loads(args.instances_json.read_text(encoding="utf-8"))
    categories = {int(item["id"]): str(item["name"]) for item in payload["categories"]}
    images = {int(item["id"]): item for item in payload["images"]}
    labels_by_image: dict[int, set[str]] = defaultdict(set)
    for annotation in payload["annotations"]:
        labels_by_image[int(annotation["image_id"])].add(categories[int(annotation["category_id"])])

    image_ids = sorted(labels_by_image)
    rows: list[dict[str, Any]] = []
    skipped = 0
    for image_id in image_ids:
        image_path = args.image_root / str(images[image_id]["file_name"])
        if not image_path.exists():
            if args.skip_missing_images:
                skipped += 1
                continue
            raise FileNotFoundError(f"Missing COCO image for image_id={image_id}: {image_path}")
        rows.append(
            {
                "image_path": str(image_path),
                "labels": sorted(labels_by_image.get(image_id, set())),
                "subset": "all",
                "source_id": str(image_id),
            }
        )
    return rows, [categories[key] for key in sorted(categories)], skipped


def main() -> int:
    args = parse_args()
    rows, class_names, skipped = _voc_rows(args) if args.source == "voc" else _coco_rows(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if args.class_names_output is not None:
        args.class_names_output.parent.mkdir(parents=True, exist_ok=True)
        args.class_names_output.write_text("\n".join(class_names) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "rows": len(rows),
                "classes": len(class_names),
                "skipped_missing_images": skipped,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
