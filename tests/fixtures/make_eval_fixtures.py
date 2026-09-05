"""Write synthetic datasets for every evaluation task, plus a smoke suite.

Nothing here is downloaded: the images are procedurally generated JPEGs and
the annotation files reproduce the real formats (Karpathy split JSON, COCO
captions/instances JSON, ImageFolder trees, a VOC ``Annotations`` directory, a
multi-label JSONL manifest and a HierarCaps-style CSV).  The ImageNet fixture
borrows its WNID directory names from the shipped ``all_synsets.pkl`` so the
WordNet hierarchy assets resolve.

Run directly to materialize a tree and the two YAMLs the driver needs::

    python tests/fixtures/make_eval_fixtures.py /tmp/eval-fixtures --checkpoint

Then::

    python scripts/evaluate.py --checkpoint /tmp/eval-fixtures/checkpoint/model \\
        --suite /tmp/eval-fixtures/smoke_suite.yaml \\
        --paths /tmp/eval-fixtures/paths.yaml \\
        --output-dir /tmp/eval-fixtures/results --device cpu --max-items 4
"""

from __future__ import annotations

import argparse
import io
import json
import pickle
from pathlib import Path
from typing import Any

from PIL import Image

__all__ = [
    "FIXTURE_CLASSES",
    "write_eval_fixtures",
    "write_tiny_checkpoint",
]

#: Class names shared by the ImageFolder and multi-label fixtures.
FIXTURE_CLASSES: tuple[str, ...] = ("bicycle", "kitchen table", "sailing boat")

_CAPTIONS = (
    "a red bicycle leaning against a stone wall",
    "a folded newspaper on a kitchen table",
    "a small sailing boat on calm water",
    "a cup of coffee beside a pair of glasses",
    "two dogs running across a wide green field",
    "a cyclist riding beside a quiet canal",
)


def _jpeg(seed: int, size: int = 48) -> bytes:
    image = Image.new("HSV", (size, size))
    pixels = image.load()
    assert pixels is not None
    for y in range(size):
        for x in range(size):
            pixels[x, y] = ((seed * 37 + x * 3) % 256, 190, 40 + (y * 5 + seed) % 200)
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def _write_image(path: Path, seed: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_jpeg(seed))
    return path


def _karpathy_payload(images: list[tuple[str, str]], split: str) -> dict[str, Any]:
    return {
        "images": [
            {
                "imgid": index,
                "split": split,
                "filepath": filepath,
                "filename": filename,
                "sentences": [
                    {"raw": _CAPTIONS[(index * 2 + offset) % len(_CAPTIONS)], "tokens": []} for offset in range(2)
                ],
            }
            for index, (filepath, filename) in enumerate(images)
        ]
    }


def _write_coco_karpathy(root: Path, count: int) -> None:
    images = []
    for index in range(count):
        filename = f"COCO_val2014_{index:012d}.jpg"
        _write_image(root / "val2014" / filename, index)
        images.append(("val2014", filename))
    (root / "karpathy").mkdir(parents=True, exist_ok=True)
    (root / "karpathy" / "dataset_coco.json").write_text(
        json.dumps(_karpathy_payload(images, "test")), encoding="utf-8"
    )


def _write_flickr30k(root: Path, count: int) -> None:
    payload: dict[str, Any] = {"images": []}
    for index in range(count):
        filename = f"{1000 + index}.jpg"
        _write_image(root / "flickr30k_images" / filename, index + 11)
        payload["images"].append(
            {
                "split": "test",
                "filename": filename,
                "sentences": [
                    {"raw": _CAPTIONS[(index + offset) % len(_CAPTIONS)], "tokens": []} for offset in range(2)
                ],
            }
        )
    root.mkdir(parents=True, exist_ok=True)
    (root / "dataset_flickr30k.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_coco_val2017(root: Path, count: int) -> None:
    images = []
    annotations = []
    for index in range(count):
        filename = f"{index:012d}.jpg"
        _write_image(root / "val2017" / filename, index + 23)
        images.append({"id": index, "file_name": filename})
        for offset in range(2):
            annotations.append(
                {
                    "id": index * 2 + offset,
                    "image_id": index,
                    "caption": _CAPTIONS[(index * 2 + offset) % len(_CAPTIONS)],
                }
            )
    (root / "annotations").mkdir(parents=True, exist_ok=True)
    (root / "annotations" / "captions_val2017.json").write_text(
        json.dumps({"images": images, "annotations": annotations}), encoding="utf-8"
    )


def _write_imagefolder(root: Path, per_class: int) -> None:
    for index, name in enumerate(FIXTURE_CLASSES):
        directory = root / f"{index:04d}_{name.replace(' ', '_')}"
        for item in range(per_class):
            _write_image(directory / f"{item:04d}.jpg", index * 7 + item)


def _write_imagenet_val(root: Path, assets_dir: Path, num_classes: int, per_class: int) -> None:
    with (assets_dir / "all_synsets.pkl").open("rb") as handle:
        synsets = pickle.load(handle)
    for index, wnid in enumerate(synsets[:num_classes]):
        for item in range(per_class):
            _write_image(root / str(wnid) / f"{item:04d}.jpg", index * 13 + item)


def _write_multilabel(root: Path, per_class: int) -> tuple[Path, Path]:
    rows = []
    for index, name in enumerate(FIXTURE_CLASSES):
        for item in range(per_class):
            path = _write_image(root / "images" / f"{index}_{item}.jpg", index * 17 + item)
            labels = [name] if item % 2 == 0 else [name, FIXTURE_CLASSES[(index + 1) % len(FIXTURE_CLASSES)]]
            rows.append({"image_path": str(path), "labels": labels, "subset": "all", "source_id": f"{index}_{item}"})
    manifest = root / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    class_names = root / "classes.txt"
    class_names.write_text("\n".join(FIXTURE_CLASSES) + "\n", encoding="utf-8")
    return manifest, class_names


def _write_hierarcaps(root: Path, count: int) -> tuple[Path, Path]:
    image_root = root / "images"
    lines = ["image_id,image_path,positive_captions"]
    for index in range(count):
        filename = f"hier_{index:04d}.jpg"
        _write_image(image_root / filename, index + 31)
        hierarchy = [
            "an object",
            f"a {FIXTURE_CLASSES[index % len(FIXTURE_CLASSES)]}",
            _CAPTIONS[index % len(_CAPTIONS)],
        ]
        lines.append(f'{index},{filename},"{" => ".join(hierarchy)}"')
    annotations = root / "annotations.csv"
    annotations.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return annotations, image_root


def _write_voc(root: Path, count: int) -> None:
    for index in range(count):
        image_id = f"{index:06d}"
        _write_image(root / "JPEGImages" / f"{image_id}.jpg", index + 41)
        names = [FIXTURE_CLASSES[index % len(FIXTURE_CLASSES)]]
        if index % 2 == 0:
            names.append("tvmonitor")
        objects = "".join(f"<object><name>{name}</name></object>" for name in names)
        annotation = root / "Annotations" / f"{image_id}.xml"
        annotation.parent.mkdir(parents=True, exist_ok=True)
        annotation.write_text(
            f"<annotation><filename>{image_id}.jpg</filename>{objects}</annotation>", encoding="utf-8"
        )


def _write_coco_instances(root: Path, count: int) -> Path:
    categories = [{"id": index + 1, "name": name} for index, name in enumerate(FIXTURE_CLASSES)]
    images = []
    annotations = []
    for index in range(count):
        filename = f"{index:012d}.jpg"
        _write_image(root / "images" / filename, index + 53)
        images.append({"id": index, "file_name": filename})
        annotations.append({"id": index, "image_id": index, "category_id": (index % len(FIXTURE_CLASSES)) + 1})
    path = root / "instances.json"
    path.write_text(
        json.dumps({"categories": categories, "images": images, "annotations": annotations}), encoding="utf-8"
    )
    return path


_SMOKE_SUITE = """# Synthetic smoke suite: one task of every kind the runner dispatches.
name: smoke_suite
defaults:
  batch_size: 4
  num_workers: 0
  pair_batch_size: 64
  precision: fp32
tasks:
  - id: coco_karpathy_retrieval
    name: coco_karpathy_retrieval
    data:
      dataset: COCO Karpathy (fixture)
      coco2014_root: ${HYPER3_COCO2014_ROOT}
      coco_karpathy_split: test

  - id: flickr30k_retrieval
    name: flickr30k_retrieval
    data:
      dataset: Flickr30K (fixture)
      flickr30k_root: ${HYPER3_FLICKR30K_ROOT}
      flickr30k_split: test

  - id: coco_val2017_retrieval
    name: coco_val2017_retrieval
    data:
      dataset: COCO val2017 (fixture)
      coco2017_root: ${HYPER3_COCO2017_ROOT}

  - id: imagenet_zero_shot
    name: imagenet_zero_shot
    data:
      dataset: ImageNet (fixture)
      imagenet_val_root: ${HYPER3_IMAGENET_VAL_ROOT}

  - id: imagenet_hierarchical
    name: imagenet_hierarchical
    data:
      dataset: ImageNet-WordNet (fixture)
      imagenet_val_root: ${HYPER3_IMAGENET_VAL_ROOT}

  - id: cifar10_zero_shot
    name: imagefolder_zero_shot
    data:
      dataset: ImageFolder (fixture)
      root: ${HYPER3_CIFAR10_ROOT}
      prompt_regime: photo

  - id: voc_multilabel
    name: multilabel_zero_shot
    data:
      dataset: VOC (fixture)
      manifest_path: ${HYPER3_VOC_MULTILABEL_MANIFEST}
      class_names_path: ${HYPER3_VOC_CLASS_NAMES_PATH}
      prompts:
        - "a photo of a {}."

  - id: hierarchy_entailment
    name: hierarchy_entailment
    data:
      dataset: HierarCaps (fixture)
      annotations_path: ${HYPER3_HIERARCAPS_ANNOTATIONS_PATH}
      image_root: ${HYPER3_HIERARCAPS_IMAGE_ROOT}
      max_negatives_per_image: 3
"""


def write_eval_fixtures(root: str | Path, *, count: int = 4) -> dict[str, Path]:
    """Materialize every dataset fixture plus ``paths.yaml`` and ``smoke_suite.yaml``.

    Returns a mapping of ``${HYPER3_*}`` variable names to paths, with the two
    YAML files under the keys ``paths_yaml`` and ``smoke_suite``.
    """
    from hyper3_clip.evaluation.class_names import ASSETS_DIR

    base = Path(root)
    base.mkdir(parents=True, exist_ok=True)
    _write_coco_karpathy(base / "coco2014", count)
    _write_flickr30k(base / "flickr30k", count)
    _write_coco_val2017(base / "coco2017", count)
    _write_imagefolder(base / "imagefolder", count)
    _write_imagenet_val(base / "imagenet_val", ASSETS_DIR, num_classes=3, per_class=2)
    manifest, class_names = _write_multilabel(base / "multilabel", count)
    annotations, image_root = _write_hierarcaps(base / "hierarcaps", count)
    _write_voc(base / "voc", count)
    _write_coco_instances(base / "coco_instances", count)

    variables = {
        "HYPER3_COCO2014_ROOT": base / "coco2014",
        "HYPER3_FLICKR30K_ROOT": base / "flickr30k",
        "HYPER3_COCO2017_ROOT": base / "coco2017",
        "HYPER3_IMAGENET_VAL_ROOT": base / "imagenet_val",
        "HYPER3_CIFAR10_ROOT": base / "imagefolder",
        "HYPER3_VOC_MULTILABEL_MANIFEST": manifest,
        "HYPER3_VOC_CLASS_NAMES_PATH": class_names,
        "HYPER3_HIERARCAPS_ANNOTATIONS_PATH": annotations,
        "HYPER3_HIERARCAPS_IMAGE_ROOT": image_root,
    }
    paths_yaml = base / "paths.yaml"
    paths_yaml.write_text("\n".join(f"{key}: {value}" for key, value in variables.items()) + "\n", encoding="utf-8")
    suite_yaml = base / "smoke_suite.yaml"
    suite_yaml.write_text(_SMOKE_SUITE, encoding="utf-8")

    result = dict(variables)
    result["voc_root"] = base / "voc"
    result["coco_instances_json"] = base / "coco_instances" / "instances.json"
    result["coco_instances_images"] = base / "coco_instances" / "images"
    result["paths_yaml"] = paths_yaml
    result["smoke_suite"] = suite_yaml
    return result


def write_tiny_checkpoint(directory: str | Path) -> Path:
    """Save a randomly initialised tiny Hyper3-CLIP and return the model directory.

    The text tower is a 2-layer, 64-dim CLIP text config with the real CLIP
    tokenizer saved beside it, so the checkpoint loads with no network.
    """
    from transformers import AutoTokenizer, CLIPTextConfig

    from hyper3_clip.models import Hyper3CLIP, Hyper3CLIPConfig

    base = Path(directory)
    text_dir = base / "text_tower"
    text_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer.save_pretrained(text_dir)
    CLIPTextConfig(
        vocab_size=int(tokenizer.vocab_size),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=77,
    ).save_pretrained(text_dir)
    config = Hyper3CLIPConfig(
        vision_backbone="vit_tiny_patch16_224",
        text_model_name=str(text_dir),
        embed_dim=32,
    )
    model = Hyper3CLIP(config)
    return model.save_pretrained(base / "model")


def main() -> int:
    """CLI entry point: ``make_eval_fixtures.py <root> [--checkpoint]``."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path)
    parser.add_argument("--count", type=int, default=4, help="images per fixture dataset")
    parser.add_argument("--checkpoint", action="store_true", help="also save a tiny random checkpoint")
    args = parser.parse_args()
    written = write_eval_fixtures(args.root, count=args.count)
    if args.checkpoint:
        written["checkpoint"] = write_tiny_checkpoint(args.root / "checkpoint")
    print(json.dumps({key: str(value) for key, value in written.items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
