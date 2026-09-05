"""Write a tiny processed-GRIT tar shard for tests and for local smoke runs.

The shard has the same member layout as the real HyCoCLIP-processed GRIT
shards (``child.jpg`` / ``child.txt`` / ``numparents.txt`` /
``parentNNN.jpg`` / ``parentNNN.txt``), with procedurally generated images so
nothing has to be downloaded.  The four captions between them exercise
multi-sentence splitting, connector-word phrase extraction, a duplicate box
text and a group with more parts than the paper's cap of five.

Run directly to write a shard::

    python tests/fixtures/make_grit_fixture.py /tmp/grit-fixture-000000.tar
"""

from __future__ import annotations

import argparse
import io
import tarfile
from pathlib import Path

from PIL import Image

__all__ = ["FIXTURE_SAMPLES", "write_grit_fixture"]

#: ``(key, caption, [(box_text, hue), ...])`` for each image group.
FIXTURE_SAMPLES: tuple[tuple[str, str, tuple[tuple[str, int], ...]], ...] = (
    (
        "sample0000",
        "A red bicycle leans against a stone wall. A wooden door stands behind it.",
        (("a red bicycle", 0), ("a stone wall", 40), ("a wooden door", 80)),
    ),
    (
        "sample0001",
        "Two dogs run across a wide green field with a wooden fence near the treeline.",
        (("two dogs", 120), ("a wide green field", 160)),
    ),
    (
        "sample0002",
        "A cup of coffee, a folded newspaper and a pair of glasses sit on a kitchen table.",
        (
            ("a cup of coffee", 200),
            ("a folded newspaper", 20),
            ("a pair of glasses", 60),
            ("a kitchen table", 100),
            ("a cup of coffee", 140),
            ("a bright window", 180),
        ),
    ),
    (
        "sample0003",
        "A cyclist rides beside a canal; sunlight falls on the water.",
        (("a cyclist", 220),),
    ),
)


def _jpeg(hue: int, size: int = 64) -> bytes:
    """Render a small deterministic JPEG with a visible gradient."""
    image = Image.new("HSV", (size, size))
    pixels = image.load()
    assert pixels is not None
    for y in range(size):
        for x in range(size):
            pixels[x, y] = ((hue + x) % 256, 180, 60 + (y * 3) % 190)
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=92)
    return buffer.getvalue()


def _add(tar: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    tar.addfile(info, io.BytesIO(payload))


def write_grit_fixture(path: str | Path, image_size: int = 64) -> Path:
    """Write the fixture shard to ``path`` and return it."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(target, "w") as tar:
        for key, caption, boxes in FIXTURE_SAMPLES:
            _add(tar, f"{key}.child.jpg", _jpeg(hash(key) % 256, image_size))
            _add(tar, f"{key}.child.txt", caption.encode("utf-8"))
            _add(tar, f"{key}.numparents.txt", str(len(boxes)).encode("ascii"))
            for index, (text, hue) in enumerate(boxes):
                _add(tar, f"{key}.parent{index:03d}.jpg", _jpeg(hue, image_size))
                _add(tar, f"{key}.parent{index:03d}.txt", text.encode("utf-8"))
    return target


def main() -> int:
    """CLI entry point: ``make_grit_fixture.py <output.tar>``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", help="path of the tar shard to write")
    parser.add_argument("--image-size", type=int, default=64)
    args = parser.parse_args()
    print(write_grit_fixture(args.output, args.image_size))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
