"""Reader for the processed-GRIT WebDataset shards (paper Sec. 4).

Shard layout -- one WebDataset sample per image group, keyed by ``__key__``:

=========================  =============================================
member                     content
=========================  =============================================
``child.jpg``              the whole image
``child.txt``              its caption
``numparents.txt``         decimal ASCII count ``N`` of localized parts
``parent{i:03d}.jpg``      part ``i``, **already cropped**
``parent{i:03d}.txt``      the text of box ``i``
=========================  =============================================

There are no bounding boxes in the shards and this reader never crops: the
official HyCoCLIP-processed GRIT shards store each localized box as its own
pre-cropped JPEG, so "box cropping" is reading ``parentNNN.jpg`` and applying
the same train transform used for the whole image, with independent randomness.

Every localized part is kept; when a group has more than ``max_parts`` (5 in
the paper recipe) a random subset of that size is drawn and ascending order is
restored, so part order always follows the shard's own.  ``max_parts=None``
disables the cap.

Shards are split across ranks (``tarfiles[rank::world_size]``) and across
DataLoader workers; the iterator loops forever, so training is measured in
optimizer steps and there is no epoch boundary.
"""

from __future__ import annotations

import copy
import glob
import hashlib
import random
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import IterableDataset, get_worker_info

from hyper3_clip.data.transforms import build_train_transform
from hyper3_clip.distributed import get_rank, get_world_size

__all__ = ["ProcessedGritDataset"]


class ProcessedGritDataset(IterableDataset):
    """Iterate processed-GRIT shards as ``{image, caption, part_images, part_texts}``.

    ``image`` and each entry of ``part_images`` are normalized
    ``[3, image_size, image_size]`` float tensors; ``caption`` and
    ``part_texts`` are strings.  ``deterministic_transforms=True`` seeds each
    image's augmentation from the sample key, which makes fixtures and
    debugging reproducible; the paper run leaves it off.
    """

    def __init__(
        self,
        tarfiles: Sequence[str],
        image_size: int = 224,
        seed: int = 0,
        *,
        shuffle_buffer: int = 4000,
        max_parts: int | None = 5,
        train_transform: str = "tight_crop_color_jitter_gray",
        image_normalization: str = "imagenet",
        deterministic_transforms: bool = False,
    ) -> None:
        self.tarfiles = _expand_tarfiles(tarfiles)
        if not self.tarfiles:
            raise FileNotFoundError(f"No GRIT processed shards matched {tarfiles!r}")
        self.tarfiles = self.tarfiles[get_rank() :: get_world_size()]
        if max_parts is not None and max_parts <= 0:
            raise ValueError("max_parts must be positive when set")
        self.shuffle_buffer = int(shuffle_buffer)
        self.seed = int(seed)
        self.max_parts = max_parts
        self.deterministic_transforms = bool(deterministic_transforms)
        self.transform = build_train_transform(image_size, preset=train_transform, normalization=image_normalization)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        import webdataset as wds

        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        rank_stride = get_rank() * 1_000_003
        shuffle_rng = random.Random(self.seed + rank_stride + worker_id)
        part_rng = random.Random(self.seed + 31_415_926 + rank_stride + worker_id)
        pipeline: Any = wds.DataPipeline(
            wds.SimpleShardList(self.tarfiles, seed=self.seed),
            wds.split_by_worker,
            wds.tarfile_to_samples(),
            wds.shuffle(self.shuffle_buffer, initial=self.shuffle_buffer, rng=shuffle_rng),
            wds.decode("pil", handler=wds.warn_and_continue),
        )
        while True:
            for sample in copy.deepcopy(pipeline):
                yield self._decode_sample(sample, part_rng)

    def _decode_sample(self, sample: dict[str, Any], rng: random.Random) -> dict[str, Any]:
        num_parents = int(_as_text(sample["numparents.txt"]))
        parent_indices = self._select_parent_indices(num_parents, rng)
        parent_keys = [f"parent{index:03d}" for index in parent_indices]
        sample_key = _as_text(sample.get("__key__", ""))
        return {
            "image": self._transform_image(sample["child.jpg"], sample_key, "child"),
            "caption": _as_text(sample["child.txt"]),
            "part_images": [self._transform_image(sample[f"{key}.jpg"], sample_key, key) for key in parent_keys],
            "part_texts": [_as_text(sample[f"{key}.txt"]) for key in parent_keys],
            "key": sample_key,
        }

    def _select_parent_indices(self, num_parents: int, rng: random.Random) -> list[int]:
        if num_parents <= 0:
            return []
        indices = list(range(num_parents))
        if self.max_parts is not None and len(indices) > self.max_parts:
            indices = sorted(rng.sample(indices, k=self.max_parts))
        return indices

    def _transform_image(self, value: Any, sample_key: str, role: str) -> torch.Tensor:
        image = _as_image(value)
        if not self.deterministic_transforms:
            return self.transform(image)
        transform_seed = _stable_seed(self.seed, sample_key, role)
        python_random_state = random.getstate()
        try:
            random.seed(transform_seed)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(transform_seed)
                return self.transform(image)
        finally:
            random.setstate(python_random_state)


def _expand_tarfiles(tarfiles: Sequence[str]) -> list[str]:
    expanded: list[str] = []
    for pattern in tarfiles:
        matches = sorted(glob.glob(pattern))
        expanded.extend(matches if matches else [pattern])
    return [str(Path(path)) for path in expanded]


def _as_text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _as_image(value: Any) -> Any:
    from PIL import Image

    if not isinstance(value, Image.Image):
        raise TypeError(f"Expected PIL image from WebDataset decode, got {type(value)!r}")
    return value.convert("RGB")


def _stable_seed(seed: int, *parts: str) -> int:
    digest = hashlib.blake2b(digest_size=8)
    digest.update(str(seed).encode("utf-8"))
    for part in parts:
        digest.update(b"\0")
        digest.update(part.encode("utf-8"))
    return int.from_bytes(digest.digest(), byteorder="big", signed=False)
