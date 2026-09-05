"""Cross-modal caption retrieval (paper Table 2, and the Avg R@10 of Table 5).

Three readers, all producing the same ``{image, captions, image_id}`` item:

``CocoKarpathyRetrievalDataset``
    COCO 2014 with the Karpathy split file — the Table 2 "COCO" block.
``Flickr30kRetrievalDataset``
    Flickr30K with its Karpathy-format split file — the Table 2 "Flickr30K" block.
``CocoVal2017RetrievalDataset``
    COCO val2017 with ``captions_val2017.json``.  This is *not* what Table 2
    reports; it is kept because the Table 5 ablation driver computed its
    Avg R@10 on val2017, and reproducing that column needs the same image set
    (see ``docs/evaluation.md``).

Naming follows the paper: ranking captions for an image is **text retrieval**
(``i2t_*``), ranking images for a caption is **image retrieval** (``t2i_*``).
Text retrieval counts a hit when any of the image's own captions is in the
top ``k``; image retrieval has a single correct image per caption.

Images use :func:`~hyper3_clip.data.transforms.build_retrieval_transform` — a
square resize with no centre crop, matching the released artifact's own
preprocessing (see ``docs/evaluation.md``).
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset

from hyper3_clip.data.transforms import build_retrieval_transform
from hyper3_clip.evaluation.encoding import encode_texts, resolve_tokenizer, similarity_matrix
from hyper3_clip.evaluation.metrics import recall_at_k, single_target_recall_at_k
from hyper3_clip.models.hyper3_clip import Hyper3CLIP

__all__ = [
    "CocoKarpathyRetrievalDataset",
    "CocoVal2017RetrievalDataset",
    "Flickr30kRetrievalDataset",
    "evaluate_caption_retrieval",
]


class _CaptionRetrievalDataset(Dataset):
    """Common item shape and transform for the three retrieval readers."""

    items: list[dict[str, Any]]

    def _setup_transform(self, image_size: int, image_normalization: str) -> None:
        self.transform = build_retrieval_transform(image_size, normalization=image_normalization)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        from PIL import Image

        item = self.items[index]
        with Image.open(item["image_path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return {"image": tensor, "captions": item["captions"], "image_id": item["image_id"]}


class CocoKarpathyRetrievalDataset(_CaptionRetrievalDataset):
    """COCO 2014 Karpathy split (paper Table 2, "COCO").

    Expects ``<root>/karpathy/dataset_coco.json`` and the ``train2014`` /
    ``val2014`` image directories the split file names.  Captions are every
    ``sentences[*].raw``, stripped — five per image, occasionally six or seven.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "test",
        image_size: int = 224,
        max_items: int | None = None,
        image_normalization: str = "imagenet",
    ) -> None:
        self.root = Path(root)
        payload = json.loads((self.root / "karpathy" / "dataset_coco.json").read_text(encoding="utf-8"))
        images = [item for item in payload["images"] if item["split"] == split]
        if max_items is not None:
            images = images[:max_items]
        self.items = [
            {
                "image_id": item["imgid"],
                "image_path": self.root / item["filepath"] / item["filename"],
                "captions": [str(sentence["raw"]).strip() for sentence in item["sentences"]],
            }
            for item in images
        ]
        self._setup_transform(image_size, image_normalization)


class Flickr30kRetrievalDataset(_CaptionRetrievalDataset):
    """Flickr30K test split (paper Table 2, "Flickr30K").

    Expects ``<root>/dataset_flickr30k.json`` and ``<root>/flickr30k_images/``.
    ``image_id`` is the enumeration index over *all* images in the split file,
    not over the filtered split, which is what the reported run recorded.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "test",
        image_size: int = 224,
        max_items: int | None = None,
        image_normalization: str = "imagenet",
    ) -> None:
        self.root = Path(root)
        payload = json.loads((self.root / "dataset_flickr30k.json").read_text(encoding="utf-8"))
        self.items = []
        for index, image_payload in enumerate(payload["images"]):
            if image_payload.get("split") != split:
                continue
            captions = [
                str(sentence.get("raw") or " ".join(sentence.get("tokens", [])))
                for sentence in image_payload["sentences"]
            ]
            self.items.append(
                {
                    "image_id": index,
                    "image_path": self.root / "flickr30k_images" / image_payload["filename"],
                    "captions": captions,
                }
            )
        if max_items is not None:
            self.items = self.items[:max_items]
        self._setup_transform(image_size, image_normalization)


class CocoVal2017RetrievalDataset(_CaptionRetrievalDataset):
    """COCO val2017 captions — the Avg R@10 convention of the Table 5 ablation.

    Expects ``<root>/annotations/captions_val2017.json`` and ``<root>/val2017/``.
    Images are ordered by COCO image id.  Table 2 does **not** use this reader.
    """

    def __init__(
        self,
        root: str | Path,
        image_size: int = 224,
        max_items: int | None = None,
        image_normalization: str = "imagenet",
    ) -> None:
        self.root = Path(root)
        payload = json.loads((self.root / "annotations" / "captions_val2017.json").read_text(encoding="utf-8"))
        file_names = {int(item["id"]): str(item["file_name"]) for item in payload["images"]}
        captions: dict[int, list[str]] = defaultdict(list)
        for annotation in payload["annotations"]:
            captions[int(annotation["image_id"])].append(str(annotation["caption"]))
        self.items = [
            {
                "image_id": image_id,
                "image_path": self.root / "val2017" / file_names[image_id],
                "captions": captions[image_id],
            }
            for image_id in sorted(captions)
        ]
        if max_items is not None:
            self.items = self.items[:max_items]
        self._setup_transform(image_size, image_normalization)


@torch.inference_mode()
def evaluate_caption_retrieval(
    model: Hyper3CLIP,
    dataset: Dataset,
    device: torch.device,
    *,
    tokenizer: Any | None = None,
    max_text_length: int = 77,
    batch_size: int = 128,
) -> dict[str, float]:
    """Recall@{1,5,10} in both directions (paper Table 2).

    Returns the fractions under ``image_to_text_r*`` / ``text_to_image_r*`` and
    the same numbers ×100 under the short aliases ``i2t_r*`` / ``t2i_r*``; the
    paper's columns are ``i2t_r5``, ``i2t_r10``, ``t2i_r5``, ``t2i_r10``.
    """
    model.eval()
    tokenizer = resolve_tokenizer(model, tokenizer)
    num_images = len(dataset)  # type: ignore[arg-type]
    if num_images == 0:
        raise ValueError("caption retrieval requires at least one image")

    image_features: list[Tensor] = []
    captions: list[str] = []
    text_to_image: list[int] = []
    batch: list[Tensor] = []
    for index in range(num_images):
        item = dataset[index]
        batch.append(item["image"])
        if len(batch) == batch_size or index == num_images - 1:
            image_features.append(model.encode_image(torch.stack(batch).to(device)).cpu())
            batch = []
        captions.extend(item["captions"])
        text_to_image.extend([index] * len(item["captions"]))

    images = torch.cat(image_features).to(device)
    texts = encode_texts(
        model, tokenizer, captions, device, max_text_length=max_text_length, batch_size=batch_size
    ).to(device)

    scores_i2t = similarity_matrix(model, images, texts)
    scores_t2i = scores_i2t.transpose(0, 1)
    image_targets = _image_to_text_targets(text_to_image, num_images)
    caption_targets = torch.tensor(text_to_image, dtype=torch.long, device=scores_i2t.device)

    fractions = {
        f"image_to_text_r{k}": recall_at_k(scores_i2t, image_targets, k) for k in (1, 5, 10)
    }
    fractions.update(
        {f"text_to_image_r{k}": single_target_recall_at_k(scores_t2i, caption_targets, k) for k in (1, 5, 10)}
    )
    aliases = {f"i2t_r{k}": 100.0 * fractions[f"image_to_text_r{k}"] for k in (1, 5, 10)}
    aliases.update({f"t2i_r{k}": 100.0 * fractions[f"text_to_image_r{k}"] for k in (1, 5, 10)})
    return {
        **fractions,
        **aliases,
        "num_images": float(num_images),
        "num_captions": float(len(captions)),
    }


def _image_to_text_targets(text_to_image: Sequence[int], num_images: int) -> list[list[int]]:
    targets: list[list[int]] = [[] for _ in range(num_images)]
    for text_index, image_index in enumerate(text_to_image):
        targets[image_index].append(text_index)
    return targets
