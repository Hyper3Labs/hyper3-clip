"""Collate image groups into the tensors ``Hyper3CLIP.forward`` consumes.

Three text groups are tokenized independently, each with longest-in-batch
padding (``padding=True``, ``truncation=True``, ``max_length=max_text_length``):
captions, box texts and queries.  The model re-pads them to one common length
before the fused text forward, so the per-group lengths only have to be
consistent within their own group.

Parts and queries are **packed, not padded**: a batch of ``B`` images
contributes ``P = sum_b n_parts(b)`` part rows and ``Q = sum_b n_queries(b)``
query rows, with ``part_owner`` / ``query_owner`` recording which image each row
belongs to.  ``query_parent`` and ``query_source_part`` are rewritten from the
per-image indices that :func:`~hyper3_clip.data.query_hierarchy.build_query_hierarchy`
produces into batch-global row indices (``-1`` stays ``-1``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from hyper3_clip.data.query_hierarchy import build_query_hierarchy

__all__ = ["GroundedCollator", "QueryConfig", "collate_grounded"]


@dataclass(frozen=True)
class QueryConfig:
    """Query-construction budget (paper Sec. 4; defaults are the paper values)."""

    enabled: bool = True
    max_sentences: int = 5
    max_phrases: int = 30
    max_queries_per_image: int = 6
    use_text_boxes: bool = True


def _attention_mask(tokens: Mapping[str, Tensor]) -> Tensor:
    if "attention_mask" in tokens:
        return tokens["attention_mask"]
    return torch.ones_like(tokens["input_ids"])


def collate_grounded(
    batch: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    max_text_length: int = 77,
    *,
    queries: QueryConfig | None = None,
) -> dict[str, Tensor]:
    """Collate reader samples into one training batch.

    ``batch`` items are the dicts produced by
    :class:`~hyper3_clip.data.grit.ProcessedGritDataset`.  Returns the tensor
    dict documented in :mod:`hyper3_clip.data.types`; the query tensors are
    omitted when ``queries`` is disabled.
    """
    queries = QueryConfig() if queries is None else queries
    images = torch.stack([item["image"] for item in batch])
    captions = [str(item["caption"]) for item in batch]

    part_image_rows: list[Tensor] = []
    part_texts: list[str] = []
    part_owner: list[int] = []
    for batch_index, item in enumerate(batch):
        for part_index, part_image in enumerate(item["part_images"]):
            part_image_rows.append(part_image)
            part_texts.append(str(item["part_texts"][part_index]))
            part_owner.append(batch_index)

    text = tokenizer(captions, padding=True, truncation=True, max_length=max_text_length, return_tensors="pt")
    text_attention_mask = _attention_mask(text)
    if part_image_rows:
        part_images = torch.stack(part_image_rows)
        part_text = tokenizer(
            part_texts, padding=True, truncation=True, max_length=max_text_length, return_tensors="pt"
        )
        part_text_input_ids = part_text["input_ids"]
        part_text_attention_mask = _attention_mask(part_text)
    else:
        part_images = images.new_zeros((0, *images.shape[1:]))
        empty_shape = (0, int(text["input_ids"].shape[1]))
        part_text_input_ids = text["input_ids"].new_zeros(empty_shape)
        part_text_attention_mask = text_attention_mask.new_zeros(empty_shape)

    collated: dict[str, Tensor] = {
        "image": images,
        "part_images": part_images,
        "part_owner": torch.tensor(part_owner, dtype=torch.long),
        "text_input_ids": text["input_ids"],
        "text_attention_mask": text_attention_mask,
        "part_text_input_ids": part_text_input_ids,
        "part_text_attention_mask": part_text_attention_mask,
    }
    if queries.enabled:
        collated.update(_query_tensors(batch, tokenizer, max_text_length, queries))
    return collated


def _query_tensors(
    batch: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    max_text_length: int,
    queries: QueryConfig,
) -> dict[str, Tensor]:
    query_texts: list[str] = []
    query_owner: list[int] = []
    query_parent: list[int] = []
    query_weight: list[float] = []
    query_source_part: list[int] = []

    part_offsets: list[int] = []
    cursor = 0
    for item in batch:
        part_offsets.append(cursor)
        cursor += len(item["part_images"])

    for batch_index, item in enumerate(batch):
        hierarchy = build_query_hierarchy(
            str(item["caption"]),
            [str(text) for text in item["part_texts"]],
            max_sentences=queries.max_sentences,
            max_phrases=queries.max_phrases,
            max_queries=queries.max_queries_per_image,
            use_text_boxes=queries.use_text_boxes,
        )
        query_offset = len(query_texts)
        for query in hierarchy:
            query_texts.append(query.text)
            query_owner.append(batch_index)
            query_parent.append(-1 if query.parent < 0 else query_offset + query.parent)
            query_weight.append(query.weight)
            query_source_part.append(
                -1 if query.source_part < 0 else part_offsets[batch_index] + query.source_part
            )

    tokens = tokenizer(query_texts, padding=True, truncation=True, max_length=max_text_length, return_tensors="pt")
    return {
        "query_input_ids": tokens["input_ids"],
        "query_attention_mask": _attention_mask(tokens),
        "query_owner": torch.tensor(query_owner, dtype=torch.long),
        "query_parent": torch.tensor(query_parent, dtype=torch.long),
        "query_weight": torch.tensor(query_weight, dtype=torch.float32),
        "query_source_part": torch.tensor(query_source_part, dtype=torch.long),
    }


class GroundedCollator:
    """Picklable ``collate_fn`` binding a tokenizer and the query budget.

    A plain closure cannot cross a DataLoader worker boundary on the ``spawn``
    start method, so the training loop uses this instead.
    """

    def __init__(self, tokenizer: Any, max_text_length: int = 77, queries: QueryConfig | None = None) -> None:
        self.tokenizer = tokenizer
        self.max_text_length = int(max_text_length)
        self.queries = QueryConfig() if queries is None else queries

    def __call__(self, batch: Sequence[Mapping[str, Any]]) -> dict[str, Tensor]:
        return collate_grounded(batch, self.tokenizer, self.max_text_length, queries=self.queries)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(max_text_length={self.max_text_length}, queries={self.queries!r})"
