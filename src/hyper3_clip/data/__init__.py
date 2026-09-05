"""Dataset-side construction for Hyper3-CLIP (paper Sec. 4)."""

from __future__ import annotations

from hyper3_clip.data.collate import GroundedCollator, QueryConfig, collate_grounded
from hyper3_clip.data.grit import ProcessedGritDataset
from hyper3_clip.data.query_hierarchy import (
    QUERY_WEIGHTS,
    Query,
    build_query_hierarchy,
    extract_lightweight_phrases,
    split_sentences,
)
from hyper3_clip.data.transforms import (
    build_eval_transform,
    build_retrieval_transform,
    build_train_transform,
)
from hyper3_clip.data.types import GroundedBatch, GroundedPart, GroundedSample

__all__ = [
    "QUERY_WEIGHTS",
    "GroundedBatch",
    "GroundedCollator",
    "GroundedPart",
    "GroundedSample",
    "ProcessedGritDataset",
    "Query",
    "QueryConfig",
    "build_eval_transform",
    "build_query_hierarchy",
    "build_retrieval_transform",
    "build_train_transform",
    "collate_grounded",
    "extract_lightweight_phrases",
    "split_sentences",
]
