"""Sample and batch types for the Hyper3-CLIP training pipeline.

A processed-GRIT sample is one *image group*: the whole image with its caption,
plus the localized parts (already-cropped box images with their box texts).
The collator turns a list of these into :class:`GroundedBatch`, whose fields
are exactly the keyword arguments of ``Hyper3CLIP.forward``.

Shapes (``B`` images, ``P`` parts, ``Q`` queries in the batch; ``L*`` are
longest-in-batch token lengths, each ``<= max_text_length``):

===============================  ==================  =======
field                            shape               dtype
===============================  ==================  =======
``image``                        ``[B, 3, H, W]``    float32
``part_images``                  ``[P, 3, H, W]``    float32
``part_owner``                   ``[P]``             int64
``text_input_ids``               ``[B, Lc]``         int64
``text_attention_mask``          ``[B, Lc]``         int64
``part_text_input_ids``          ``[P, Lp]``         int64
``part_text_attention_mask``     ``[P, Lp]``         int64
``query_input_ids``              ``[Q, Lq]``         int64
``query_attention_mask``         ``[Q, Lq]``         int64
``query_owner``                  ``[Q]``             int64
``query_parent``                 ``[Q]``             int64
``query_weight``                 ``[Q]``             float32
``query_source_part``            ``[Q]``             int64
===============================  ==================  =======

``query_parent`` and ``query_source_part`` are *batch-global* row indices
(``-1`` for "no parent" / "not a text box"); ``part_owner`` and ``query_owner``
index the image rows.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from torch import Tensor

__all__ = ["GroundedBatch", "GroundedPart", "GroundedSample"]


@dataclass(frozen=True)
class GroundedPart:
    """One localized part: a box text and, on the manifest path, its geometry.

    The processed-GRIT shards store each box as a pre-cropped JPEG, so
    ``bbox`` is ``None`` there; it is kept because the JSON-manifest variant of
    the dataset carries explicit ``(x0, y0, x1, y1)`` boxes.
    """

    text: str
    image_path: Path | None = None
    bbox: tuple[float, float, float, float] | None = None


@dataclass
class GroundedSample:
    """One image group as produced by the dataset reader."""

    image: Tensor
    caption: str
    part_images: list[Tensor] = field(default_factory=list)
    part_texts: list[str] = field(default_factory=list)
    key: str = ""

    def __post_init__(self) -> None:
        if len(self.part_images) != len(self.part_texts):
            raise ValueError("part_images and part_texts must have the same length")

    def as_dict(self) -> dict[str, Any]:
        """Return the plain-dict form the collator and DataLoader use."""
        return {
            "image": self.image,
            "caption": self.caption,
            "part_images": list(self.part_images),
            "part_texts": list(self.part_texts),
            "key": self.key,
        }


@dataclass
class GroundedBatch:
    """A collated training batch; see the module docstring for shapes."""

    image: Tensor
    part_images: Tensor
    part_owner: Tensor
    text_input_ids: Tensor
    text_attention_mask: Tensor
    part_text_input_ids: Tensor
    part_text_attention_mask: Tensor
    query_input_ids: Tensor | None = None
    query_attention_mask: Tensor | None = None
    query_owner: Tensor | None = None
    query_parent: Tensor | None = None
    query_weight: Tensor | None = None
    query_source_part: Tensor | None = None

    def as_dict(self) -> dict[str, Tensor]:
        """Return the non-``None`` fields as a plain tensor dict."""
        return {name: value for name, value in asdict(self).items() if value is not None}

    @classmethod
    def from_dict(cls, payload: dict[str, Tensor]) -> GroundedBatch:
        """Build a batch from the collator's tensor dict."""
        known = {f.name for f in fields(cls)}
        unknown = set(payload) - known
        if unknown:
            raise ValueError(f"unexpected batch keys: {sorted(unknown)}")
        return cls(**payload)  # type: ignore[arg-type]

    def to(self, device: Any, *, non_blocking: bool = False) -> GroundedBatch:
        """Move every tensor to ``device``."""
        moved = {
            name: (value.to(device, non_blocking=non_blocking) if value is not None else None)
            for name, value in self.__dict__.items()
        }
        return GroundedBatch(**moved)

    def __iter__(self) -> Iterator[str]:
        return iter(self.as_dict())
