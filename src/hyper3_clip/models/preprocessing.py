"""Config-driven preprocessing for :class:`~hyper3_clip.models.hyper3_clip.Hyper3CLIP`.

Thin wrappers that read ``image_size`` / ``text_model_name`` /
``max_text_length`` off a model or data config, so callers do not have to
restate the paper's numbers.  The image statistics are ImageNet's, not CLIP's
(paper Sec. 4).

Offline use
-----------
:func:`build_tokenizer` fetches the CLIP tokenizer from the Hugging Face hub the
first time it runs and then reads it from the local cache.  To run with no
network, pre-populate the cache once (``huggingface-cli download
openai/clip-vit-base-patch32``) and export ``HF_HUB_OFFLINE=1``, or point
``text_model_name`` at a local directory holding the tokenizer files.
"""

from __future__ import annotations

from typing import Any, Protocol

from hyper3_clip.data.transforms import (
    build_eval_transform as _build_eval_transform,
)
from hyper3_clip.data.transforms import (
    build_retrieval_transform as _build_retrieval_transform,
)
from hyper3_clip.data.transforms import (
    build_train_transform as _build_train_transform,
)

__all__ = ["build_eval_transform", "build_retrieval_transform", "build_tokenizer", "build_train_transform"]


class _HasImageSize(Protocol):
    image_size: int


def _image_size(config: Any) -> int:
    size = getattr(config, "image_size", None)
    if size is None and isinstance(config, dict):
        size = config.get("image_size")
    return int(224 if size is None else size)


def _normalization(config: Any) -> str:
    value = getattr(config, "image_normalization", None)
    if value is None and isinstance(config, dict):
        value = config.get("image_normalization")
    return str(value or "imagenet")


def build_eval_transform(config: Any) -> Any:
    """Resize-then-centre-crop evaluation transform for ``config``."""
    return _build_eval_transform(_image_size(config), _normalization(config))


def build_retrieval_transform(config: Any) -> Any:
    """Squash-resize transform matching the released artifact's own preprocessing."""
    return _build_retrieval_transform(_image_size(config), _normalization(config))


def build_train_transform(config: Any, preset: str = "tight_crop_color_jitter_gray") -> Any:
    """Training augmentation for ``config`` (paper preset by default)."""
    return _build_train_transform(_image_size(config), preset, _normalization(config))


def build_tokenizer(config: Any) -> Any:
    """Return the CLIP tokenizer named by ``config.text_model_name``."""
    from transformers import AutoTokenizer

    name = getattr(config, "text_model_name", None)
    if name is None and isinstance(config, dict):
        name = config.get("text_model_name")
    if name is None:
        raise ValueError("config must carry a text_model_name")
    return AutoTokenizer.from_pretrained(str(name))
