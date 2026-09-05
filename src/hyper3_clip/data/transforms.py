"""Image transforms for Hyper3-CLIP (paper Sec. 4, "Training data").

The paper run uses ImageNet normalization (not CLIP's) and a single train
preset, ``tight_crop_color_jitter_gray``, applied *identically and
independently* to the whole image and to each already-cropped part image.
There is no horizontal flip anywhere in the pipeline.
"""

from __future__ import annotations

from typing import Literal

from torchvision import transforms

__all__ = [
    "CLIP_MEAN",
    "CLIP_STD",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "TRAIN_PRESETS",
    "build_eval_transform",
    "build_retrieval_transform",
    "build_train_transform",
    "normalization_stats",
]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)

Normalization = Literal["imagenet", "clip", "siglip"]
#: Train presets shipped with this release; the paper recipe uses the only entry.
TRAIN_PRESETS = ("tight_crop_color_jitter_gray",)


def normalization_stats(normalization: str) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return ``(mean, std)`` for a normalization name."""
    if normalization == "imagenet":
        return IMAGENET_MEAN, IMAGENET_STD
    if normalization == "clip":
        return CLIP_MEAN, CLIP_STD
    if normalization == "siglip":
        return SIGLIP_MEAN, SIGLIP_STD
    raise ValueError("normalization must be one of 'imagenet', 'clip', or 'siglip'")


def build_train_transform(
    image_size: int = 224,
    preset: str = "tight_crop_color_jitter_gray",
    normalization: str = "imagenet",
) -> transforms.Compose:
    """Build the training augmentation.

    ``tight_crop_color_jitter_gray`` is ``RandomResizedCrop(scale=(0.8, 1.0),
    bicubic)`` -> ``ColorJitter(0.4, 0.4, 0.4, 0.1)`` with probability 0.8 ->
    ``RandomGrayscale(p=0.2)`` -> ``ToTensor`` -> ``Normalize``.  The crop's
    aspect-ratio range is torchvision's default ``(3/4, 4/3)``.
    """
    if preset != "tight_crop_color_jitter_gray":
        raise ValueError(f"Unsupported train transform preset {preset!r}; expected one of {TRAIN_PRESETS}")
    mean, std = normalization_stats(normalization)
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size=image_size,
                scale=(0.8, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def build_eval_transform(image_size: int = 224, normalization: str = "imagenet") -> transforms.Compose:
    """Resize the short side then centre-crop: the standard evaluation view."""
    mean, std = normalization_stats(normalization)
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def build_retrieval_transform(image_size: int = 224, normalization: str = "imagenet") -> transforms.Compose:
    """Squash-resize to ``(image_size, image_size)`` with no crop.

    This is what the checkpoint published on the Hub applies inside its own
    ``preprocess_image``, so reproducing the released retrieval numbers uses
    this transform rather than :func:`build_eval_transform`.
    """
    mean, std = normalization_stats(normalization)
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
