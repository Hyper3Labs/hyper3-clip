"""Checkpoint interchange for :class:`~hyper3_clip.models.hyper3_clip.Hyper3CLIP`.

Two on-disk layouts load into the same module:

**(a) this repository's own checkpoints** -- either a directory holding
``config.yaml`` + ``model.safetensors`` (written by ``save_pretrained``) or a
training checkpoint ``.pt`` holding ``{"model", "config", ...}``.  Key names
are already the public ones.

**(b) the released Hugging Face artifact** (``config.json`` +
``model.safetensors``, https://huggingface.co/hyper3labs/hyper3-clip).  Encoder,
projection, ``visual_alpha``, ``textual_alpha`` and ``log_curv`` names are
identical; the artifact carries a *single* ``logit_scale`` where this model has
three, and it was trained with ``global_pool="avg"``.

Every mapping here is name-level only -- no tensor is reshaped, transposed or
rescaled.
"""

from __future__ import annotations

from collections.abc import Mapping

from torch import Tensor, nn

__all__ = [
    "HUB_BROADCAST_MAP",
    "detect_checkpoint_flavour",
    "load_mapped_state_dict",
    "remap_state_dict",
]

#: The released artifact's single temperature -> the three this model has.
#: All three scale contrastive logits only, so broadcasting one value into all
#: three is inference-inert.
HUB_BROADCAST_MAP: dict[str, tuple[str, ...]] = {
    "logit_scale": ("global_logit_scale", "local_logit_scale", "global_local_logit_scale"),
}

_TEXT_BACKBONE_PREFIX = "text_encoder.backbone."
_TEXT_MODEL_PREFIX = "text_encoder.backbone.text_model."


def detect_checkpoint_flavour(state_dict: Mapping[str, Tensor]) -> str:
    """Return ``"public"`` or ``"hub"`` for a raw state dict."""
    keys = set(state_dict)
    if "logit_scale" in keys and "global_logit_scale" not in keys:
        return "hub"
    return "public"


def remap_state_dict(
    state_dict: Mapping[str, Tensor],
    target_keys: set[str],
    *,
    flavour: str | None = None,
) -> tuple[dict[str, Tensor], dict[str, str]]:
    """Rename ``state_dict`` onto this module's key namespace.

    ``target_keys`` is ``dict(model.state_dict()).keys()``; it is used both to
    resolve the ``text_model.`` prefix (present with ``transformers`` 4.x,
    absent with 5.x) and to report what could not be matched.  Returns the
    renamed dict and a ``{old: new}`` log of every rename applied.
    """
    flavour = detect_checkpoint_flavour(state_dict) if flavour is None else flavour
    renames: dict[str, str] = {}
    mapped: dict[str, Tensor] = {}

    for key, value in state_dict.items():
        names = list(HUB_BROADCAST_MAP[key]) if flavour == "hub" and key in HUB_BROADCAST_MAP else [key]
        for name in names:
            name = _align_text_prefix(name, target_keys)
            if name != key:
                renames[key] = name
            mapped[name] = value

    return mapped, renames


def _align_text_prefix(key: str, target_keys: set[str]) -> str:
    """Add or drop the CLIP ``text_model.`` prefix to match the live module.

    ``transformers`` 5 removed the ``CLIPTextModel.text_model`` wrapper, so a
    checkpoint written under 4.x carries one extra path segment (or one fewer,
    the other way round).  The weights themselves are unchanged.
    """
    if key in target_keys:
        return key
    if key.startswith(_TEXT_MODEL_PREFIX):
        stripped = _TEXT_BACKBONE_PREFIX + key[len(_TEXT_MODEL_PREFIX) :]
        if stripped in target_keys:
            return stripped
    elif key.startswith(_TEXT_BACKBONE_PREFIX):
        expanded = _TEXT_MODEL_PREFIX + key[len(_TEXT_BACKBONE_PREFIX) :]
        if expanded in target_keys:
            return expanded
    return key


def load_mapped_state_dict(
    model: nn.Module,
    state_dict: Mapping[str, Tensor],
    *,
    flavour: str | None = None,
    strict: bool = True,
) -> dict[str, str]:
    """Remap ``state_dict`` and load it into ``model``.

    With ``strict=True`` a mismatch raises ``RuntimeError`` naming every key
    that the checkpoint did not supply and every key it supplied that this
    module does not have, after renaming -- so the message points at the real
    incompatibility rather than at the renaming.
    """
    target = dict(model.state_dict())
    target_keys = set(target)
    mapped, renames = remap_state_dict(state_dict, target_keys, flavour=flavour)

    missing = sorted(target_keys - set(mapped))
    unexpected = sorted(set(mapped) - target_keys)
    shape_mismatch = [
        f"{key}: checkpoint {tuple(mapped[key].shape)} != model {tuple(target[key].shape)}"
        for key in sorted(set(mapped) & target_keys)
        if tuple(mapped[key].shape) != tuple(target[key].shape)
    ]
    if strict and (missing or unexpected or shape_mismatch):
        detected = detect_checkpoint_flavour(state_dict) if flavour is None else flavour
        raise RuntimeError(
            "checkpoint is not compatible with this model configuration "
            f"(detected layout: {detected!r})."
            + (f"\n  missing from the checkpoint ({len(missing)}): {missing}" if missing else "")
            + (f"\n  unmatched checkpoint keys ({len(unexpected)}): {unexpected}" if unexpected else "")
            + (f"\n  shape mismatches: {shape_mismatch}" if shape_mismatch else "")
        )
    model.load_state_dict(mapped, strict=strict)
    return renames
