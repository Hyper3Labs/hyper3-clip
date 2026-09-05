"""The two Hyper3-CLIP towers.

Both towers are trained from random initialisation, which is the paper recipe
and the only mode this release supports:

* :class:`VisionEncoder` wraps a timm ViT.  The paper model is
  ``vit_base_patch16_224`` with ``global_pool="token"`` (the pooled feature is
  the CLS token after ``backbone.norm``), a fixed 2-D sine-cosine positional
  embedding and an explicit ``nn.LayerNorm``.  It exposes
  :meth:`VisionEncoder.forward_with_tokens`, which returns the pooled feature
  *and* the ``[B, 1 + N, d_v]`` patch-token sequence that query-conditioned
  pooling (Sec. 3.2) attends over.
* :class:`TextEncoder` wraps the ``transformers`` CLIP text architecture
  (``openai/clip-vit-base-patch32``), built from its config so the weights are
  random, and pools with CLIP's own pooler: the final-layer-norm hidden state
  at the EOT position.

``global_pool="avg"`` is kept because the checkpoint published on the Hub was
trained with it; see ``docs/method.md``.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = ["TextEncoder", "VisionEncoder", "apply_sincos2d_pos_embed"]

_GLOBAL_POOL_CHOICES = frozenset({"avg", "token"})
_NORM_LAYER_CHOICES = frozenset({"default", "layer_norm"})


class VisionEncoder(nn.Module):
    """timm ViT tower returning pooled features and patch tokens.

    ``output_dim`` is the backbone width (768 for ViT-B/16).  With
    ``global_pool="token"`` timm keeps a trained ``backbone.norm`` LayerNorm
    and an identity ``fc_norm``; with ``"avg"`` it is the other way round.
    That single choice is the most visible state-dict difference between the
    paper model and the released checkpoint.
    """

    def __init__(
        self,
        backbone_name: str,
        *,
        global_pool: str = "token",
        use_sincos2d_pos: bool = False,
        norm_layer: str | None = None,
    ) -> None:
        super().__init__()
        if global_pool not in _GLOBAL_POOL_CHOICES:
            raise ValueError("global_pool must be 'avg' or 'token'")
        if norm_layer not in (None, *_NORM_LAYER_CHOICES):
            raise ValueError("norm_layer must be None, 'default', or 'layer_norm'")

        import timm

        kwargs: dict[str, object] = {
            "pretrained": False,
            "num_classes": 0,
            "global_pool": global_pool,
        }
        if norm_layer == "layer_norm":
            kwargs["norm_layer"] = nn.LayerNorm
        self.backbone = timm.create_model(backbone_name, **kwargs)
        if use_sincos2d_pos:
            apply_sincos2d_pos_embed(self.backbone)
        self.global_pool = global_pool
        self.output_dim: int = int(self.backbone.num_features)

    def forward(self, image: Tensor) -> Tensor:
        """Return pooled image features, shape ``[B, output_dim]``."""
        return self.backbone(image)

    def forward_with_tokens(self, image: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(pooled [B, d_v], tokens [B, 1 + N, d_v])``.

        For ViT-B/16 at 224 px the token sequence is ``[B, 197, 768]``: one CLS
        token followed by 196 patch tokens.
        """
        if not hasattr(self.backbone, "forward_features"):
            pooled = self.backbone(image)
            return pooled, pooled[:, None, :]
        features = self.backbone.forward_features(image)
        if hasattr(self.backbone, "forward_head"):
            pooled = self.backbone.forward_head(features, pre_logits=False)
        else:
            pooled = self.backbone(image)
        return pooled, _tokens_from_features(features)


class TextEncoder(nn.Module):
    """CLIP text tower built from a ``transformers`` config, with random weights.

    The tokenizer is loaded from the same hub id and is the tokenizer the
    collator uses, so ``pad_token_id`` is always consistent between padding and
    the fused forward.
    """

    def __init__(self, model_name: str) -> None:
        super().__init__()
        from transformers import AutoTokenizer, CLIPTextConfig, CLIPTextModel

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = CLIPTextModel(CLIPTextConfig.from_pretrained(model_name))
        self.output_dim: int = int(self.backbone.config.hidden_size)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Return pooled text features (CLIP's EOT pooler), shape ``[N, output_dim]``."""
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = getattr(out, "pooler_output", None)
        return out.last_hidden_state[:, 0] if pooled is None else pooled


def _tokens_from_features(features: Tensor | dict | tuple | list) -> Tensor:
    """Normalise a timm ``forward_features`` output into ``[B, N, d]`` tokens."""
    if isinstance(features, dict):
        for key in ("x", "last_hidden_state", "features"):
            if key in features:
                features = features[key]
                break
        else:
            features = next(iter(features.values()))
    if isinstance(features, tuple | list):
        features = features[0]
    if not torch.is_tensor(features):
        raise TypeError(f"Expected tensor features, got {type(features)!r}")
    if features.ndim == 4:
        return features.flatten(2).transpose(1, 2)
    if features.ndim == 3:
        return features
    if features.ndim == 2:
        return features[:, None, :]
    raise ValueError(f"Unsupported feature tensor shape {tuple(features.shape)}")


def apply_sincos2d_pos_embed(model: nn.Module) -> None:
    """Overwrite a ViT's ``pos_embed`` with a fixed 2-D sine-cosine grid.

    The parameter is kept (so it stays in the state dict) but frozen with
    ``requires_grad_(False)``, which also keeps it out of every optimizer
    group.  The ``w``/``h`` role assignment below is the one the paper run
    used; on the square 14x14 grid of ViT-B/16 at 224 px it is a transpose of
    the more common MAE ordering, and it is kept as it is so checkpoints stay
    comparable.
    """
    if not hasattr(model, "patch_embed") or not hasattr(model.patch_embed, "grid_size"):
        raise ValueError("Fixed 2D sine-cosine positions require a ViT-style patch_embed.grid_size")
    if not hasattr(model, "pos_embed"):
        raise ValueError("Fixed 2D sine-cosine positions require model.pos_embed")
    height, width = model.patch_embed.grid_size
    embed_dim = int(model.pos_embed.shape[-1])
    if embed_dim % 4 != 0:
        raise ValueError("ViT embed_dim must be divisible by 4 for 2D sine-cosine positions")

    grid_w = torch.arange(width, dtype=torch.float32)
    grid_h = torch.arange(height, dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")

    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1.0 / (10000.0**omega)
    out_w = torch.einsum("m,d->md", [grid_w.flatten(), omega])
    out_h = torch.einsum("m,d->md", [grid_h.flatten(), omega])
    pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]

    if model.pos_embed.shape[1] == pos_emb.shape[1] + 1:
        cls_pos = torch.zeros((1, 1, embed_dim), dtype=torch.float32)
        pos_emb = torch.cat([cls_pos, pos_emb], dim=1)
    if model.pos_embed.shape != pos_emb.shape:
        raise ValueError(
            f"Fixed position shape {tuple(pos_emb.shape)} does not match model.pos_embed "
            f"{tuple(model.pos_embed.shape)}"
        )
    model.pos_embed.data.copy_(pos_emb.to(device=model.pos_embed.device, dtype=model.pos_embed.dtype))
    model.pos_embed.requires_grad_(False)
