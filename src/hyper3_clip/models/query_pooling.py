"""Query-conditioned visual pooling (paper Sec. 3.2).

``v~_q = MHA(Q = W_Q h_q, K = Z_I, V = Z_I)``, ``v_q = v~_q + MLP(LN(v~_q))``
with 8 heads and MLP hidden dim ``4 * d_v`` (GELU).  ``h_q`` is the query's
text embedding projected into the visual width, ``Z_I`` the patch tokens of the
image that owns the query, and ``v_q`` the pooled visual node for query ``q``.

The CLS token is always dropped, so pooling attends over patch tokens only.
"""

from __future__ import annotations

from typing import Literal

from torch import Tensor, nn

__all__ = ["QueryConditionedPooling"]


class QueryConditionedPooling(nn.Module):
    """Pool visual patch tokens conditioned on per-query text embeddings.

    ``mode="conditioned"`` is the paper model.  ``mode="mean"`` is the
    unconditioned-pooling control of the ablation table: a plain mean over the
    owning image's patch tokens, with the pooling parameters kept in the
    autograd graph so the parameter layout and DDP's reduction invariant are
    unchanged.
    """

    def __init__(
        self,
        visual_dim: int,
        text_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        mode: Literal["conditioned", "mean"] = "conditioned",
    ) -> None:
        super().__init__()
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if mlp_ratio <= 0.0:
            raise ValueError("mlp_ratio must be positive")
        if visual_dim % num_heads != 0:
            raise ValueError("visual_dim must be divisible by num_heads")
        if mode not in ("conditioned", "mean"):
            raise ValueError(f"unknown mode: {mode!r}")

        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.num_heads = num_heads
        self.mlp_ratio = float(mlp_ratio)
        self.mode = mode

        hidden_dim = max(1, int(round(visual_dim * mlp_ratio)))
        self.text_query_proj = nn.Linear(text_dim, visual_dim)
        self.cross_attention = nn.MultiheadAttention(visual_dim, num_heads, batch_first=True)
        self.mlp_norm = nn.LayerNorm(visual_dim)
        self.pool_mlp = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, visual_dim),
        )

    def forward(self, patch_tokens: Tensor, query_feats: Tensor, owner: Tensor) -> Tensor:
        """Return one pooled visual node per query, shape ``[Q, d_v]``.

        ``patch_tokens``: ``[B, 1 + N, d_v]`` image tokens, CLS first;
        ``query_feats``: ``[Q, d_t]`` query text embeddings; ``owner``: ``[Q]``
        image index per query.
        """
        if patch_tokens.ndim != 3:
            raise ValueError(f"patch_tokens must have shape [batch, tokens, dim], got {tuple(patch_tokens.shape)}")
        if query_feats.ndim != 2:
            raise ValueError(f"query_feats must have shape [Q, d_t], got {tuple(query_feats.shape)}")
        if owner.ndim != 1 or owner.numel() != query_feats.size(0):
            raise ValueError("owner must be 1-D with one entry per query")

        tokens = self._patch_tokens(patch_tokens, owner)
        if self.mode == "mean":
            # The zero-valued conditioned call keeps every pooling parameter in
            # the autograd graph, which DDP's reduction invariant requires.
            zero = self._conditioned_pool(tokens.narrow(0, 0, 0), query_feats.narrow(0, 0, 0)).sum()
            return tokens.to(dtype=query_feats.dtype).mean(dim=1) + zero
        return self._conditioned_pool(tokens, query_feats)

    def _patch_tokens(self, patch_tokens: Tensor, owner: Tensor) -> Tensor:
        """Select the owning image's patch tokens per query, ``[Q, N, d_v]``."""
        if owner.numel() > 0 and (owner.min().item() < 0 or owner.max().item() >= patch_tokens.size(0)):
            raise IndexError("owner contains an out-of-range image index")
        without_cls = patch_tokens[:, 1:, :] if patch_tokens.size(1) > 1 else patch_tokens
        return without_cls.index_select(0, owner)

    def _conditioned_pool(self, tokens: Tensor, query_feats: Tensor) -> Tensor:
        query = self.text_query_proj(query_feats).unsqueeze(1)
        keys = tokens.to(dtype=query.dtype)
        attended, _ = self.cross_attention(query, keys, keys, need_weights=False)
        pooled = attended.squeeze(1)
        return pooled + self.pool_mlp(self.mlp_norm(pooled))
