"""Random but Lorentz-valid objective inputs shared by the objective tests.

Batch 4, 6 packed parts, 8 queries with valid owners, one root query per image
and one zero-weight query.
"""

from __future__ import annotations

import torch

from hyper3_clip.geometry.lorentz import exp_map0

BATCH = 4
PARTS = 6
QUERIES = 8
EMBED = 16

PART_OWNER = (0, 0, 1, 2, 2, 3)
QUERY_OWNER = (0, 0, 1, 1, 2, 2, 3, 3)
#: ``-1`` marks a root query; every other parent points at an earlier query.
QUERY_PARENT = (-1, 0, -1, 2, -1, 4, -1, 6)
#: the last query has weight 0 and must therefore contribute nothing.
QUERY_WEIGHT = (1.0, 0.5, 1.0, 0.25, 1.0, 2.0, 1.0, 0.0)


def build_nodes(seed: int = 0, *, embed: int = EMBED) -> dict:
    """Build one batch of nodes in the public ``compute_objective`` contract."""
    gen = torch.Generator().manual_seed(seed)
    kappa = torch.tensor(1.0)

    def lorentz(n: int) -> torch.Tensor:
        return exp_map0(torch.randn(n, embed, generator=gen) * (embed**-0.5), kappa)

    part_owner = torch.tensor(PART_OWNER, dtype=torch.long)
    image_feats = lorentz(BATCH)
    text_feats = lorentz(BATCH)
    return {
        "image_feats": image_feats,
        "text_feats": text_feats,
        # Single-process run: the "gathered" negative pools are the local ones.
        # They are listed explicitly so that swapping ``image_feats`` for a
        # gradient probe leaves the negative pool alone.
        "all_image_feats": image_feats,
        "all_text_feats": text_feats,
        "part_image_flat": lorentz(PARTS),
        "part_text_flat": lorentz(PARTS),
        "part_owner": part_owner,
        "targets": torch.arange(BATCH),
        "kappa": kappa,
        "entail_weight_scale": torch.ones(()),
        "query_image_feats": lorentz(QUERIES),
        "query_text_feats": lorentz(QUERIES),
        "query_owner": torch.tensor(QUERY_OWNER, dtype=torch.long),
        "query_parent": torch.tensor(QUERY_PARENT, dtype=torch.long),
        "query_weight": torch.tensor(QUERY_WEIGHT),
        "logit_scales": {
            "global": torch.tensor(1 / 0.07).log(),
            "local": torch.tensor(1 / 0.05).log(),
            "global_local": torch.tensor(1 / 0.06).log(),
        },
    }
