"""Query-conditioned pooling: shapes, conditioning and the mean-pooled control."""

from __future__ import annotations

import pytest
import torch

from hyper3_clip.models.query_pooling import QueryConditionedPooling

VISUAL_DIM = 32
TEXT_DIM = 16
TOKENS = 10
BATCH = 3
OWNER = (0, 0, 1, 2, 2, 2)


def _module(seed: int = 0, **kwargs) -> QueryConditionedPooling:
    torch.manual_seed(seed)
    module = QueryConditionedPooling(VISUAL_DIM, TEXT_DIM, **kwargs)
    return module.eval()


def _inputs(seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    patch_tokens = torch.randn(BATCH, TOKENS, VISUAL_DIM, generator=gen)
    query_feats = torch.randn(len(OWNER), TEXT_DIM, generator=gen)
    return patch_tokens, query_feats, torch.tensor(OWNER, dtype=torch.long)


def test_parameter_names_are_the_public_contract() -> None:
    """These four names are what a checkpoint's query-pooling block must carry."""
    module = _module()
    names = {name.split(".")[0] for name, _ in module.named_parameters()}
    assert names == {"text_query_proj", "cross_attention", "mlp_norm", "pool_mlp"}
    assert module.cross_attention.num_heads == 8
    assert module.pool_mlp[0].out_features == 4 * VISUAL_DIM
    assert isinstance(module.pool_mlp[1], torch.nn.GELU)


def test_output_shape_is_one_node_per_query() -> None:
    module = _module()
    patch_tokens, query_feats, owner = _inputs()
    assert module(patch_tokens, query_feats, owner).shape == (len(OWNER), VISUAL_DIM)


@pytest.mark.parametrize("owner", [(0, 1, 2), (2, 2, 2, 1, 0), (1,)])
def test_ragged_owner_sets_pool_the_right_image(owner) -> None:
    module = _module(seed=1)
    gen = torch.Generator().manual_seed(2)
    patch_tokens = torch.randn(BATCH, TOKENS, VISUAL_DIM, generator=gen)
    query_feats = torch.randn(len(owner), TEXT_DIM, generator=gen)
    owner_tensor = torch.tensor(owner, dtype=torch.long)
    pooled = module(patch_tokens, query_feats, owner_tensor)
    assert pooled.shape == (len(owner), VISUAL_DIM)
    for row, image in enumerate(owner):
        single = module(patch_tokens, query_feats[row : row + 1], owner_tensor[row : row + 1])
        assert torch.allclose(pooled[row], single[0], atol=1e-5), image


def test_the_cls_token_is_never_pooled() -> None:
    module = _module()
    patch_tokens, query_feats, owner = _inputs()
    perturbed = patch_tokens.clone()
    perturbed[:, 0, :] += 100.0  # only the CLS position changes
    assert torch.allclose(module(patch_tokens, query_feats, owner), module(perturbed, query_feats, owner), atol=1e-5)


def test_conditioned_pooling_depends_on_the_query() -> None:
    module = _module()
    patch_tokens, query_feats, owner = _inputs()
    pooled = module(patch_tokens, query_feats, owner)
    # queries 0 and 1 share an image; different text must give different nodes
    assert not torch.allclose(pooled[0], pooled[1], atol=1e-4)


def test_mean_mode_is_the_plain_token_mean() -> None:
    module = _module(mode="mean")
    patch_tokens, query_feats, owner = _inputs()
    pooled = module(patch_tokens, query_feats, owner)
    expected = patch_tokens[:, 1:, :].index_select(0, owner).mean(dim=1)
    assert torch.allclose(pooled, expected, atol=1e-5)
    # the control ignores the query text entirely
    assert torch.allclose(pooled[0], pooled[1], atol=1e-6)


def test_mean_mode_keeps_pooling_parameters_in_the_graph() -> None:
    """The zero-valued pooling call is what keeps DDP's reduction invariant."""
    module = _module(mode="mean").train()
    patch_tokens, query_feats, owner = _inputs()
    module(patch_tokens, query_feats, owner).sum().backward()
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_gradients_flow_to_every_parameter_in_conditioned_mode() -> None:
    module = _module().train()
    patch_tokens, query_feats, owner = _inputs()
    module(patch_tokens, query_feats, owner).sum().backward()
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_rejects_bad_shapes_and_owners() -> None:
    module = _module()
    patch_tokens, query_feats, owner = _inputs()
    with pytest.raises(ValueError):
        module(patch_tokens[0], query_feats, owner)
    with pytest.raises(ValueError):
        module(patch_tokens, query_feats, owner[:-1])
    with pytest.raises(IndexError):
        module(patch_tokens, query_feats, torch.full_like(owner, BATCH))
    with pytest.raises(ValueError):
        QueryConditionedPooling(VISUAL_DIM, TEXT_DIM, num_heads=7)
    with pytest.raises(ValueError):
        QueryConditionedPooling(VISUAL_DIM, TEXT_DIM, mode="other")
