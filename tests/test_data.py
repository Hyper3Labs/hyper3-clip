"""Reader -> collator -> model forward -> objective on the tiny GRIT fixture."""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest
import torch

from conftest import requires_tokenizer
from hyper3_clip.data import ProcessedGritDataset, QueryConfig, collate_grounded
from hyper3_clip.data.transforms import build_eval_transform, build_retrieval_transform, build_train_transform
from hyper3_clip.data.types import GroundedBatch
from hyper3_clip.models import Hyper3CLIP, Hyper3CLIPConfig, ObjectiveConfig


def _samples(shard: Path, count: int = 4, **kwargs) -> list[dict]:
    dataset = ProcessedGritDataset([str(shard)], image_size=224, seed=7, shuffle_buffer=4, **kwargs)
    return list(itertools.islice(iter(dataset), count))


def test_reader_yields_image_groups(grit_shard: Path) -> None:
    samples = _samples(grit_shard, max_parts=5)
    assert len(samples) == 4
    for sample in samples:
        assert sample["image"].shape == (3, 224, 224)
        assert len(sample["part_images"]) == len(sample["part_texts"]) >= 1
        assert all(part.shape == (3, 224, 224) for part in sample["part_images"])
    # The six-box group is capped at max_parts.
    assert max(len(sample["part_images"]) for sample in samples) == 5


def test_max_parts_caps_the_group(grit_shard: Path) -> None:
    assert max(len(s["part_images"]) for s in _samples(grit_shard, max_parts=2)) == 2
    uncapped = max(len(s["part_images"]) for s in _samples(grit_shard, max_parts=None))
    assert uncapped == 6


def test_transform_presets_produce_the_expected_shape() -> None:
    from PIL import Image

    image = Image.new("RGB", (300, 200), (120, 60, 30))
    for transform in (build_train_transform(224), build_eval_transform(224), build_retrieval_transform(224)):
        assert transform(image).shape == (3, 224, 224)
    with pytest.raises(ValueError, match="Unsupported train transform preset"):
        build_train_transform(224, preset="wide_random_crop")


@requires_tokenizer
def test_collator_packs_parts_and_globalizes_query_indices(grit_shard: Path, tiny_text_dir: Path) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(tiny_text_dir))
    samples = _samples(grit_shard, max_parts=5)
    batch = collate_grounded(samples, tokenizer, 77, queries=QueryConfig())

    parts = sum(len(sample["part_images"]) for sample in samples)
    assert batch["part_images"].shape[0] == parts == batch["part_owner"].numel()
    assert batch["part_owner"].tolist() == sorted(batch["part_owner"].tolist())
    queries = batch["query_owner"].numel()
    assert queries > 0
    assert batch["query_input_ids"].shape[0] == queries
    # Parent and source-part indices are batch-global and in range, or -1.
    parent = batch["query_parent"]
    assert int(parent.min()) >= -1 and int(parent.max()) < queries
    assert bool((parent < 0).any()), "the caption root must be parentless"
    source = batch["query_source_part"]
    assert int(source.min()) >= -1 and int(source.max()) < parts
    # A query always precedes its parent's own children: parents come earlier.
    valid = parent >= 0
    assert bool((parent[valid] < torch.arange(queries)[valid]).all())

    typed = GroundedBatch.from_dict(batch)
    assert set(typed.as_dict()) == set(batch)


@requires_tokenizer
@pytest.mark.parametrize("pooling_mode", ["conditioned", "mean"])
def test_forward_and_backward_on_the_fixture(
    grit_shard: Path, tiny_model_config: Hyper3CLIPConfig, pooling_mode: str
) -> None:
    from dataclasses import replace

    from transformers import AutoTokenizer

    config = replace(
        tiny_model_config,
        query_pooling_mode=pooling_mode,
        objective=ObjectiveConfig(),
    )
    tokenizer = AutoTokenizer.from_pretrained(config.text_model_name)
    batch = collate_grounded(_samples(grit_shard, max_parts=5), tokenizer, 77)

    model = Hyper3CLIP(config).train()
    losses = model.compute_loss(batch, step=0)
    assert torch.isfinite(losses["loss"])
    assert int(losses["part_count"]) == batch["part_owner"].numel()
    assert float(losses["query_count"]) > 0

    losses["loss"].backward()
    without_grad = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not without_grad, without_grad
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@requires_tokenizer
def test_shuffled_parent_control_changes_only_the_hierarchy(
    grit_shard: Path, tiny_model_config: Hyper3CLIPConfig
) -> None:
    from dataclasses import replace

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tiny_model_config.text_model_name)
    batch = collate_grounded(_samples(grit_shard, max_parts=5), tokenizer, 77)

    torch.manual_seed(0)
    true_model = Hyper3CLIP(tiny_model_config).eval()
    shuffled_model = Hyper3CLIP(replace(tiny_model_config, query_parent_mode="shuffled")).eval()
    shuffled_model.load_state_dict(true_model.state_dict())

    with torch.no_grad():
        true_nodes = true_model(batch)
        shuffled_nodes = shuffled_model(batch)
    assert torch.equal(true_nodes["query_image_feats"], shuffled_nodes["query_image_feats"])
    assert not torch.equal(true_nodes["query_parent"], shuffled_nodes["query_parent"])
    # The control preserves the number of hierarchy edges.
    assert int((true_nodes["query_parent"] >= 0).sum()) == int((shuffled_nodes["query_parent"] >= 0).sum())


@requires_tokenizer
def test_model_without_query_pooling_emits_empty_query_nodes(
    grit_shard: Path, tiny_model_config: Hyper3CLIPConfig
) -> None:
    from dataclasses import replace

    from transformers import AutoTokenizer

    config = replace(tiny_model_config, query_pooling=False)
    tokenizer = AutoTokenizer.from_pretrained(config.text_model_name)
    batch = collate_grounded(_samples(grit_shard, max_parts=5), tokenizer, 77)

    model = Hyper3CLIP(config).train()
    losses = model.compute_loss(batch)
    assert torch.isfinite(losses["loss"])
    assert float(losses["query_count"]) == 0.0
    assert not any(name.startswith("query_pooling.") for name in model.state_dict())


@requires_tokenizer
def test_query_pooling_stays_in_the_graph_on_a_query_free_batch(
    grit_shard: Path, tiny_model_config: Hyper3CLIPConfig
) -> None:
    """With pooling on but no queries, the pooling parameters still get gradients."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tiny_model_config.text_model_name)
    batch = collate_grounded(
        _samples(grit_shard, max_parts=5),
        tokenizer,
        77,
        queries=QueryConfig(enabled=False),
    )
    assert "query_input_ids" not in batch

    model = Hyper3CLIP(tiny_model_config).train()
    losses = model.compute_loss(batch)
    assert torch.isfinite(losses["loss"])
    assert float(losses["query_count"]) == 0.0
    losses["loss"].backward()
    pooling = dict(model.named_parameters())["query_pooling.cross_attention.in_proj_weight"]
    assert pooling.grad is not None and torch.isfinite(pooling.grad).all()
