"""Model construction, config round-trip and checkpoint interchange."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from conftest import artifact_dir, requires_tokenizer
from hyper3_clip.models import Hyper3CLIP, Hyper3CLIPConfig, ObjectiveConfig
from hyper3_clip.models.checkpoints import detect_checkpoint_flavour, remap_state_dict


# --------------------------------------------------------------- config
def test_config_round_trips_through_yaml(tmp_path: Path) -> None:
    config = Hyper3CLIPConfig(
        embed_dim=64,
        query_pooling_mode="mean",
        query_parent_mode="shuffled",
        objective=ObjectiveConfig(entail_weight=0.5),
    )
    path = tmp_path / "config.yaml"
    config.to_yaml(path)
    restored = Hyper3CLIPConfig.from_yaml(path)
    assert restored == config
    assert restored.objective.entail_weight == 0.5


def test_config_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="unknown Hyper3CLIPConfig keys"):
        Hyper3CLIPConfig.from_dict({"vision_backbone": "vit_base_patch16_224", "vision_pretrained": False})


def test_config_validates_controls() -> None:
    with pytest.raises(ValueError, match="query_pooling_mode"):
        Hyper3CLIPConfig(query_pooling_mode="average")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="query_parent_mode"):
        Hyper3CLIPConfig(query_parent_mode="random")  # type: ignore[arg-type]


# --------------------------------------------------------------- key mapping
def test_hub_logit_scale_broadcasts_to_the_three_temperatures() -> None:
    target = {"global_logit_scale", "local_logit_scale", "global_local_logit_scale", "log_curv"}
    hub = {"logit_scale": torch.tensor(4.6), "log_curv": torch.tensor(0.0)}
    assert detect_checkpoint_flavour(hub) == "hub"
    mapped, _ = remap_state_dict(hub, target)
    assert set(mapped) == target
    assert all(float(mapped[name]) == pytest.approx(4.6) for name in target if name.endswith("logit_scale"))


def test_text_model_prefix_is_aligned_to_the_installed_transformers() -> None:
    """``transformers`` 5 dropped the ``text_model`` wrapper; both spellings load."""
    four_x = {"text_encoder.backbone.text_model.embeddings.weight": torch.zeros(1)}
    five_x = {"text_encoder.backbone.embeddings.weight": torch.zeros(1)}
    for state, target in ((four_x, set(five_x)), (five_x, set(four_x))):
        mapped, renames = remap_state_dict(state, target)
        assert set(mapped) == target
        assert len(renames) == 1


def test_public_state_dict_is_left_alone() -> None:
    state = {"global_logit_scale": torch.zeros(()), "image_proj.weight": torch.zeros(2, 2)}
    assert detect_checkpoint_flavour(state) == "public"
    mapped, renames = remap_state_dict(state, set(state))
    assert set(mapped) == set(state)
    assert renames == {}


# --------------------------------------------------------------- round trip
@requires_tokenizer
def test_save_and_load_pretrained_round_trip(tmp_path: Path, tiny_model_config: Hyper3CLIPConfig) -> None:
    model = Hyper3CLIP(tiny_model_config)
    directory = model.save_pretrained(tmp_path / "ckpt")
    assert (directory / "config.yaml").is_file()
    assert (directory / "model.safetensors").is_file()

    restored = Hyper3CLIP.from_pretrained(directory)
    assert restored.config == tiny_model_config
    for (name, expected), (_, actual) in zip(
        model.state_dict().items(), restored.state_dict().items(), strict=True
    ):
        assert torch.equal(expected, actual), name


@requires_tokenizer
def test_from_pretrained_reports_unmatched_keys(tmp_path: Path, tiny_model_config: Hyper3CLIPConfig) -> None:
    model = Hyper3CLIP(tiny_model_config)
    state = {key: value for key, value in model.state_dict().items() if key != "log_curv"}
    state["not_a_parameter"] = torch.zeros(())
    checkpoint = tmp_path / "broken.pt"
    torch.save({"step": 0, "model": state, "config": {}}, checkpoint)
    with pytest.raises(RuntimeError) as error:
        Hyper3CLIP.from_pretrained(checkpoint, config=tiny_model_config)
    message = str(error.value)
    assert "log_curv" in message
    assert "not_a_parameter" in message


# --------------------------------------------------------------- inference
@requires_tokenizer
def test_encode_shapes_and_similarity_agrees_with_the_inner_product(
    tiny_model_config: Hyper3CLIPConfig,
) -> None:
    from hyper3_clip.geometry import pairwise_lorentz_inner

    model = Hyper3CLIP(tiny_model_config).eval()
    dim = tiny_model_config.embed_dim
    images = torch.randn(3, 3, 224, 224)
    ids = torch.randint(0, 100, (4, 12))
    with torch.no_grad():
        image_feats = model.encode_image(images)
        text_feats = model.encode_text(ids)
        tangent = model.encode_image_tangent(images)
        scores = model.similarity(image_feats, text_feats)
        inner = pairwise_lorentz_inner(image_feats, text_feats)
    assert image_feats.shape == (3, dim + 1)
    assert text_feats.shape == (4, dim + 1)
    assert tangent.shape == (3, dim)
    # -d_L is a strictly decreasing function of -<x, y>_L, so ranking agrees.
    assert torch.equal(scores.argsort(dim=1), inner.argsort(dim=1))


# --------------------------------------------------------------- released artifact
@pytest.mark.skipif(artifact_dir() is None, reason="set HYPER3_CLIP_ARTIFACT_DIR to run this")
@requires_tokenizer
def test_released_artifact_loads_strictly_and_encodes() -> None:
    """The published checkpoint loads with no missing or unexpected key."""
    directory = artifact_dir()
    assert directory is not None
    model = Hyper3CLIP.from_pretrained(directory, strict=True).eval()

    shipped = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    assert model.config.embed_dim == shipped["embed_dim"]
    # The released artifact used average pooling, so timm keeps `fc_norm`.
    assert model.config.vision_global_pool == "avg"
    assert not model.config.query_pooling

    from PIL import Image

    from hyper3_clip.models import build_retrieval_transform, build_tokenizer

    transform = build_retrieval_transform(model.config)
    tokenizer = build_tokenizer(model.config)
    image = transform(Image.new("RGB", (320, 240), (110, 140, 90)))[None]
    tokens = tokenizer(["a red bicycle"], padding=True, truncation=True, max_length=77, return_tensors="pt")
    with torch.no_grad():
        image_feats = model.encode_image(image)
        text_feats = model.encode_text(tokens["input_ids"], tokens["attention_mask"])
    assert image_feats.shape == (1, model.config.embed_dim + 1)
    assert text_feats.shape == (1, model.config.embed_dim + 1)
    assert torch.isfinite(model.similarity(image_feats, text_feats)).all()


@requires_tokenizer
@pytest.mark.slow
def test_paper_configuration_has_the_reported_parameter_inventory() -> None:
    """The paper recipe builds exactly the 368-tensor model the paper reports."""
    from hyper3_clip.training import load_run_config
    from hyper3_clip.training.config import DEFAULT_NO_DECAY_PARAMS
    from hyper3_clip.training.optim import parameter_counts, split_parameter_groups

    config_path = Path(__file__).resolve().parents[1] / "configs/paper/hyper3_clip_vitb_500k.yaml"
    model = Hyper3CLIP(load_run_config(config_path).model_config())

    assert len(list(model.parameters())) == 368
    assert parameter_counts(model) == {
        "total": 157_101_318,
        "trainable": 156_950_022,
        "frozen": 151_296,  # the frozen sin-cos pos_embed
    }
    decay, no_decay = split_parameter_groups(model, DEFAULT_NO_DECAY_PARAMS)
    assert (len(decay), sum(p.numel() for p in decay)) == (131, 156_736_768)
    assert (len(no_decay), sum(p.numel() for p in no_decay)) == (236, 213_254)

    keys = set(model.state_dict())
    # global_pool="token" keeps `norm`, not `fc_norm`.
    assert "vision_encoder.backbone.norm.weight" in keys
    assert "vision_encoder.backbone.fc_norm.weight" not in keys
    assert {"global_logit_scale", "local_logit_scale", "global_local_logit_scale"} <= keys
    assert "logit_scale" not in keys
    assert sum(key.startswith("query_pooling.") for key in keys) == 12
    assert not model.vision_encoder.backbone.pos_embed.requires_grad
