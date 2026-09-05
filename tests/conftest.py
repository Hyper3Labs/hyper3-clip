"""Shared fixtures: a tiny text-tower directory, a tiny model config, a tar shard."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fixtures.make_grit_fixture import write_grit_fixture  # noqa: E402

TEXT_MODEL_NAME = "openai/clip-vit-base-patch32"


def artifact_dir() -> Path | None:
    """Local copy of the released Hub artifact, or ``None``.

    Set ``HYPER3_CLIP_ARTIFACT_DIR`` to a directory holding the artifact's
    ``config.json`` + ``model.safetensors`` to enable the artifact test; it
    skips otherwise.
    """
    value = os.environ.get("HYPER3_CLIP_ARTIFACT_DIR")
    if not value:
        return None
    directory = Path(value)
    return directory if (directory / "model.safetensors").is_file() else None


def _tokenizer_available() -> bool:
    try:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    except Exception:  # noqa: BLE001 - any failure means "no tokenizer, skip"
        return False
    return True


requires_tokenizer = pytest.mark.skipif(
    not _tokenizer_available(),
    reason=f"the {TEXT_MODEL_NAME} tokenizer is not available offline",
)


@pytest.fixture(scope="session")
def tiny_text_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A local directory holding a 2-layer, 64-dim CLIP text config + the CLIP tokenizer."""
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer, CLIPTextConfig

    try:
        tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    except Exception as error:  # noqa: BLE001
        pytest.skip(f"CLIP tokenizer unavailable: {error}")
    directory = tmp_path_factory.mktemp("tiny_text_tower")
    tokenizer.save_pretrained(directory)
    CLIPTextConfig(
        vocab_size=int(tokenizer.vocab_size),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=77,
    ).save_pretrained(directory)
    return directory


@pytest.fixture(scope="session")
def tiny_model_config(tiny_text_dir: Path):
    """A :class:`Hyper3CLIPConfig` small enough to train a few CPU steps."""
    from hyper3_clip.models import Hyper3CLIPConfig, ObjectiveConfig

    return Hyper3CLIPConfig(
        vision_backbone="vit_tiny_patch16_224",
        vision_global_pool="token",
        vision_sincos2d_pos=True,
        vision_norm_layer="layer_norm",
        text_model_name=str(tiny_text_dir),
        embed_dim=32,
        query_pooling=True,
        query_num_heads=4,
        query_mlp_ratio=2.0,
        objective=ObjectiveConfig(),
        image_size=224,
        max_text_length=77,
    )


@pytest.fixture(scope="session")
def grit_shard(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A four-sample processed-GRIT tar shard."""
    directory = tmp_path_factory.mktemp("grit_fixture")
    return write_grit_fixture(directory / "fixture-000000.tar")
