"""Every shipped config loads, and the ablation rows differ only where documented."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from hyper3_clip.training import load_run_config

CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs" / "paper"
PAPER_CONFIG = CONFIG_ROOT / "hyper3_clip_vitb_500k.yaml"
BASE_CONFIG = CONFIG_ROOT / "ablations" / "base_80k.yaml"

#: Keys every ablation row is expected to change relative to base_80k, on top
#: of the per-row entries below.  Each row is a separate 20k-step continuation
#: of the same 80k checkpoint, so it needs its own name, directory and budget.
COMMON_ABLATION_KEYS = frozenset(
    {
        "project.experiment",
        "output_dir",
        "training.total_steps",
        "training.resume_from",
    }
)

#: row -> the controls it is allowed to change, beyond COMMON_ABLATION_KEYS.
ABLATION_KEYS: dict[str, frozenset[str]] = {
    "no_query": frozenset(),
    "query_image_text_only": frozenset({"objective.query_image_text_weight"}),
    "visual_hierarchy_only": frozenset({"objective.visual_hierarchy_weight"}),
    "text_hierarchy_only": frozenset({"objective.text_hierarchy_weight"}),
    "full_objective_100k": frozenset(
        {
            "objective.query_image_text_weight",
            "objective.visual_hierarchy_weight",
            "objective.text_hierarchy_weight",
        }
    ),
    "mean_pooled_visual": frozenset(
        {
            "objective.query_image_text_weight",
            "objective.visual_hierarchy_weight",
            "objective.text_hierarchy_weight",
            "model.query_pooling_mode",
        }
    ),
    "reversed_visual_order": frozenset(
        {
            "objective.query_image_text_weight",
            "objective.visual_hierarchy_weight",
            "objective.text_hierarchy_weight",
            "objective.reverse_visual_order",
        }
    ),
    "shuffled_text_parents": frozenset(
        {
            "objective.query_image_text_weight",
            "objective.visual_hierarchy_weight",
            "objective.text_hierarchy_weight",
            "model.query_parent_mode",
        }
    ),
}

ALL_CONFIGS = sorted(CONFIG_ROOT.rglob("*.yaml"))


def _flatten(payload: Any, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in payload.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{path}."))
        else:
            flat[path] = tuple(value) if isinstance(value, list) else value
    return flat


def _diff(left: dict[str, Any], right: dict[str, Any]) -> set[str]:
    return {key for key in set(left) | set(right) if left.get(key) != right.get(key)}


def test_every_config_is_present() -> None:
    names = {path.relative_to(CONFIG_ROOT).as_posix() for path in ALL_CONFIGS}
    assert "hyper3_clip_vitb_500k.yaml" in names
    assert {f"ablations/{row}.yaml" for row in ABLATION_KEYS} | {"ablations/base_80k.yaml"} <= names
    assert {f"max_parts_sweep/max_parts_{cap}.yaml" for cap in (2, 3, 4, 5, 6, "all")} <= names


@pytest.mark.parametrize("path", ALL_CONFIGS, ids=lambda p: p.relative_to(CONFIG_ROOT).as_posix())
def test_config_loads_into_the_dataclasses(path: Path) -> None:
    config = load_run_config(path)
    model = config.model_config()
    assert model.embed_dim == 512
    assert model.query_pooling
    assert config.data.type == "processed_grit"
    assert config.data.tarfiles


def test_paper_config_matches_the_recipe() -> None:
    config = load_run_config(PAPER_CONFIG)
    model = config.model_config()
    assert (model.vision_backbone, model.vision_global_pool) == ("vit_base_patch16_224", "token")
    assert model.vision_sincos2d_pos and model.vision_norm_layer == "layer_norm"
    assert model.text_model_name == "openai/clip-vit-base-patch32"
    assert (model.curv_init, model.learn_curv) == (1.0, True)
    assert (model.query_num_heads, model.query_mlp_ratio) == (8, 4.0)
    assert model.query_pooling_mode == "conditioned" and model.query_parent_mode == "true"

    objective = model.objective
    assert objective.entail_weight == 0.2
    assert (objective.inter_aperture_scale, objective.intra_aperture_scale) == (0.7, 1.2)
    assert objective.calibration_alpha == 10.0

    training = config.training
    assert (training.total_steps, training.global_batch_size) == (500_000, 768)
    assert (training.lr, training.weight_decay, training.betas) == (5e-4, 0.2, (0.9, 0.98))
    assert (training.warmup_steps, training.amp, training.max_grad_norm) == (4000, True, 1.0)
    assert (training.ckpt_interval, training.log_interval) == (10_000, 20)

    data = config.data
    assert data.max_parts == 5
    assert data.train_transform == "tight_crop_color_jitter_gray"
    assert data.image_normalization == "imagenet"
    assert (data.max_sentences, data.max_phrases, data.max_queries_per_image) == (5, 30, 6)
    assert data.use_text_boxes


def test_base_ablation_disables_every_query_relation() -> None:
    objective = load_run_config(BASE_CONFIG).model_config().objective
    assert objective.query_image_text_weight == 0.0
    assert objective.visual_hierarchy_weight == 0.0
    assert objective.text_hierarchy_weight == 0.0
    training = load_run_config(BASE_CONFIG).training
    assert training.total_steps == 80_000
    assert training.cosine_horizon == 100_000


@pytest.mark.parametrize("row", sorted(ABLATION_KEYS))
def test_ablation_row_changes_only_its_documented_keys(row: str) -> None:
    base = _flatten(load_run_config(BASE_CONFIG).to_dict())
    ablation = _flatten(load_run_config(CONFIG_ROOT / "ablations" / f"{row}.yaml").to_dict())
    changed = _diff(base, ablation)
    allowed = COMMON_ABLATION_KEYS | ABLATION_KEYS[row]
    assert changed <= allowed, f"{row} also changes {sorted(changed - allowed)}"
    assert COMMON_ABLATION_KEYS <= changed
    assert ABLATION_KEYS[row] <= changed


@pytest.mark.parametrize("row", sorted(ABLATION_KEYS))
def test_ablation_row_resumes_the_shared_base_checkpoint(row: str) -> None:
    config = load_run_config(CONFIG_ROOT / "ablations" / f"{row}.yaml")
    base = load_run_config(BASE_CONFIG)
    assert config.training.resume_from == f"{base.output_dir}/checkpoint_step_80000.pt"
    assert config.training.total_steps == 100_000
    assert config.training.cosine_horizon == base.training.cosine_horizon


@pytest.mark.parametrize("cap", [2, 3, 4, 5, 6, "all"])
def test_max_parts_sweep_changes_only_the_cap(cap: int | str) -> None:
    paper = _flatten(load_run_config(PAPER_CONFIG).to_dict())
    sweep_path = CONFIG_ROOT / "max_parts_sweep" / f"max_parts_{cap}.yaml"
    sweep = _flatten(load_run_config(sweep_path).to_dict())
    allowed = {
        "project.experiment",
        "output_dir",
        "data.max_parts",
        "training.total_steps",
        "training.scheduler_total_steps",
        "training.ckpt_interval",
    }
    assert _diff(paper, sweep) <= allowed
    assert load_run_config(sweep_path).data.max_parts == (None if cap == "all" else cap)
    assert load_run_config(sweep_path).training.total_steps == 10_000
