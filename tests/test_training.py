"""Optimizer groups, LR schedule, config overrides, and a 3-step train/resume run."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

from conftest import requires_tokenizer
from hyper3_clip.models import Hyper3CLIP, Hyper3CLIPConfig
from hyper3_clip.training import RunConfig, load_run_config
from hyper3_clip.training.config import DEFAULT_NO_DECAY_PARAMS, apply_overrides
from hyper3_clip.training.optim import CosineWithWarmup, build_optimizer, split_parameter_groups
from hyper3_clip.training.trainer import Trainer, build_dataloader


# --------------------------------------------------------------- schedule
def test_cosine_with_warmup_shape() -> None:
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(2, 2))], lr=1.0)
    scheduler = CosineWithWarmup(optimizer, warmup_steps=4, total_steps=12, base_lr=1.0)
    warmup = [scheduler.lr_at(step) for step in range(4)]
    assert warmup == pytest.approx([0.25, 0.5, 0.75, 1.0])
    assert scheduler.lr_at(4) == pytest.approx(1.0)
    assert scheduler.lr_at(12) == pytest.approx(0.0, abs=1e-12)
    decay = [scheduler.lr_at(step) for step in range(4, 13)]
    assert all(later <= earlier + 1e-12 for earlier, later in zip(decay, decay[1:], strict=False))

    scheduler.step(2)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.75)
    restored = CosineWithWarmup(optimizer, 0, 1, 0.0)
    restored.load_state_dict(scheduler.state_dict())
    assert (restored.warmup_steps, restored.total_steps, restored.base_lr) == (4, 12, 1.0)


# --------------------------------------------------------------- optimizer
@requires_tokenizer
def test_parameter_groups_follow_the_recipe(tiny_model_config: Hyper3CLIPConfig) -> None:
    model = Hyper3CLIP(tiny_model_config)
    decay, no_decay = split_parameter_groups(model, DEFAULT_NO_DECAY_PARAMS)
    by_id = {id(param): name for name, param in model.named_parameters()}
    decay_names = {by_id[id(param)] for param in decay}
    no_decay_names = {by_id[id(param)] for param in no_decay}

    assert decay_names.isdisjoint(no_decay_names)
    # Frozen sin-cos positions belong to neither group.
    assert "vision_encoder.backbone.pos_embed" not in decay_names | no_decay_names
    # cls_token is 3-D and carries no "norm" in its name, so it is decayed.
    assert "vision_encoder.backbone.cls_token" in decay_names
    assert "image_proj.weight" in decay_names
    assert {"image_proj.bias", "log_curv", "visual_alpha", "global_logit_scale"} <= no_decay_names
    assert all("norm" not in name.lower() for name in decay_names)

    optimizer = build_optimizer(model, no_decay_params=DEFAULT_NO_DECAY_PARAMS, weight_decay=0.2)
    assert [group["weight_decay"] for group in optimizer.param_groups] == [0.2, 0.0]


# --------------------------------------------------------------- config
def test_overrides_use_dotted_paths_and_yaml_values() -> None:
    payload = apply_overrides(
        {"training": {"total_steps": 10}},
        ["training.total_steps=25", "data.max_parts=null", "training.betas=[0.9, 0.95]", "seed=7"],
    )
    assert payload["training"]["total_steps"] == 25
    assert payload["data"]["max_parts"] is None
    assert payload["training"]["betas"] == [0.9, 0.95]
    assert payload["seed"] == 7


def test_run_config_rejects_unknown_sections_and_keys(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"training": {"not_a_training_key": 1.0}}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys in config section 'training'"):
        load_run_config(path)
    path.write_text(yaml.safe_dump({"not_a_section": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown top-level config keys"):
        load_run_config(path)


def test_run_config_round_trips(tmp_path: Path) -> None:
    config = RunConfig()
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8")
    assert load_run_config(path).to_dict() == config.to_dict()


# --------------------------------------------------------------- train/resume
def _tiny_run_config(config: Hyper3CLIPConfig, shard: Path, output_dir: Path, total_steps: int) -> RunConfig:
    model_section = config.to_dict()
    objective_section = model_section.pop("objective")
    return RunConfig.from_dict(
        {
            "project": {"name": "hyper3-clip-test", "experiment": "tiny"},
            "seed": 3,
            "output_dir": str(output_dir),
            "model": model_section,
            "objective": objective_section,
            "training": {
                "total_steps": total_steps,
                "batch_size": 2,
                "lr": 1e-4,
                "warmup_steps": 1,
                "amp": False,
                "log_interval": 1,
                "ckpt_interval": 0,
                "resume": True,
                "find_unused_parameters": False,
            },
            "data": {
                "tarfiles": [str(shard)],
                "num_workers": 0,
                "pin_memory": False,
                "shuffle_buffer": 4,
                "max_parts": 3,
                "deterministic_transforms": True,
            },
        }
    )


@requires_tokenizer
def test_three_steps_then_save_and_resume(
    tmp_path: Path, tiny_model_config: Hyper3CLIPConfig, grit_shard: Path
) -> None:
    output_dir = tmp_path / "run"
    config = _tiny_run_config(replace(tiny_model_config), grit_shard, output_dir, total_steps=3)

    trainer = Trainer(config, device=torch.device("cpu"))
    trainer.fit(build_dataloader(config, trainer.raw_model.tokenizer))
    assert trainer.step == 3
    checkpoint = trainer.save()
    assert checkpoint.name == "checkpoint_step_3.pt"

    rows = [json.loads(line) for line in (output_dir / "train_log.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows] == [1, 2, 3]
    assert all("loss" in row and "contrastive" in row and "entailment" in row for row in rows)
    assert all(row["lr"] > 0 for row in rows)

    # A fresh trainer picks the newest checkpoint up and restores step + weights.
    resumed_config = _tiny_run_config(replace(tiny_model_config), grit_shard, output_dir, total_steps=5)
    resumed = Trainer(resumed_config, device=torch.device("cpu"))
    assert resumed.resume() == 3
    assert resumed.step == 3
    for (name, before), after in zip(
        trainer.raw_model.state_dict().items(), resumed.raw_model.state_dict().values(), strict=True
    ):
        assert torch.equal(before, after), name

    resumed.fit(build_dataloader(resumed_config, resumed.raw_model.tokenizer))
    assert resumed.step == 5

    # A training checkpoint is also a valid `from_pretrained` source: its
    # `config` section carries the model and objective sections verbatim.
    loaded = Hyper3CLIP.from_pretrained(checkpoint)
    assert loaded.config == tiny_model_config
    assert torch.equal(loaded.log_curv, trainer.raw_model.log_curv)


@requires_tokenizer
def test_resume_beyond_total_steps_is_rejected(
    tmp_path: Path, tiny_model_config: Hyper3CLIPConfig, grit_shard: Path
) -> None:
    from hyper3_clip.training import run_training

    output_dir = tmp_path / "run"
    config = _tiny_run_config(replace(tiny_model_config), grit_shard, output_dir, total_steps=2)
    trainer = Trainer(config, device=torch.device("cpu"))
    trainer.step = 2
    trainer.save()

    with pytest.raises(ValueError, match="is not below training.total_steps"):
        run_training(config)
