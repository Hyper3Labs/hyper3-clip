"""Training loop, configuration, optimizer and checkpointing for Hyper3-CLIP."""

from __future__ import annotations

from hyper3_clip.training.checkpoint import latest_checkpoint, load_checkpoint, save_checkpoint, set_seed
from hyper3_clip.training.config import (
    DataConfig,
    OptimizerConfig,
    ProjectConfig,
    RunConfig,
    TrainingConfig,
    apply_overrides,
    load_run_config,
)
from hyper3_clip.training.optim import CosineWithWarmup, build_optimizer, parameter_counts
from hyper3_clip.training.trainer import Trainer, build_dataloader, run_training

__all__ = [
    "CosineWithWarmup",
    "DataConfig",
    "OptimizerConfig",
    "ProjectConfig",
    "RunConfig",
    "Trainer",
    "TrainingConfig",
    "apply_overrides",
    "build_dataloader",
    "build_optimizer",
    "latest_checkpoint",
    "load_checkpoint",
    "load_run_config",
    "parameter_counts",
    "run_training",
    "save_checkpoint",
    "set_seed",
]
