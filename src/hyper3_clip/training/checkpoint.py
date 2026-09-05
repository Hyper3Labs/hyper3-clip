"""Training checkpoints: format, save, resume.

A checkpoint is a ``torch.save`` pickle holding::

    {
      "step":           int,   completed optimizer steps
      "model":          state dict of the *unwrapped* model (no "module." prefix)
      "optimizer":      AdamW.state_dict()
      "scheduler":      {"warmup_steps", "total_steps", "base_lr"}
      "scaler":         GradScaler.state_dict()
      "config":         the resolved run config as a nested dict
      "rng_by_rank":    list[dict] of per-rank python / numpy / torch / cuda RNG
      "rng_world_size": int
    }

Files are written atomically (``.tmp`` then ``replace``) as
``checkpoint_step_{N}.pt`` every ``training.ckpt_interval`` steps and
``checkpoint_final.pt`` at the end.  Only rank 0 writes, but **every** rank has
to call :func:`save_checkpoint`, because gathering the RNG states is a
collective.

Resume order: ``$RESUME_FROM_CHECKPOINT`` (the env var named by
``training.resume_from_env``), then ``training.resume_from``, then the newest
``checkpoint_step_*.pt`` under ``output_dir`` when ``training.resume`` is set,
then step 0.  The model load is strict by default.  RNG is restored per rank
only when the saved world size matches the current one; otherwise each rank is
reseeded deterministically with ``seed + rank``.
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from hyper3_clip.distributed import get_rank, get_world_size, is_distributed

__all__ = [
    "latest_checkpoint",
    "load_checkpoint",
    "resolve_resume_path",
    "save_checkpoint",
    "set_seed",
]

LOGGER = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Seed ``random``, ``numpy`` and torch (CPU and every CUDA device)."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    path: str | Path,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    config: dict[str, Any],
) -> None:
    """Write a checkpoint atomically.  Collective: call it on every rank."""
    rng_by_rank = _gather_rng_states(_rng_state())
    if get_rank() != 0:
        return
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_name(f"{checkpoint_path.name}.tmp")
    torch.save(
        {
            "step": int(step),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "config": config,
            "rng_by_rank": rng_by_rank,
            "rng_world_size": len(rng_by_rank),
        },
        tmp_path,
    )
    tmp_path.replace(checkpoint_path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    scaler: Any | None,
    device: torch.device | str = "cpu",
    *,
    model_only: bool = False,
    strict_model: bool = True,
    reset_step: bool = False,
    base_seed: int | None = None,
) -> int:
    """Restore a checkpoint and return the step to resume from."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=strict_model)
    if model_only:
        return 0 if reset_step else int(checkpoint["step"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    _restore_rank_rng_state(checkpoint, base_seed=base_seed)
    return int(checkpoint["step"])


def latest_checkpoint(output_dir: str | Path) -> Path | None:
    """Return the ``checkpoint_step_*.pt`` with the largest step, if any."""
    paths = list(Path(output_dir).glob("checkpoint_step_*.pt"))
    if not paths:
        return None
    return max(paths, key=_checkpoint_step)


def resolve_resume_path(training: Any, output_dir: str | Path) -> str | None:
    """Apply the resume-order rule and return the checkpoint path, or ``None``."""
    env_name = getattr(training, "resume_from_env", None)
    if env_name:
        from_env = os.environ.get(str(env_name))
        if from_env:
            return from_env
    if getattr(training, "resume_from", None):
        return str(training.resume_from)
    if getattr(training, "resume", False):
        newest = latest_checkpoint(output_dir)
        if newest is not None:
            return str(newest)
    return None


def _checkpoint_step(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[1])


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _gather_rng_states(local_state: dict[str, Any]) -> list[dict[str, Any]]:
    world_size = get_world_size()
    if world_size == 1 or not is_distributed():
        return [local_state]
    states: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(states, local_state)
    if any(state is None for state in states):
        raise RuntimeError("Failed to gather RNG state from every rank")
    return [state for state in states if state is not None]


def _restore_rank_rng_state(checkpoint: dict[str, Any], *, base_seed: int | None) -> None:
    rank = get_rank()
    world_size = get_world_size()
    if "rng_by_rank" in checkpoint:
        states = checkpoint["rng_by_rank"]
        saved_world_size = int(checkpoint.get("rng_world_size", len(states)))
    elif "rng" in checkpoint:
        states = [checkpoint["rng"]]
        saved_world_size = 1
    else:
        return
    if saved_world_size == world_size and len(states) == world_size:
        _set_rng_state(states[rank])
        return
    seed = int(checkpoint.get("config", {}).get("seed", 0) if base_seed is None else base_seed) + rank
    LOGGER.warning(
        "Checkpoint RNG world size (%d) differs from the current world size (%d); "
        "reseeding rank %d with seed %d instead of restoring RNG state",
        saved_world_size,
        world_size,
        rank,
        seed,
    )
    set_seed(seed)


def _set_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(_cpu_byte_tensor(state["torch"]))
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([_cpu_byte_tensor(cuda_state) for cuda_state in state["cuda"]])


def _cpu_byte_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", dtype=torch.uint8)
    return torch.as_tensor(value, dtype=torch.uint8, device="cpu")
