"""The Hyper3-CLIP training loop (paper Sec. 4, "Training details").

One optimizer step is: zero grads -> set the LR for this step index -> move the
batch to the device -> fp16 autocast forward -> ``loss / grad_accum`` ->
``scaler.scale(loss).backward()`` -> unscale, clip to ``max_grad_norm`` -> step
-> update.  The objective itself is evaluated in float32, and the query pooling
and every Lorentz operation disable autocast explicitly, so only the encoder
towers actually run in half precision.

Logging goes to stdout and to ``train_log.jsonl`` under ``output_dir``, at step
1 and every ``log_interval`` steps thereafter; every scalar the objective
returns is appended to the row.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset

from hyper3_clip.data.collate import GroundedCollator
from hyper3_clip.data.grit import ProcessedGritDataset
from hyper3_clip.models.hyper3_clip import Hyper3CLIP
from hyper3_clip.training.checkpoint import load_checkpoint, resolve_resume_path, save_checkpoint, set_seed
from hyper3_clip.training.config import RunConfig
from hyper3_clip.training.distributed import (
    barrier,
    destroy_distributed,
    get_rank,
    get_world_size,
    init_distributed,
    is_main_process,
    log_main,
    resolve_device,
    wrap_ddp,
)
from hyper3_clip.training.optim import CosineWithWarmup, build_optimizer, parameter_counts

__all__ = ["JsonlLogger", "Trainer", "build_dataloader", "run_training"]


class JsonlLogger:
    """Append one JSON object per line to a log file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, row: Mapping[str, Any]) -> None:
        """Append ``row`` to the file."""
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(row)) + "\n")


def local_batch_size(config: RunConfig) -> int:
    """Return this rank's batch size from ``batch_size`` or ``global_batch_size``."""
    if config.training.batch_size is not None:
        return int(config.training.batch_size)
    world_size = get_world_size()
    if config.training.global_batch_size % world_size != 0:
        raise ValueError("training.global_batch_size must be divisible by the world size")
    return config.training.global_batch_size // world_size


def build_dataloader(config: RunConfig, tokenizer: Any) -> DataLoader:
    """Build the processed-GRIT loader for this rank."""
    data = config.data
    if data.type != "processed_grit":
        raise ValueError(f"Unsupported data.type {data.type!r}")
    dataset = ProcessedGritDataset(
        tarfiles=data.tarfiles,
        image_size=data.image_size,
        seed=config.seed,
        shuffle_buffer=data.shuffle_buffer,
        max_parts=data.max_parts,
        train_transform=data.train_transform,
        image_normalization=data.image_normalization,
        deterministic_transforms=data.deterministic_transforms,
    )
    kwargs: dict[str, Any] = {}
    if data.num_workers > 0:
        kwargs["persistent_workers"] = bool(data.persistent_workers)
        if data.prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(data.prefetch_factor)
    return DataLoader(
        dataset,
        batch_size=local_batch_size(config),
        sampler=None,
        shuffle=not isinstance(dataset, IterableDataset),
        num_workers=data.num_workers,
        pin_memory=bool(data.pin_memory),
        drop_last=True,
        collate_fn=GroundedCollator(tokenizer, data.max_text_length, data.query_config()),
        **kwargs,
    )


class Trainer:
    """Owns the model, optimizer, schedule and the step loop for one run."""

    def __init__(self, config: RunConfig, *, device: torch.device | None = None) -> None:
        self.config = config
        self.device = resolve_device() if device is None else device
        set_seed(config.seed + get_rank())

        self.raw_model = Hyper3CLIP(config.model_config()).to(self.device)
        self.model = wrap_ddp(
            self.raw_model, self.device, find_unused_parameters=config.training.find_unused_parameters
        )
        self.optimizer = build_optimizer(
            self.raw_model,
            lr=config.training.lr,
            weight_decay=config.training.weight_decay,
            betas=config.training.betas,
            eps=config.optimizer.eps,
            no_decay_params=config.optimizer.no_decay_params,
        )
        self.scheduler = CosineWithWarmup(
            self.optimizer,
            warmup_steps=config.training.warmup_steps,
            total_steps=config.training.cosine_horizon,
            base_lr=config.training.lr,
        )
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=bool(config.training.amp))
        self.output_dir = Path(config.output_dir)
        self.logger = JsonlLogger(self.output_dir / "train_log.jsonl")
        self.step = 0

    # -- lifecycle --------------------------------------------------------
    def resume(self) -> int:
        """Restore from the resolved checkpoint, if any, and return the start step."""
        path = resolve_resume_path(self.config.training, self.output_dir)
        if path is None:
            self.step = 0
            return 0
        self.step = load_checkpoint(
            path,
            self.raw_model,
            self.optimizer,
            self.scheduler,
            self.scaler,
            self.device,
            model_only=self.config.training.resume_model_only,
            strict_model=self.config.training.resume_strict_model,
            reset_step=self.config.training.resume_reset_step,
            base_seed=self.config.seed,
        )
        if self.config.training.resume_retarget_scheduler:
            self.scheduler.warmup_steps = int(self.config.training.warmup_steps)
            self.scheduler.total_steps = int(self.config.training.cosine_horizon)
            self.scheduler.base_lr = float(self.config.training.lr)
        log_main(f"resumed from {path} at step {self.step}")
        return self.step

    def save(self, name: str | None = None) -> Path:
        """Write a checkpoint for the current step.  Collective on every rank."""
        filename = name or f"checkpoint_step_{self.step}.pt"
        path = self.output_dir / filename
        save_checkpoint(
            path, self.step, self.raw_model, self.optimizer, self.scheduler, self.scaler, self.config.to_dict()
        )
        if is_main_process():
            (self.output_dir / "latest_checkpoint.txt").write_text(f"{path}\n", encoding="utf-8")
        return path

    # -- the step loop ----------------------------------------------------
    def train_step(self, batch: Mapping[str, Tensor], *, micro_step: int) -> tuple[dict[str, Tensor], float | None]:
        """Run one micro-batch; step the optimizer at the end of an accumulation cycle."""
        training = self.config.training
        accum = max(1, training.grad_accum_steps)
        if micro_step % accum == 0:
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step(self.step)

        non_blocking = training.non_blocking_transfer
        batch = {key: value.to(self.device, non_blocking=non_blocking) for key, value in batch.items()}
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=bool(training.amp)):
            nodes = self.model(batch, step=self.step)
        losses = self.raw_model.loss_from_nodes(nodes)
        (self.scaler.scale(losses["loss"] / accum)).backward()

        if (micro_step + 1) % accum != 0:
            return losses, None

        grad_norm: float | None = None
        if training.max_grad_norm > 0:
            self.scaler.unscale_(self.optimizer)
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), training.max_grad_norm).detach().cpu().item()
            )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.step += 1
        return losses, grad_norm

    def fit(self, dataloader: DataLoader) -> int:
        """Run until ``training.total_steps``, logging and checkpointing on the way."""
        training = self.config.training
        self.model.train()
        micro_step = 0
        batch_global = local_batch_size(self.config) * get_world_size() * max(1, training.grad_accum_steps)
        last_time = time.perf_counter()
        self.optimizer.zero_grad(set_to_none=True)

        while self.step < training.total_steps:
            for batch in dataloader:
                if self.step >= training.total_steps:
                    break
                losses, grad_norm = self.train_step(batch, micro_step=micro_step)
                micro_step += 1
                if (micro_step % max(1, training.grad_accum_steps)) != 0:
                    continue

                now = time.perf_counter()
                step_seconds = now - last_time
                last_time = now
                if self.step == 1 or self.step % training.log_interval == 0:
                    self._log(losses, grad_norm=grad_norm, step_seconds=step_seconds, batch_global=batch_global)
                if training.ckpt_interval > 0 and self.step % training.ckpt_interval == 0:
                    self.save()
        return self.step

    # -- logging ----------------------------------------------------------
    def _log(
        self,
        losses: Mapping[str, Tensor],
        *,
        grad_norm: float | None,
        step_seconds: float,
        batch_global: int,
    ) -> None:
        training = self.config.training
        row: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "step": self.step,
            "lr": self.optimizer.param_groups[0]["lr"],
            "grad_norm": grad_norm,
            "step_time_seconds": step_seconds,
            "samples_per_second": batch_global / max(step_seconds, 1e-12),
            "samples_seen": self.step * batch_global,
            "progress": self.step / max(1, training.total_steps),
            "rank": get_rank(),
            "world_size": get_world_size(),
            "global_batch_size": batch_global,
        }
        for key, value in losses.items():
            if torch.is_tensor(value) and value.numel() == 1:
                row[key] = _scalar(value)
        if self.device.type == "cuda":
            row["cuda_max_memory_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024**2)
        if is_main_process():
            self.logger.write(row)
            print(" ".join(f"{key}={value}" for key, value in row.items()), flush=True)


def _scalar(value: Tensor) -> float | int:
    detached = value.detach().cpu()
    if detached.dtype is torch.bool:
        return int(detached.item())
    if not detached.is_floating_point():
        return int(detached.item())
    return float(detached.item())


def run_training(config: RunConfig) -> int:
    """Run one training job end to end: init, resume, fit, final checkpoint."""
    init_distributed()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if is_main_process():
        (output_dir / "config.yaml").write_text(
            yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8"
        )

    trainer = Trainer(config)
    counts = parameter_counts(trainer.raw_model)
    log_main(
        f"parameter_counts total={counts['total']} trainable={counts['trainable']} frozen={counts['frozen']}"
    )
    start_step = trainer.resume()
    if start_step >= config.training.total_steps:
        raise ValueError(
            f"Resume start step {start_step} is not below training.total_steps={config.training.total_steps}; "
            "for a warm start set training.resume_model_only=true and training.resume_reset_step=true"
        )

    dataloader = build_dataloader(config, trainer.raw_model.tokenizer)
    final_step = trainer.fit(dataloader)
    barrier()
    trainer.save("checkpoint_final.pt")
    barrier()
    destroy_distributed()
    return final_step
