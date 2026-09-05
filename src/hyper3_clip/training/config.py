"""Nested YAML configuration for a Hyper3-CLIP training run.

A config file has seven sections::

    project:    {name, experiment}
    seed:       int              # per-rank seed is seed + rank
    output_dir: str
    model:      Hyper3CLIPConfig fields, minus `objective`
    objective:  ObjectiveConfig fields
    training:   TrainingConfig fields
    optimizer:  OptimizerConfig fields
    data:       DataConfig fields

``objective`` is a top-level section rather than nested under ``model`` so an
ablation row can override a single loss weight without restating the
architecture; :meth:`RunConfig.model_config` merges the two.

``--override key=value`` uses dotted paths into the same tree
(``--override training.total_steps=100000``); values are parsed as YAML
scalars, so ``true``, ``null``, ``1e-4`` and ``[0.9, 0.98]`` all work.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from hyper3_clip.models.hyper3_clip import Hyper3CLIPConfig
from hyper3_clip.models.objective import ObjectiveConfig

__all__ = [
    "DataConfig",
    "OptimizerConfig",
    "ProjectConfig",
    "RunConfig",
    "TrainingConfig",
    "apply_overrides",
    "load_run_config",
]

#: Parameters excluded from weight decay by leaf name, on top of the structural
#: rules in :mod:`hyper3_clip.training.optim`.
DEFAULT_NO_DECAY_PARAMS = (
    "logit_scale",
    "global_logit_scale",
    "local_logit_scale",
    "global_local_logit_scale",
    "visual_alpha",
    "textual_alpha",
    "log_curv",
)


@dataclass
class ProjectConfig:
    """Run identity, used for the metadata stamp and for logging."""

    name: str = "hyper3-clip"
    experiment: str = "hyper3_clip_vitb_500k"


@dataclass
class TrainingConfig:
    """Optimizer-step schedule, AMP, checkpointing and resume policy."""

    total_steps: int = 500_000
    #: Horizon the cosine is shaped over; defaults to ``total_steps``.  Setting
    #: it larger lets a run stop early without changing the LR curve, which is
    #: how the 80k -> 100k ablations keep the same schedule as the paper run.
    scheduler_total_steps: int | None = None
    global_batch_size: int = 768
    batch_size: int | None = None
    grad_accum_steps: int = 1
    lr: float = 5e-4
    weight_decay: float = 0.2
    betas: tuple[float, float] = (0.9, 0.98)
    warmup_steps: int = 4000
    amp: bool = True
    max_grad_norm: float = 1.0
    log_interval: int = 20
    ckpt_interval: int = 10_000
    resume: bool = True
    resume_from: str | None = None
    resume_from_env: str = "RESUME_FROM_CHECKPOINT"
    resume_model_only: bool = False
    resume_reset_step: bool = False
    resume_strict_model: bool = True
    resume_retarget_scheduler: bool = True
    find_unused_parameters: bool = True
    non_blocking_transfer: bool = True

    def __post_init__(self) -> None:
        self.betas = tuple(float(value) for value in self.betas)  # type: ignore[assignment]
        if len(self.betas) != 2:
            raise ValueError("training.betas must have two entries")
        if self.grad_accum_steps < 1:
            raise ValueError("training.grad_accum_steps must be >= 1")
        if self.warmup_steps < 0:
            raise ValueError("training.warmup_steps must be non-negative")

    @property
    def cosine_horizon(self) -> int:
        """Steps the cosine decays over: ``scheduler_total_steps`` or ``total_steps``."""
        return int(self.scheduler_total_steps or self.total_steps)


@dataclass
class OptimizerConfig:
    """AdamW parameter grouping."""

    no_decay_params: tuple[str, ...] = DEFAULT_NO_DECAY_PARAMS
    eps: float = 1e-8

    def __post_init__(self) -> None:
        self.no_decay_params = tuple(self.no_decay_params)


@dataclass
class DataConfig:
    """Processed-GRIT shards, part cap and query budget."""

    type: str = "processed_grit"
    tarfiles: tuple[str, ...] = ()
    image_size: int = 224
    max_text_length: int = 77
    shuffle_buffer: int = 4000
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = False
    prefetch_factor: int | None = None
    max_parts: int | None = 5
    train_transform: str = "tight_crop_color_jitter_gray"
    image_normalization: str = "imagenet"
    deterministic_transforms: bool = False
    queries_enabled: bool = True
    max_sentences: int = 5
    max_phrases: int = 30
    max_queries_per_image: int = 6
    use_text_boxes: bool = True

    def __post_init__(self) -> None:
        self.tarfiles = tuple(self.tarfiles)

    def query_config(self) -> Any:
        """Return the :class:`~hyper3_clip.data.collate.QueryConfig` for this data section."""
        from hyper3_clip.data.collate import QueryConfig

        return QueryConfig(
            enabled=self.queries_enabled,
            max_sentences=self.max_sentences,
            max_phrases=self.max_phrases,
            max_queries_per_image=self.max_queries_per_image,
            use_text_boxes=self.use_text_boxes,
        )


@dataclass
class RunConfig:
    """A whole training run: project, seed, output directory and the five sections."""

    project: ProjectConfig = field(default_factory=ProjectConfig)
    seed: int = 31
    output_dir: str = "runs/hyper3_clip"
    model: dict[str, Any] = field(default_factory=dict)
    objective: dict[str, Any] = field(default_factory=dict)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def model_config(self) -> Hyper3CLIPConfig:
        """Merge the ``model`` and ``objective`` sections into a model config."""
        payload = dict(self.model)
        payload["objective"] = ObjectiveConfig(**dict(self.objective))
        payload.setdefault("image_size", self.data.image_size)
        payload.setdefault("max_text_length", self.data.max_text_length)
        return Hyper3CLIPConfig.from_dict(payload)

    def to_dict(self) -> dict[str, Any]:
        """Return a YAML-safe nested dict."""
        return {
            "project": asdict(self.project),
            "seed": self.seed,
            "output_dir": self.output_dir,
            "model": dict(self.model),
            "objective": dict(self.objective),
            "training": _yaml_safe(asdict(self.training)),
            "optimizer": _yaml_safe(asdict(self.optimizer)),
            "data": _yaml_safe(asdict(self.data)),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RunConfig:
        """Build a run config from a parsed YAML tree, rejecting unknown keys."""
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError(f"unknown top-level config keys: {unknown}")
        return cls(
            project=ProjectConfig(**_section(payload, "project", ProjectConfig)),
            seed=int(payload.get("seed", 31)),
            output_dir=str(payload.get("output_dir", "runs/hyper3_clip")),
            model=dict(payload.get("model") or {}),
            objective=dict(payload.get("objective") or {}),
            training=TrainingConfig(**_section(payload, "training", TrainingConfig)),
            optimizer=OptimizerConfig(**_section(payload, "optimizer", OptimizerConfig)),
            data=DataConfig(**_section(payload, "data", DataConfig)),
        )


def _section(payload: Mapping[str, Any], name: str, cls: type) -> dict[str, Any]:
    section = payload.get(name) or {}
    if not isinstance(section, Mapping):
        raise ValueError(f"config section {name!r} must be a mapping")
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(section) - known)
    if unknown:
        raise ValueError(f"unknown keys in config section {name!r}: {unknown}")
    return dict(section)


def _yaml_safe(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: list(value) if isinstance(value, tuple) else value for key, value in payload.items()}


def apply_overrides(payload: dict[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """Apply ``key.path=value`` strings to a parsed config tree, in order.

    Values are parsed with ``yaml.safe_load``, so they keep their YAML types.
    A path whose parent section does not exist is created.
    """
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"override {override!r} must look like key.path=value")
        dotted, raw_value = override.split("=", 1)
        keys = dotted.strip().split(".")
        node = payload
        for key in keys[:-1]:
            child = node.get(key)
            if not isinstance(child, dict):
                child = {}
                node[key] = child
            node = child
        node[keys[-1]] = yaml.safe_load(raw_value)
    return payload


def load_run_config(path: str | Path, overrides: Sequence[str] = ()) -> RunConfig:
    """Read a YAML config, apply ``--override`` strings and validate it."""
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a YAML mapping")
    payload = apply_overrides(payload, overrides)
    config = RunConfig.from_dict(payload)
    config.model_config()  # fail fast on an invalid model/objective section
    return config
