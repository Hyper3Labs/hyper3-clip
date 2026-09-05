"""Suite driver: load a checkpoint once, run every task, write one JSON each.

``EvalRunner`` loads the model with :meth:`Hyper3CLIP.from_pretrained`, reads
``image_size`` / ``max_text_length`` off ``model.config`` (never off the suite),
and dispatches each task by its ``name``:

===========================  =================================================
``coco_karpathy_retrieval``  paper Table 2, COCO 2014 Karpathy test
``flickr30k_retrieval``      paper Table 2, Flickr30K test
``coco_val2017_retrieval``   the Avg R@10 convention of the Table 5 ablation
``imagenet_zero_shot``       paper Table 3, "IN" column
``imagenet_hierarchical``    paper Table 2, TIE / LCA / J / H-P / H-R
``imagefolder_zero_shot``    paper Tables 3 and 6, the other 15 datasets
``multilabel_zero_shot``     paper Table 4, VOC / COCO mAP
``hierarchy_entailment``     paper Table 5, HierarCaps-style AP / AUROC
===========================  =================================================

When a suite contains both ImageNet tasks with the same root and the same
``max_items``, one fused forward pass produces both records, so the Table 2
hierarchy row and the Table 3 ImageNet column can never come from different
predictions (see ``docs/evaluation.md``).

Each record carries a ``cache_key`` over the suite, the task, the checkpoint
signature, the device type, the precision and the ``--max-items`` override; a
re-run whose key matches an existing file is skipped unless ``force=True``.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from hyper3_clip.evaluation.config import SuiteSpec, TaskSpec
from hyper3_clip.models.hyper3_clip import Hyper3CLIP

__all__ = ["EvalRunner", "ModelSpec", "TASK_NAMES"]

#: Every task ``name`` the runner can dispatch.
TASK_NAMES: tuple[str, ...] = (
    "coco_karpathy_retrieval",
    "flickr30k_retrieval",
    "coco_val2017_retrieval",
    "imagenet_zero_shot",
    "imagenet_hierarchical",
    "imagefolder_zero_shot",
    "multilabel_zero_shot",
    "hierarchy_entailment",
)


@dataclass(frozen=True)
class ModelSpec:
    """The checkpoint under evaluation: a local path or a Hugging Face repo id."""

    id: str
    checkpoint: str
    group: str | None = None
    notes: str | None = None


class EvalRunner:
    """Run one suite against one checkpoint and write per-task JSON records."""

    def __init__(
        self,
        suite: SuiteSpec,
        output_dir: str | Path,
        *,
        device: str | torch.device | None = None,
        precision: str | None = None,
        batch_size: int | None = None,
        max_items: int | None = None,
        force: bool = False,
    ) -> None:
        self.suite = suite
        self.output_dir = Path(output_dir)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.precision = str(precision or suite.defaults.get("precision", "fp32"))
        self.batch_size = batch_size
        self.max_items = max_items
        self.force = force

    # -- entry point ------------------------------------------------------
    def run(self, model_spec: ModelSpec, model: Hyper3CLIP | None = None) -> list[Path]:
        """Evaluate every task and return the paths of the records written or reused."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if model is None:
            model = Hyper3CLIP.from_pretrained(model_spec.checkpoint, map_location=self.device, strict=True)
        model.eval()
        tokenizer = _tokenizer_for(model)
        written: list[Path] = []
        handled = self._run_fused_imagenet(model_spec, model, tokenizer, written)
        for task in self.suite.tasks:
            if task.id in handled:
                continue
            written.append(self._run_task(model_spec, model, tokenizer, task))
        return written

    # -- one task ---------------------------------------------------------
    def _run_task(self, model_spec: ModelSpec, model: Hyper3CLIP, tokenizer: Any, task: TaskSpec) -> Path:
        output_path = self._task_output_path(model_spec.id, task.id)
        cache_key = self._cache_key(model_spec, task)
        if self._cached(output_path, cache_key):
            print(f"cache_hit: {model_spec.id}.{task.id}")
            return output_path
        started = time.monotonic()
        with _autocast(self.device, self.precision):
            results = self._evaluate(model, tokenizer, task)
        elapsed = time.monotonic() - started
        self._write(output_path, model_spec, task, results, elapsed, cache_key)
        print(f"completed: {model_spec.id}.{task.id} elapsed_seconds={elapsed:.1f}")
        return output_path

    def _evaluate(self, model: Hyper3CLIP, tokenizer: Any, task: TaskSpec) -> dict[str, float]:
        from hyper3_clip.evaluation import classification, hierarchy_entailment, imagenet, multilabel, retrieval

        options = self._options(task)
        data = task.data
        shared = self._shared_kwargs(model, options)
        batch_size = shared["batch_size"]
        max_items = shared["max_items"]

        if task.name in ("coco_karpathy_retrieval", "flickr30k_retrieval", "coco_val2017_retrieval"):
            if task.name == "coco_karpathy_retrieval":
                dataset: Any = retrieval.CocoKarpathyRetrievalDataset(
                    data["coco2014_root"],
                    split=str(data.get("coco_karpathy_split", "test")),
                    image_size=shared["image_size"],
                    max_items=max_items,
                    image_normalization=shared["image_normalization"],
                )
            elif task.name == "flickr30k_retrieval":
                dataset = retrieval.Flickr30kRetrievalDataset(
                    data["flickr30k_root"],
                    split=str(data.get("flickr30k_split", "test")),
                    image_size=shared["image_size"],
                    max_items=max_items,
                    image_normalization=shared["image_normalization"],
                )
            else:
                dataset = retrieval.CocoVal2017RetrievalDataset(
                    data["coco2017_root"],
                    image_size=shared["image_size"],
                    max_items=max_items,
                    image_normalization=shared["image_normalization"],
                )
            return retrieval.evaluate_caption_retrieval(
                model,
                dataset,
                self.device,
                tokenizer=tokenizer,
                max_text_length=shared["max_text_length"],
                batch_size=batch_size,
            )

        if task.name in ("imagenet_zero_shot", "imagenet_hierarchical"):
            zero_shot, hierarchy = imagenet.evaluate_imagenet(
                model,
                data["imagenet_val_root"],
                self.device,
                assets_root=data.get("imagenet_hierarchy_assets_root"),
                hierarchy=task.name == "imagenet_hierarchical",
                tokenizer=tokenizer,
                batch_size=batch_size,
                image_size=shared["image_size"],
                image_normalization=shared["image_normalization"],
                max_text_length=shared["max_text_length"],
                max_items=max_items,
                num_workers=shared["num_workers"],
            )
            return hierarchy if task.name == "imagenet_hierarchical" else zero_shot

        if task.name == "imagefolder_zero_shot":
            return classification.evaluate_imagefolder_zero_shot(
                model,
                data["root"],
                self.device,
                tokenizer=tokenizer,
                prompts=data.get("prompts"),
                prompt_set=data.get("prompt_set"),
                prompt_regime=data.get("prompt_regime"),
                class_names=data.get("class_names"),
                class_names_key=data.get("class_names_key"),
                batch_size=batch_size,
                image_size=shared["image_size"],
                image_normalization=shared["image_normalization"],
                max_text_length=shared["max_text_length"],
                max_items=max_items,
                num_workers=shared["num_workers"],
            )

        if task.name == "multilabel_zero_shot":
            from hyper3_clip.evaluation.class_names import load_class_name_list

            samples = multilabel.load_multilabel_samples(
                data["manifest_path"],
                image_root=data.get("image_root"),
                max_items=max_items,
            )
            return multilabel.evaluate_multilabel_zero_shot(
                model,
                samples,
                self.device,
                class_names=load_class_name_list(data.get("class_names_path"), data.get("class_names")),
                tokenizer=tokenizer,
                prompts=tuple(data.get("prompts") or ("a photo of a {}.",)),
                batch_size=batch_size,
                text_batch_size=int(options["text_batch_size"]) if options.get("text_batch_size") else None,
                image_size=shared["image_size"],
                image_normalization=shared["image_normalization"],
                max_text_length=shared["max_text_length"],
            )

        if task.name == "hierarchy_entailment":
            return hierarchy_entailment.evaluate_hierarchy_entailment(
                model,
                data["annotations_path"],
                self.device,
                image_root=data.get("image_root"),
                tokenizer=tokenizer,
                score=str(data.get("score", "entailment_score")),
                max_negatives_per_image=data.get("max_negatives_per_image", 100),
                batch_size=batch_size,
                pair_batch_size=int(options.get("pair_batch_size", 8192)),
                image_size=shared["image_size"],
                image_normalization=shared["image_normalization"],
                max_text_length=shared["max_text_length"],
                max_items=max_items,
            )

        raise ValueError(f"Unsupported task name {task.name!r}; expected one of {list(TASK_NAMES)}")

    # -- fused ImageNet ---------------------------------------------------
    def _run_fused_imagenet(
        self,
        model_spec: ModelSpec,
        model: Hyper3CLIP,
        tokenizer: Any,
        written: list[Path],
    ) -> set[str]:
        from hyper3_clip.evaluation import imagenet

        zero_task = _find(self.suite.tasks, "imagenet_zero_shot")
        hierarchy_task = _find(self.suite.tasks, "imagenet_hierarchical")
        if zero_task is None or hierarchy_task is None:
            return set()
        if zero_task.data.get("imagenet_val_root") != hierarchy_task.data.get("imagenet_val_root"):
            return set()
        zero_options = self._options(zero_task)
        hierarchy_options = self._options(hierarchy_task)
        if self._max_items(zero_options) != self._max_items(hierarchy_options):
            return set()

        zero_key = self._cache_key(model_spec, zero_task)
        hierarchy_key = self._cache_key(model_spec, hierarchy_task)
        zero_path = self._task_output_path(model_spec.id, zero_task.id)
        hierarchy_path = self._task_output_path(model_spec.id, hierarchy_task.id)
        zero_cached = self._cached(zero_path, zero_key)
        hierarchy_cached = self._cached(hierarchy_path, hierarchy_key)
        if zero_cached and hierarchy_cached:
            print(f"cache_hit: {model_spec.id}.{zero_task.id}")
            print(f"cache_hit: {model_spec.id}.{hierarchy_task.id}")
            written.extend([zero_path, hierarchy_path])
            return {zero_task.id, hierarchy_task.id}

        shared = self._shared_kwargs(model, zero_options)
        started = time.monotonic()
        with _autocast(self.device, self.precision):
            zero_metrics, hierarchy_metrics = imagenet.evaluate_imagenet(
                model,
                zero_task.data["imagenet_val_root"],
                self.device,
                assets_root=hierarchy_task.data.get("imagenet_hierarchy_assets_root"),
                hierarchy=True,
                tokenizer=tokenizer,
                batch_size=shared["batch_size"],
                image_size=shared["image_size"],
                image_normalization=shared["image_normalization"],
                max_text_length=shared["max_text_length"],
                max_items=shared["max_items"],
                num_workers=shared["num_workers"],
            )
        elapsed = time.monotonic() - started
        self._write(zero_path, model_spec, zero_task, zero_metrics, elapsed, zero_key)
        self._write(hierarchy_path, model_spec, hierarchy_task, hierarchy_metrics, elapsed, hierarchy_key)
        written.extend([zero_path, hierarchy_path])
        print(f"completed: {model_spec.id}.imagenet_fused elapsed_seconds={elapsed:.1f}")
        return {zero_task.id, hierarchy_task.id}

    # -- plumbing ---------------------------------------------------------
    def _options(self, task: TaskSpec) -> dict[str, Any]:
        return {**self.suite.defaults, **task.options}

    def _max_items(self, options: Mapping[str, Any]) -> int | None:
        if self.max_items is not None:
            return int(self.max_items)
        value = options.get("max_items")
        return int(value) if value is not None else None

    def _shared_kwargs(self, model: Hyper3CLIP, options: Mapping[str, Any]) -> dict[str, Any]:
        config = model.config
        return {
            "image_size": int(getattr(config, "image_size", 224)),
            "max_text_length": int(getattr(config, "max_text_length", 77)),
            "image_normalization": str(options.get("image_normalization", "imagenet")),
            "batch_size": int(self.batch_size or options.get("batch_size", 128)),
            "num_workers": int(options.get("num_workers", 0)),
            "max_items": self._max_items(options),
        }

    def _task_output_path(self, model_id: str, task_id: str) -> Path:
        return self.output_dir / _safe_name(model_id) / f"{_safe_name(task_id)}.json"

    def _cache_key(self, model_spec: ModelSpec, task: TaskSpec) -> str:
        payload = {
            "suite": self.suite.name,
            "task": asdict(task),
            "task_options": self._options(task),
            "model": {"id": model_spec.id, "checkpoint": _checkpoint_signature(model_spec.checkpoint)},
            "device_type": self.device.type,
            "precision": self.precision,
            "max_items_override": self.max_items,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    def _cached(self, output_path: Path, cache_key: str) -> bool:
        if self.force or not output_path.exists():
            return False
        try:
            payload = json.loads(output_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False
        return payload.get("cache_key") == cache_key

    def _write(
        self,
        output_path: Path,
        model_spec: ModelSpec,
        task: TaskSpec,
        results: Mapping[str, float],
        elapsed_seconds: float,
        cache_key: str,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "suite": self.suite.name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": {
                "id": model_spec.id,
                "group": model_spec.group,
                "checkpoint": model_spec.checkpoint,
                "notes": model_spec.notes,
                "checkpoint_signature": _checkpoint_signature(model_spec.checkpoint),
            },
            "task": {
                "id": task.id,
                "name": task.name,
                "dataset": task.data.get("dataset"),
                "data": task.data,
                "options": self._options(task),
            },
            "device": str(self.device),
            "precision": self.precision,
            "elapsed_seconds": elapsed_seconds,
            "cache_key": cache_key,
            "results": dict(results),
        }
        output_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _tokenizer_for(model: Hyper3CLIP) -> Any:
    from hyper3_clip.evaluation.encoding import resolve_tokenizer

    return resolve_tokenizer(model)


def _find(tasks: tuple[TaskSpec, ...], name: str) -> TaskSpec | None:
    for task in tasks:
        if task.name == name:
            return task
    return None


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)


def _checkpoint_signature(checkpoint: str | Path) -> dict[str, Any]:
    """Identify a checkpoint by ``(size, mtime)`` when local, by name when a hub id."""
    path = Path(checkpoint)
    if path.exists():
        target = path / "model.safetensors" if path.is_dir() else path
        if target.exists():
            stat = target.stat()
            return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        return {"path": str(path), "size": None, "mtime_ns": None}
    return {"path": str(checkpoint), "size": None, "mtime_ns": None}


@contextmanager
def _autocast(device: torch.device, precision: str):
    """``fp32`` (the default) is a no-op; ``bf16``/``fp16`` only apply on CUDA."""
    if device.type != "cuda" or precision == "fp32":
        yield
        return
    if precision in {"bf16", "bfloat16"}:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            yield
        return
    if precision in {"fp16", "float16"}:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            yield
        return
    raise ValueError(f"Unsupported precision: {precision}")
