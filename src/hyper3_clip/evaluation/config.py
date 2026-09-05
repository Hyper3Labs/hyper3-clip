"""Evaluation suite configuration: task specs and ``${VAR}`` path substitution.

A suite YAML is ``{name, defaults, tasks}``; every task is
``{id, name, data, options}`` where ``name`` selects the evaluator and ``data``
carries its dataset arguments.  ``defaults`` is merged under each task's
``options``.

Dataset locations never appear in a committed suite: they are written as
``${HYPER3_...}`` placeholders and resolved from a ``--paths`` YAML layered over
the process environment.  ``configs/eval/local_paths.example.yaml`` lists every
variable the shipped suites use, with placeholder values only.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "SuiteSpec",
    "TaskSpec",
    "filter_tasks",
    "load_suite",
    "load_variables",
    "resolve_templates",
]

_TEMPLATE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


@dataclass(frozen=True)
class TaskSpec:
    """One evaluation task: an evaluator ``name`` plus its ``data`` arguments."""

    id: str
    name: str
    data: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SuiteSpec:
    """A named list of tasks with shared ``defaults``."""

    name: str
    defaults: dict[str, Any]
    tasks: tuple[TaskSpec, ...]


def load_variables(paths_file: str | Path | None = None) -> dict[str, str]:
    """Environment variables, overlaid with the entries of a ``--paths`` YAML."""
    variables = dict(os.environ)
    if paths_file is None:
        return variables
    path = Path(paths_file)
    if not path.exists():
        raise FileNotFoundError(f"Paths file does not exist: {path}")
    payload = _load_yaml(path)
    for key, value in payload.items():
        variables[str(key)] = str(value)
    return variables


def resolve_templates(value: Any, variables: Mapping[str, str]) -> Any:
    """Recursively replace ``${NAME}`` in every string; a missing name raises."""
    if isinstance(value, dict):
        return {key: resolve_templates(item, variables) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_templates(item, variables) for item in value]
    if isinstance(value, str):
        return _TEMPLATE.sub(lambda match: _lookup(match.group(1), variables), value)
    return value


def _lookup(key: str, variables: Mapping[str, str]) -> str:
    if key not in variables:
        raise KeyError(
            f"Missing template variable {key!r}; set it in the --paths YAML "
            f"(see configs/eval/local_paths.example.yaml) or in the environment"
        )
    return str(variables[key])


def load_suite(path: str | Path, variables: Mapping[str, str] | None = None) -> SuiteSpec:
    """Load one suite YAML with its placeholders resolved."""
    payload = _load_yaml(Path(path))
    resolved = resolve_templates(payload, dict(variables) if variables is not None else load_variables())
    tasks = tuple(_task_from_mapping(item) for item in resolved.get("tasks", ()))
    if not tasks:
        raise ValueError(f"Suite {path} must define at least one task")
    return SuiteSpec(
        name=str(resolved.get("name") or Path(path).stem),
        defaults=dict(resolved.get("defaults") or {}),
        tasks=tasks,
    )


def filter_tasks(suite: SuiteSpec, task_ids: Iterable[str] | None) -> SuiteSpec:
    """Restrict a suite to the named task ids (or names); unknown ids raise."""
    wanted = {str(value) for value in task_ids} if task_ids else set()
    if not wanted:
        return suite
    selected = tuple(task for task in suite.tasks if task.id in wanted or task.name in wanted)
    known = {task.id for task in selected} | {task.name for task in selected}
    missing = sorted(wanted - known)
    if missing:
        raise ValueError(f"Unknown task ids: {', '.join(missing)}")
    return SuiteSpec(name=suite.name, defaults=suite.defaults, tasks=selected)


def _task_from_mapping(item: Mapping[str, Any]) -> TaskSpec:
    name = str(item["name"])
    return TaskSpec(
        id=str(item.get("id") or name),
        name=name,
        data=dict(item.get("data") or {}),
        options=dict(item.get("options") or {}),
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return payload
