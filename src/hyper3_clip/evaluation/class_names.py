"""Class-name lists for the zero-shot tables (paper Tables 3, 4 and 6).

``assets/class_names_hycoclip.py`` is the official HyCoCLIP list, one entry per
``class_names_key`` used by ``configs/eval/paper_table_3.yaml``.  It is loaded
by file path, never imported as a module of this package.

ImageNet is the exception: its prompt names come from torchvision's
``_IMAGENET_CATEGORIES``, which is the list that produced the paper's rows.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

__all__ = [
    "ASSETS_DIR",
    "imagenet_class_names",
    "load_class_name_table",
    "load_class_name_list",
    "read_label_to_wnid",
    "resolve_class_names",
]

#: Directory holding the vendored evaluation assets (see ``assets/PROVENANCE.md``).
ASSETS_DIR = Path(__file__).resolve().parents[0] / "assets"

_CLASS_NAMES_MODULE = ASSETS_DIR / "class_names_hycoclip.py"


@lru_cache(maxsize=1)
def load_class_name_table() -> dict[str, tuple[str, ...]]:
    """Load the vendored ``CLASS_NAMES`` mapping, keyed by dataset."""
    spec = importlib.util.spec_from_file_location("hyper3_clip_vendored_class_names", _CLASS_NAMES_MODULE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load class names from {_CLASS_NAMES_MODULE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payload: Any = getattr(module, "CLASS_NAMES", None)
    if not isinstance(payload, dict):
        raise ValueError(f"Class-name module must define a CLASS_NAMES mapping: {_CLASS_NAMES_MODULE}")
    return {str(key): tuple(str(name) for name in value) for key, value in payload.items()}


def load_class_name_list(
    path: str | Path | None = None,
    class_names: Sequence[str] | None = None,
) -> list[str] | None:
    """Read a flat class-name list for the multi-label tables (paper Table 4).

    A ``.txt`` file is one class per non-empty line, in file order; ``.json``
    is a list or an object with a ``class_names`` / ``classes`` / ``labels``
    key; ``.csv`` / ``.tsv`` uses the first present of the
    ``class_name`` / ``class`` / ``label`` / ``name`` columns.  Order is always
    preserved because it defines the score-matrix column order.
    """
    if class_names is not None:
        return [str(name) for name in class_names]
    if path is None:
        return None
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [str(item) for item in payload]
        if isinstance(payload, dict):
            for key in ("class_names", "classes", "labels"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [str(item) for item in value]
        raise ValueError(f"Unsupported class-name JSON payload in {source}")
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with source.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter=delimiter))
        for key in ("class_name", "class", "label", "name"):
            values = [row.get(key) for row in rows if row.get(key)]
            if values:
                return [str(item) for item in values]
        raise ValueError(f"Class-name table must contain one of class_name/class/label/name: {source}")
    return [line.strip() for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]


def imagenet_class_names(folder_classes: Sequence[str], imagenet_val_root: str | Path) -> list[str]:
    """Prompt names for a 1000-class ImageNet ``ImageFolder``.

    Only a folder of 1000 WNID directories is remapped onto torchvision's
    category list; anything else is prompted with its own folder names, which
    is what makes tiny fixtures work.  When
    ``<root>/imagenet_label_to_wnid.tsv`` exists the folder order is reordered
    into official label order through it, otherwise folder order is assumed to
    already be official order.
    """
    from torchvision.models._meta import _IMAGENET_CATEGORIES

    names = list(folder_classes)
    if len(names) != 1000 or not all(looks_like_wnid(name) for name in names):
        return names
    official = list(_IMAGENET_CATEGORIES)
    wnid_to_label = read_label_to_wnid(imagenet_val_root)
    if wnid_to_label is None:
        return official
    return [official[wnid_to_label[name]] for name in names]


def read_label_to_wnid(imagenet_val_root: str | Path) -> dict[str, int] | None:
    """Parse ``<root>/imagenet_label_to_wnid.tsv`` (``"<label>\\t<wnid>"``) if present."""
    path = Path(imagenet_val_root) / "imagenet_label_to_wnid.tsv"
    if not path.exists():
        return None
    mapping: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        label, wnid = line.split("\t", maxsplit=1)
        mapping[wnid.strip()] = int(label)
    return mapping


def looks_like_wnid(name: str) -> bool:
    """True for a WordNet id such as ``n01440764``."""
    return len(name) == 9 and name.startswith("n") and name[1:].isdigit()


def resolve_class_names(
    folder_classes: Sequence[str],
    *,
    class_names: Sequence[str] | None = None,
    class_names_key: str | None = None,
) -> list[str]:
    """Resolve the prompt names for an ``ImageFolder`` zero-shot task.

    Precedence is explicit ``class_names`` > ``class_names_key`` (looked up in
    the vendored table) > the folder names themselves.  The materialized
    ImageFolder layout names its class directories ``f"{index:04d}_{slug}"`` so
    that sorted folder order equals the vendored list's order; a length
    mismatch therefore means the folder was built differently and raises.
    """
    if class_names is not None:
        resolved = [str(name) for name in class_names]
    elif class_names_key is not None:
        table = load_class_name_table()
        if class_names_key not in table:
            raise ValueError(f"Unknown class_names_key {class_names_key!r}; expected one of {sorted(table)}")
        resolved = list(table[class_names_key])
    else:
        resolved = [str(name) for name in folder_classes]
    if len(resolved) != len(folder_classes):
        raise ValueError(f"Resolved {len(resolved)} class names for {len(folder_classes)} dataset classes")
    return resolved
