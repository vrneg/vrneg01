"""Reconstruct the repository's nested dataclass configs from checkpoints."""

from __future__ import annotations

import types
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _convert(annotation: Any, value: Any) -> Any:
    if value is None:
        return None
    if annotation is Path:
        return Path(value)
    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin is tuple:
        item_type = arguments[0] if arguments else Any
        return tuple(_convert(item_type, item) for item in value)
    if origin is list:
        item_type = arguments[0] if arguments else Any
        return [_convert(item_type, item) for item in value]
    if origin in (Union, types.UnionType):
        candidates = [candidate for candidate in arguments if candidate is not type(None)]
        return _convert(candidates[0], value) if candidates else value
    if isinstance(annotation, type) and is_dataclass(annotation):
        return dataclass_from_dict(annotation, value)
    return value


def dataclass_from_dict(cls: type, payload: dict[str, Any]) -> Any:
    """Build a dataclass while tolerating fields added after an old checkpoint."""

    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass type")
    hints = get_type_hints(cls)
    known_fields = {field.name for field in fields(cls)}
    kwargs = {
        name: _convert(hints.get(name, Any), value)
        for name, value in payload.items()
        if name in known_fields
    }
    return cls(**kwargs)


def relocate_checkpoint_paths(config: Any) -> Any:
    """Point stale absolute dataset/TabPFN paths at this project checkout.

    Training artifacts can be moved with the repository.  Preserve every configured
    path that still exists, and relocate only known project assets by basename.
    """

    changes: dict[str, Path] = {}
    dataset_path = getattr(config, "dataset_path", None)
    if dataset_path is not None:
        dataset_path = Path(dataset_path)
        relocated = PROJECT_ROOT / "data" / "trainsets" / dataset_path.name
        if not dataset_path.is_dir() and relocated.is_dir():
            changes["dataset_path"] = relocated
    tabpfn_model_path = getattr(config, "tabpfn_model_path", None)
    if tabpfn_model_path is not None:
        tabpfn_model_path = Path(tabpfn_model_path)
        relocated = PROJECT_ROOT / "data" / "tabpfn" / tabpfn_model_path.name
        if not tabpfn_model_path.is_file() and relocated.is_file():
            changes["tabpfn_model_path"] = relocated
    return replace(config, **changes) if changes else config
