"""Structured configuration: dataclass configs + YAML load + dotlist overrides.

We deliberately avoid a hard Hydra dependency so the lab runs in minimal /
ARM64 environments without the full Hydra+OmegaConf stack. Configs are plain
``@dataclass`` objects (typed, IDE-friendly, ablation-friendly); this module
adds the ergonomics that make Hydra pleasant:

  * YAML <-> dataclass (de)serialization
  * ``key.subkey=value`` dotlist overrides (Hydra-style CLI ergonomics)
  * deep-merge with clear precedence: ``defaults < yaml file < CLI overrides``
  * resolution to a frozen, JSON-serializable dict + content hash for provenance

Research rationale
------------------
A foundation-model lab lives or dies on *reproducibility* and *ablation
velocity*. Typed dataclasses give us static structure (so an ablation is a diff
on a config, not a guess), while the dotlist overrides give us the fast
command-line sweeps SAM-style data-engine iteration depends on.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Type, TypeVar, get_args, get_origin, get_type_hints

import yaml

T = TypeVar("T")


# --------------------------------------------------------------------------- #
# dataclass <-> dict
# --------------------------------------------------------------------------- #
def to_dict(obj: Any) -> Any:
    """Recursively convert a dataclass (or container of them) into plain dicts."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    if isinstance(obj, Path):
        return str(obj)
    return obj


def from_dict(cls: Type[T], data: dict[str, Any]) -> T:
    """Build a (possibly nested) dataclass instance from a plain dict.

    Unknown keys raise, so a typo in a config or override fails loudly instead
    of silently doing nothing -- a common source of irreproducible results.
    """
    if not is_dataclass(cls):
        return data  # type: ignore[return-value]
    hints = get_type_hints(cls)
    field_names = {f.name for f in fields(cls)}
    unknown = set(data) - field_names
    if unknown:
        raise KeyError(
            f"Unknown config key(s) for {cls.__name__}: {sorted(unknown)}. "
            f"Valid keys: {sorted(field_names)}"
        )
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        kwargs[f.name] = _coerce(hints.get(f.name, Any), data[f.name])
    return cls(**kwargs)  # type: ignore[call-arg]


def _coerce(tp: Any, value: Any) -> Any:
    origin = get_origin(tp)
    if is_dataclass(tp) and isinstance(value, dict):
        return from_dict(tp, value)
    if origin in (list, tuple) and isinstance(value, (list, tuple)):
        args = get_args(tp)
        inner = args[0] if args else Any  # first arg; handles list[X], tuple[X], tuple[X, ...]
        seq = [_coerce(inner, v) for v in value]
        return tuple(seq) if origin is tuple else seq
    return value


# --------------------------------------------------------------------------- #
# YAML + overrides
# --------------------------------------------------------------------------- #
def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Top-level YAML in {path} must be a mapping, got {type(data)}")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (override wins)."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def apply_overrides(data: dict[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Apply Hydra-style ``a.b.c=value`` dotlist overrides.

    Values are parsed with YAML semantics so ``lr=1e-4``, ``use_bf16=true`` and
    ``sizes=[1024,2048]`` get correct Python types.
    """
    out = json.loads(json.dumps(data))  # deep copy of plain data
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override '{item}' is not of the form key.path=value")
        key, _, raw = item.partition("=")
        value = yaml.safe_load(raw)
        cursor = out
        parts = key.split(".")
        for p in parts[:-1]:
            cursor = cursor.setdefault(p, {})
            if not isinstance(cursor, dict):
                raise ValueError(f"Cannot descend into non-mapping at '{p}' for override '{item}'")
        cursor[parts[-1]] = value
    return out


def build_config(
    cls: Type[T],
    yaml_path: str | Path | None = None,
    overrides: Iterable[str] | None = None,
) -> T:
    """Resolve ``defaults(cls) < yaml < overrides`` into a typed dataclass."""
    base = to_dict(cls())  # dataclass defaults
    if yaml_path is not None:
        base = deep_merge(base, load_yaml(yaml_path))
    if overrides:
        base = apply_overrides(base, overrides)
    return from_dict(cls, base)


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #
def to_yaml(obj: Any) -> str:
    return yaml.safe_dump(to_dict(obj), sort_keys=True, default_flow_style=False)


def config_hash(obj: Any, length: int = 12) -> str:
    """Stable content hash of a config -- used to dedupe / name experiment runs."""
    canonical = json.dumps(to_dict(obj), sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(canonical.encode()).hexdigest()[:length]


def save_config(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(to_yaml(obj))
