"""Tiny config layer: YAML file + dotted command-line overrides.

Deliberately dependency-free (no hydra / omegaconf) so the repo stays easy to
run on a lab machine or in Colab.

    cfg = load_config("configs/toy.yaml", ["selector.name=submodular", "budget.K=8"])
    cfg.selector.name  ->  "submodular"
    cfg["selector"]["name"]  ->  "submodular"
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


class Config(dict):
    """dict with attribute access, recursively applied."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:  # pragma: no cover - debugging aid
            raise AttributeError(f"no config key {key!r}") from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    @classmethod
    def from_obj(cls, obj: Any) -> Any:
        if isinstance(obj, Mapping):
            return cls({k: cls.from_obj(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [cls.from_obj(v) for v in obj]
        return obj

    def to_plain(self) -> Any:
        def unwrap(obj: Any) -> Any:
            if isinstance(obj, Mapping):
                return {k: unwrap(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [unwrap(v) for v in obj]
            if isinstance(obj, Path):
                return str(obj)
            return obj

        return unwrap(self)

    def dumps(self) -> str:
        return json.dumps(self.to_plain(), indent=2, sort_keys=True, default=str)


def _coerce(text: str) -> Any:
    """Turn a CLI string into the obvious Python value."""
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [_coerce(part.strip()) for part in inner.split(",")]
    return text


def _expand(obj: Any) -> Any:
    """Expand ``${VAR}`` and ``~`` inside every string in the config.

    Datasets live outside the repo (they are large, and on a OneDrive-synced
    machine they must not be inside the synced tree), but a committed config
    must not hardcode one person's absolute path. So configs refer to
    ``${AFS_DATA}`` and each person points that at their own data directory.
    An unset variable is left verbatim, and the loaders then fail with a clear
    path in the message rather than silently reading the wrong place.
    """
    import os

    if isinstance(obj, str):
        return os.path.expanduser(os.path.expandvars(obj))
    if isinstance(obj, Mapping):
        return {k: _expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand(v) for v in obj]
    return obj


def _merge(base: dict, extra: Mapping) -> dict:
    for key, value in extra.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def apply_override(cfg: dict, override: str) -> None:
    """Apply one ``a.b.c=value`` string in place."""
    if "=" not in override:
        raise ValueError(f"override must look like key.sub=value, got {override!r}")
    dotted, raw = override.split("=", 1)
    node = cfg
    parts = dotted.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, Config())
    node[parts[-1]] = _coerce(raw)


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> Config:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    # support a single `_base_: other.yaml` inheritance hop, relative to this file
    base_name = raw.pop("_base_", None)
    if base_name:
        with (path.parent / base_name).open("r", encoding="utf-8") as fh:
            base = yaml.safe_load(fh) or {}
        raw = _merge(base, raw)

    cfg = Config.from_obj(_expand(raw))
    for override in overrides:
        apply_override(cfg, override)
    cfg.setdefault("_config_path", str(path))
    return cfg
