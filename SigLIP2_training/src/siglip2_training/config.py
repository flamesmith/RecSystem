from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as source:
        value = yaml.safe_load(source)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def config_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def gib_to_bytes(value: float | int) -> int:
    return int(float(value) * 1024**3)


def mib_to_bytes(value: float | int) -> int:
    return int(float(value) * 1024**2)


def resolve_from(base: str | Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(base) / path
