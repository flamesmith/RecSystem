from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def _package_versions(names: Iterable[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _git_commit(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def build_manifest(
    *,
    project_root: Path,
    command: list[str],
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    configuration: dict[str, Any],
    statistics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "git_commit": _git_commit(project_root),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": _package_versions(
            ["torch", "transformers", "numpy", "pandas", "Pillow", "PyYAML", "requests"]
        ),
        "inputs": inputs,
        "outputs": outputs,
        "configuration": configuration,
        "statistics": statistics,
    }


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".part")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(output_path)
