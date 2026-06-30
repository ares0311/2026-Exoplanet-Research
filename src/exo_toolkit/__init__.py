from __future__ import annotations

import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

_PACKAGE_NAME = "exo-toolkit"
_FALLBACK_VERSION = "0.2.7"


def _source_tree_version() -> str | None:
    """Return the committed source-tree version when running from checkout."""
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    try:
        data: dict[str, Any] = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return None

    project = data.get("project")
    if not isinstance(project, dict):
        return None
    if project.get("name") != _PACKAGE_NAME:
        return None
    project_version = project.get("version")
    return project_version if isinstance(project_version, str) else None


def _installed_version() -> str | None:
    """Return installed package metadata version when available."""
    try:
        return version(_PACKAGE_NAME)
    except PackageNotFoundError:
        return None


__version__ = _source_tree_version() or _installed_version() or _FALLBACK_VERSION
