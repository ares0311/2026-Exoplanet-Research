"""Version metadata tests."""
from __future__ import annotations

import tomllib
from pathlib import Path

import exo_toolkit


def test_source_version_matches_pyproject() -> None:
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    assert exo_toolkit.__version__ == data["project"]["version"]
