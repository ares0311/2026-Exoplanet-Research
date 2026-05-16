"""Tests for Skills/notebook_generator.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Skills.notebook_generator import _build_notebook_dict, generate_notebook  # noqa: E402


def test_returned_path_exists(tmp_path: Path) -> None:
    out = tmp_path / "TIC_123.ipynb"
    path = generate_notebook(123, output_path=out)
    assert path.exists()


def test_file_is_valid_json(tmp_path: Path) -> None:
    out = tmp_path / "nb.ipynb"
    generate_notebook(456, output_path=out)
    json.loads(out.read_text())


def test_nbformat_is_4() -> None:
    nb = _build_notebook_dict(1, "TESS", None, None, 5.0)
    assert nb["nbformat"] == 4


def test_cells_is_list() -> None:
    nb = _build_notebook_dict(1, "TESS", None, None, 5.0)
    assert isinstance(nb["cells"], list)


def test_at_least_one_code_cell() -> None:
    nb = _build_notebook_dict(1, "TESS", None, None, 5.0)
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) >= 1


def test_tic_id_in_cell_source() -> None:
    nb = _build_notebook_dict(99999, "TESS", None, None, 5.0)
    sources = " ".join(c["source"] for c in nb["cells"])
    assert "99999" in sources


def test_mission_in_cell_source() -> None:
    nb = _build_notebook_dict(1, "Kepler", None, None, 5.0)
    sources = " ".join(c["source"] for c in nb["cells"])
    assert "Kepler" in sources


def test_stellar_radius_in_source_when_provided() -> None:
    nb = _build_notebook_dict(1, "TESS", 1.5, None, 5.0)
    sources = " ".join(c["source"] for c in nb["cells"])
    assert "1.5" in sources


def test_output_path_respected(tmp_path: Path) -> None:
    out = tmp_path / "custom" / "nb.ipynb"
    path = generate_notebook(1, output_path=out)
    assert path == out
    assert out.exists()


def test_default_path_uses_tic_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    path = generate_notebook(77777)
    assert "77777" in path.name
