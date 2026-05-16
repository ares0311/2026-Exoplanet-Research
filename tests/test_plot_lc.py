"""Tests for Skills/plot_lc.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Skills.plot_lc import phase_fold, plot_all, plot_candidate  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row(**kwargs: object) -> dict:
    base = {
        "candidate_id": "TIC1_01",
        "period_days": 5.0,
        "epoch_bjd": 2458600.0,
        "depth_ppm": 1000.0,
        "snr": 10.0,
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# phase_fold
# ---------------------------------------------------------------------------


class TestPhaseFold:
    def test_output_shape_matches_input(self) -> None:
        t = np.linspace(0, 20, 200)
        f = np.ones(200)
        ph, fl = phase_fold(t, f, period=5.0, epoch=0.0)
        assert ph.shape == (200,) and fl.shape == (200,)

    def test_phase_range(self) -> None:
        t = np.linspace(0, 100, 1000)
        f = np.ones(1000)
        ph, _ = phase_fold(t, f, period=7.3, epoch=1.0)
        assert ph.min() >= -0.5
        assert ph.max() < 0.5

    def test_sorted_by_phase(self) -> None:
        t = np.random.default_rng(0).uniform(0, 50, 500)
        f = np.ones(500)
        ph, _ = phase_fold(t, f, period=5.0, epoch=0.0)
        assert np.all(np.diff(ph) >= 0)

    def test_zero_period_raises(self) -> None:
        with pytest.raises(ValueError, match="period"):
            phase_fold(np.array([0.0, 1.0]), np.array([1.0, 1.0]), period=0.0, epoch=0.0)

    def test_flux_is_permutation_of_input(self) -> None:
        t = np.array([0.0, 2.5, 5.0, 7.5])
        f = np.array([1.0, 2.0, 3.0, 4.0])
        _, fl = phase_fold(t, f, period=5.0, epoch=0.0)
        assert set(fl.tolist()) == set(f.tolist())


# ---------------------------------------------------------------------------
# plot_candidate
# ---------------------------------------------------------------------------


class TestPlotCandidate:
    def test_returns_path_when_matplotlib_available(self, tmp_path: Path) -> None:
        pytest.importorskip("matplotlib")
        row = _row()
        p = plot_candidate(row, output_dir=tmp_path)
        assert p is not None
        assert p.exists()
        assert p.suffix == ".png"

    def test_none_when_matplotlib_missing(self, tmp_path: Path) -> None:
        with patch.dict("sys.modules", {"matplotlib": None, "matplotlib.pyplot": None}):
            row = _row()
            result = plot_candidate(row, output_dir=tmp_path)
        assert result is None

    def test_output_filename_contains_candidate_id(self, tmp_path: Path) -> None:
        pytest.importorskip("matplotlib")
        row = _row(candidate_id="TIC999_02")
        p = plot_candidate(row, output_dir=tmp_path)
        assert p is not None and "TIC999_02" in p.name

    def test_with_time_flux_arrays(self, tmp_path: Path) -> None:
        pytest.importorskip("matplotlib")
        t = np.linspace(0, 30, 300)
        f = np.ones(300)
        row = _row()
        p = plot_candidate(row, output_dir=tmp_path, time=t, flux=f)
        assert p is not None and p.exists()


# ---------------------------------------------------------------------------
# plot_all
# ---------------------------------------------------------------------------


class TestPlotAll:
    def test_plots_all_rows_from_list(self, tmp_path: Path) -> None:
        pytest.importorskip("matplotlib")
        rows = [_row(candidate_id=f"TIC1_0{i}") for i in range(3)]
        json_file = tmp_path / "results.json"
        json_file.write_text(json.dumps(rows))
        paths = plot_all(json_file, output_dir=tmp_path)
        assert len(paths) == 3

    def test_plots_single_dict(self, tmp_path: Path) -> None:
        pytest.importorskip("matplotlib")
        json_file = tmp_path / "result.json"
        json_file.write_text(json.dumps(_row()))
        paths = plot_all(json_file, output_dir=tmp_path)
        assert len(paths) == 1

    def test_empty_list_produces_no_plots(self, tmp_path: Path) -> None:
        pytest.importorskip("matplotlib")
        json_file = tmp_path / "empty.json"
        json_file.write_text(json.dumps([]))
        paths = plot_all(json_file, output_dir=tmp_path)
        assert paths == []
