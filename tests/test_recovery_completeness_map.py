"""Tests for Skills.recovery_completeness_map."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from Skills.recovery_completeness_map import (
    CompletenessMap,
    build_completeness_map,
    load_completeness_map,
    save_completeness_map,
)


def _lc(n: int = 500) -> tuple[np.ndarray, np.ndarray]:
    time = np.linspace(2458000.0, 2458027.0, n)
    flux = np.ones(n)
    return time, flux


class TestBuildCompletenessMap:
    def test_rates_shape_matches_period_depth_grid(self) -> None:
        time, flux = _lc()
        periods = [5.0, 10.0]
        depths = [500.0, 1000.0]
        cmap = build_completeness_map(periods, depths, time=time, flux=flux, n_per_cell=2)
        assert len(cmap.recovery_rates) == 2
        assert len(cmap.recovery_rates[0]) == 2

    def test_recovery_rates_in_zero_one(self) -> None:
        time, flux = _lc()
        cmap = build_completeness_map([5.0], [1000.0], time=time, flux=flux, n_per_cell=3)
        for row in cmap.recovery_rates:
            for r in row:
                assert 0.0 <= r <= 1.0

    def test_always_recover_fn_gives_rate_one(self) -> None:
        time, flux = _lc()
        cmap = build_completeness_map(
            [5.0], [1000.0], time=time, flux=flux, n_per_cell=5,
            recovery_fn=lambda t, f, p, d, dur: True,
        )
        assert cmap.recovery_rates[0][0] == pytest.approx(1.0)

    def test_never_recover_fn_gives_rate_zero(self) -> None:
        time, flux = _lc()
        cmap = build_completeness_map(
            [5.0], [1000.0], time=time, flux=flux, n_per_cell=5,
            recovery_fn=lambda t, f, p, d, dur: False,
        )
        assert cmap.recovery_rates[0][0] == pytest.approx(0.0)

    def test_target_id_stored(self) -> None:
        time, flux = _lc()
        cmap = build_completeness_map([5.0], [1000.0], time=time, flux=flux, target_id="TIC 1")
        assert cmap.target_id == "TIC 1"

    def test_n_per_cell_stored(self) -> None:
        time, flux = _lc()
        cmap = build_completeness_map([5.0], [1000.0], time=time, flux=flux, n_per_cell=7)
        assert cmap.n_per_cell == 7

    def test_period_and_depth_grids_stored(self) -> None:
        time, flux = _lc()
        periods = [3.0, 7.0, 14.0]
        depths = [200.0, 1000.0]
        cmap = build_completeness_map(periods, depths, time=time, flux=flux, n_per_cell=2)
        assert cmap.period_grid == periods
        assert cmap.depth_grid_ppm == depths

    def test_returns_completeness_map_instance(self) -> None:
        time, flux = _lc()
        cmap = build_completeness_map([5.0], [1000.0], time=time, flux=flux, n_per_cell=2)
        assert isinstance(cmap, CompletenessMap)


class TestSaveLoadCompletenessMap:
    def test_save_writes_json(self, tmp_path: Path) -> None:
        time, flux = _lc()
        cmap = build_completeness_map([5.0], [1000.0], time=time, flux=flux, n_per_cell=2)
        path = save_completeness_map(cmap, tmp_path / "cmap.json")
        assert path.exists()
        data = json.loads(path.read_text())
        assert "period_grid" in data
        assert "recovery_rates" in data

    def test_load_roundtrip(self, tmp_path: Path) -> None:
        time, flux = _lc()
        cmap = build_completeness_map(
            [5.0, 10.0], [500.0, 1000.0], time=time, flux=flux, n_per_cell=2,
            target_id="TIC 42",
        )
        path = save_completeness_map(cmap, tmp_path / "cmap.json")
        cmap2 = load_completeness_map(path)
        assert cmap2.period_grid == cmap.period_grid
        assert cmap2.depth_grid_ppm == cmap.depth_grid_ppm
        assert cmap2.target_id == cmap.target_id
        assert cmap2.n_per_cell == cmap.n_per_cell

    def test_recovery_rates_preserved_after_roundtrip(self, tmp_path: Path) -> None:
        time, flux = _lc()
        cmap = build_completeness_map(
            [5.0], [1000.0], time=time, flux=flux, n_per_cell=4,
            recovery_fn=lambda *_: True,
        )
        path = save_completeness_map(cmap, tmp_path / "cmap.json")
        cmap2 = load_completeness_map(path)
        assert cmap2.recovery_rates[0][0] == pytest.approx(cmap.recovery_rates[0][0])

    def test_json_has_target_id_key(self, tmp_path: Path) -> None:
        time, flux = _lc()
        cmap = build_completeness_map([5.0], [1000.0], time=time, flux=flux, target_id="TIC 7")
        path = save_completeness_map(cmap, tmp_path / "cmap.json")
        data = json.loads(path.read_text())
        assert data["target_id"] == "TIC 7"
