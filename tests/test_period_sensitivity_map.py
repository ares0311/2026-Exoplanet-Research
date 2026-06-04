"""Tests for Skills/period_sensitivity_map.py."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from period_sensitivity_map import compute_sensitivity_map, format_sensitivity_map


class TestComputeSensitivityMap:
    def test_basic_map(self) -> None:
        r = compute_sensitivity_map(baseline_days=365.0, rms_ppm=500.0)
        assert r.flag == "OK"
        assert r.n_periods > 0
        assert r.n_depths > 0

    def test_deep_transit_detectable(self) -> None:
        r = compute_sensitivity_map(
            baseline_days=365.0,
            rms_ppm=200.0,
            depth_grid_ppm=[30000.0],
            period_grid_days=[1.0],
        )
        assert r.detectable_grid[0][0]

    def test_shallow_long_period_not_detectable(self) -> None:
        r = compute_sensitivity_map(
            baseline_days=30.0,
            rms_ppm=5000.0,
            depth_grid_ppm=[100.0],
            period_grid_days=[100.0],
        )
        assert not r.detectable_grid[0][0]

    def test_invalid_baseline(self) -> None:
        r = compute_sensitivity_map(baseline_days=0.0, rms_ppm=500.0)
        assert r.flag == "INVALID_BASELINE"

    def test_invalid_rms(self) -> None:
        r = compute_sensitivity_map(baseline_days=365.0, rms_ppm=-1.0)
        assert r.flag == "INVALID_RMS"

    def test_grid_dimensions(self) -> None:
        periods = [1.0, 5.0, 10.0]
        depths = [1000.0, 5000.0]
        r = compute_sensitivity_map(
            baseline_days=365.0,
            rms_ppm=500.0,
            period_grid_days=periods,
            depth_grid_ppm=depths,
        )
        assert r.n_periods == 3
        assert r.n_depths == 2
        assert len(r.snr_grid) == 3
        assert len(r.snr_grid[0]) == 2

    def test_snr_positive(self) -> None:
        r = compute_sensitivity_map(
            baseline_days=365.0, rms_ppm=500.0,
            period_grid_days=[1.0], depth_grid_ppm=[10000.0],
        )
        assert r.snr_grid[0][0] > 0

    def test_higher_rms_fewer_detections(self) -> None:
        r_low = compute_sensitivity_map(baseline_days=365.0, rms_ppm=100.0)
        r_high = compute_sensitivity_map(baseline_days=365.0, rms_ppm=10000.0)
        assert r_low.n_detectable_cells >= r_high.n_detectable_cells

    def test_custom_snr_threshold(self) -> None:
        r_strict = compute_sensitivity_map(
            baseline_days=365.0, rms_ppm=500.0, snr_threshold=100.0
        )
        r_loose = compute_sensitivity_map(
            baseline_days=365.0, rms_ppm=500.0, snr_threshold=1.0
        )
        assert r_loose.n_detectable_cells >= r_strict.n_detectable_cells

    def test_n_detectable_cells_range(self) -> None:
        r = compute_sensitivity_map(baseline_days=365.0, rms_ppm=500.0)
        assert 0 <= r.n_detectable_cells <= r.n_periods * r.n_depths

    def test_n_transits_override(self) -> None:
        r = compute_sensitivity_map(
            baseline_days=365.0, rms_ppm=500.0,
            n_transits_per_period=10,
            period_grid_days=[10.0], depth_grid_ppm=[1000.0],
        )
        assert r.flag == "OK"

    def test_format_output(self) -> None:
        r = compute_sensitivity_map(
            baseline_days=365.0, rms_ppm=500.0,
            period_grid_days=[1.0, 10.0],
            depth_grid_ppm=[1000.0, 10000.0],
        )
        s = format_sensitivity_map(r)
        assert "|" in s
        assert "detectable" in s.lower() or "Y" in s or "n(" in s

    def test_period_grid_preserved(self) -> None:
        periods = [2.0, 7.0, 15.0]
        r = compute_sensitivity_map(
            baseline_days=365.0, rms_ppm=500.0, period_grid_days=periods
        )
        assert list(r.period_grid_days) == periods
