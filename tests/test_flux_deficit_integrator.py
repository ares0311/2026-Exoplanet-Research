"""Tests for Skills/flux_deficit_integrator.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from flux_deficit_integrator import format_flux_deficit_result, integrate_flux_deficit


def _box_transit(depth_ppm: float = 1000.0, n: int = 20) -> tuple[list[float], list[float]]:
    """Uniform time grid with a box transit from t=0.4 to t=0.6."""
    times = [i / (n - 1) for i in range(n)]
    flux = [1.0 - depth_ppm / 1e6 if 0.4 <= t <= 0.6 else 1.0 for t in times]
    return times, flux


class TestFluxDeficitIntegrator:
    def test_basic_box_transit(self) -> None:
        t, f = _box_transit(1000.0)
        r = integrate_flux_deficit(t, f, 0.4, 0.6)
        assert r.flag == "OK"
        assert r.deficit_ppm_hours > 0.0

    def test_depth_ppm_hours_correct_order(self) -> None:
        t, f = _box_transit(1000.0)
        r = integrate_flux_deficit(t, f, 0.4, 0.6)
        # depth=1000 ppm, duration ~0.2 days = 4.8 hours => ~4800 ppm·h
        assert r.deficit_ppm_hours > 100.0

    def test_insufficient_data(self) -> None:
        r = integrate_flux_deficit([0.5], [0.999], 0.4, 0.6)
        assert r.flag == "INSUFFICIENT_DATA"

    def test_invalid_window_reversed(self) -> None:
        t, f = _box_transit()
        r = integrate_flux_deficit(t, f, 0.6, 0.4)
        assert r.flag == "INVALID_WINDOW"

    def test_invalid_window_equal(self) -> None:
        t, f = _box_transit()
        r = integrate_flux_deficit(t, f, 0.5, 0.5)
        assert r.flag == "INVALID_WINDOW"

    def test_no_in_transit_points(self) -> None:
        t, f = _box_transit()
        r = integrate_flux_deficit(t, f, 2.0, 3.0)
        assert r.flag == "NO_IN_TRANSIT_POINTS"

    def test_mean_depth_ppm_positive(self) -> None:
        t, f = _box_transit(500.0)
        r = integrate_flux_deficit(t, f, 0.4, 0.6)
        assert r.mean_depth_ppm > 0.0

    def test_duration_hours_positive(self) -> None:
        t, f = _box_transit()
        r = integrate_flux_deficit(t, f, 0.4, 0.6)
        assert r.duration_hours > 0.0

    def test_asymmetry_near_zero_symmetric(self) -> None:
        t, f = _box_transit()
        r = integrate_flux_deficit(t, f, 0.4, 0.6)
        assert abs(r.asymmetry) < 0.5

    def test_n_points_in_window(self) -> None:
        t, f = _box_transit(n=20)
        r = integrate_flux_deficit(t, f, 0.4, 0.6)
        assert r.n_points >= 2

    def test_deeper_transit_larger_deficit(self) -> None:
        t1, f1 = _box_transit(500.0)
        t2, f2 = _box_transit(1000.0)
        r1 = integrate_flux_deficit(t1, f1, 0.4, 0.6)
        r2 = integrate_flux_deficit(t2, f2, 0.4, 0.6)
        assert r2.deficit_ppm_hours > r1.deficit_ppm_hours

    def test_format_returns_string(self) -> None:
        t, f = _box_transit()
        r = integrate_flux_deficit(t, f, 0.4, 0.6)
        s = format_flux_deficit_result(r)
        assert isinstance(s, str)
        assert "deficit" in s.lower()
