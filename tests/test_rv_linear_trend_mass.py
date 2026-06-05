"""Tests for Skills/rv_linear_trend_mass.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from rv_linear_trend_mass import compute_rv_trend_companion_mass, format_rv_trend_result


class TestComputeRvTrendCompanionMass:
    def test_ok_flag(self) -> None:
        r = compute_rv_trend_companion_mass(0.05, 1000.0)
        assert r.flag == "OK"

    def test_mass_positive(self) -> None:
        r = compute_rv_trend_companion_mass(0.05, 1000.0)
        assert r.min_companion_mass_mjup > 0.0

    def test_period_lower_bound_positive(self) -> None:
        r = compute_rv_trend_companion_mass(0.05, 1000.0)
        assert r.outer_period_lower_bound_yr > 0.0

    def test_larger_trend_larger_mass(self) -> None:
        r1 = compute_rv_trend_companion_mass(0.01, 1000.0)
        r2 = compute_rv_trend_companion_mass(0.10, 1000.0)
        assert r2.min_companion_mass_mjup > r1.min_companion_mass_mjup

    def test_longer_baseline_longer_period(self) -> None:
        r1 = compute_rv_trend_companion_mass(0.05, 500.0)
        r2 = compute_rv_trend_companion_mass(0.05, 2000.0)
        assert r2.outer_period_lower_bound_yr > r1.outer_period_lower_bound_yr

    def test_companion_class_set(self) -> None:
        r = compute_rv_trend_companion_mass(0.05, 1000.0)
        assert r.companion_class in ("STELLAR", "BROWN_DWARF", "PLANET")

    def test_with_assumed_separation(self) -> None:
        r = compute_rv_trend_companion_mass(0.05, 1000.0, assumed_separation_au=5.0)
        assert r.flag == "OK"
        assert r.min_companion_mass_mjup > 0.0

    def test_mass_in_solar_units_consistent(self) -> None:
        r = compute_rv_trend_companion_mass(0.05, 1000.0)
        assert abs(r.min_companion_mass_mjup * (1.898e27 / 1.989e30)
                   - r.min_companion_mass_msun) < 1e-5

    def test_invalid_baseline(self) -> None:
        r = compute_rv_trend_companion_mass(0.05, 0.0)
        assert r.flag == "INVALID_BASELINE"

    def test_invalid_stellar_mass(self) -> None:
        r = compute_rv_trend_companion_mass(0.05, 1000.0, stellar_mass_msun=0.0)
        assert r.flag == "INVALID_STELLAR_MASS"

    def test_zero_trend(self) -> None:
        r = compute_rv_trend_companion_mass(0.0, 1000.0)
        assert r.flag == "ZERO_TREND"

    def test_result_frozen(self) -> None:
        r = compute_rv_trend_companion_mass(0.05, 1000.0)
        try:
            r.min_companion_mass_mjup = 0.0  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except (AttributeError, TypeError):
            pass

    def test_format_returns_string(self) -> None:
        r = compute_rv_trend_companion_mass(0.05, 1000.0)
        s = format_rv_trend_result(r)
        assert isinstance(s, str)
        assert r.flag in s
