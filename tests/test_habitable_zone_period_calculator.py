"""Tests for Skills/habitable_zone_period_calculator.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from habitable_zone_period_calculator import compute_hz_periods, format_hz_period_result


class TestComputeHzPeriods:
    def test_ok_flag(self) -> None:
        r = compute_hz_periods(1.0)
        assert r.flag == "OK"

    def test_inner_period_less_than_outer(self) -> None:
        r = compute_hz_periods(1.0)
        assert r.hz_inner_period_days < r.hz_outer_period_days

    def test_optimistic_inner_less_than_conservative_inner(self) -> None:
        r = compute_hz_periods(1.0)
        assert r.hz_optimistic_inner_period_days < r.hz_inner_period_days

    def test_optimistic_outer_greater_than_conservative_outer(self) -> None:
        r = compute_hz_periods(1.0)
        assert r.hz_optimistic_outer_period_days > r.hz_outer_period_days

    def test_solar_hz_inner_distance_approx(self) -> None:
        r = compute_hz_periods(1.0)
        assert 0.9 < r.hz_inner_au < 1.1

    def test_solar_hz_outer_distance_approx(self) -> None:
        r = compute_hz_periods(1.0)
        assert 1.5 < r.hz_outer_au < 2.0

    def test_brighter_star_farther_hz(self) -> None:
        r1 = compute_hz_periods(0.5)
        r2 = compute_hz_periods(2.0)
        assert r2.hz_inner_au > r1.hz_inner_au

    def test_periods_positive(self) -> None:
        r = compute_hz_periods(1.0)
        assert r.hz_inner_period_days > 0.0
        assert r.hz_outer_period_days > 0.0

    def test_invalid_luminosity(self) -> None:
        r = compute_hz_periods(0.0)
        assert r.flag == "INVALID_LUMINOSITY"

    def test_invalid_stellar_mass(self) -> None:
        r = compute_hz_periods(1.0, stellar_mass_msun=0.0)
        assert r.flag == "INVALID_STELLAR_MASS"

    def test_result_frozen(self) -> None:
        r = compute_hz_periods(1.0)
        try:
            r.hz_inner_period_days = 0.0  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except (AttributeError, TypeError):
            pass

    def test_format_returns_string(self) -> None:
        r = compute_hz_periods(1.0)
        s = format_hz_period_result(r)
        assert isinstance(s, str)
        assert r.flag in s
