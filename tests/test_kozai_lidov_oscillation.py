"""Tests for Skills/kozai_lidov_oscillation.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from kozai_lidov_oscillation import compute_kozai_lidov, format_kozai_lidov_result


class TestComputeKozaiLidov:
    def test_ok_flag(self) -> None:
        r = compute_kozai_lidov(10.0, 3650.0, 60.0)
        assert r.flag == "OK"

    def test_critical_inclination_near_39(self) -> None:
        r = compute_kozai_lidov(10.0, 3650.0, 60.0)
        assert abs(r.critical_inclination_deg - 39.23) < 0.1

    def test_above_critical_at_60_deg(self) -> None:
        r = compute_kozai_lidov(10.0, 3650.0, 60.0)
        assert r.is_above_critical_angle

    def test_below_critical_at_20_deg(self) -> None:
        r = compute_kozai_lidov(10.0, 3650.0, 20.0)
        assert not r.is_above_critical_angle

    def test_max_eccentricity_bounded(self) -> None:
        r = compute_kozai_lidov(10.0, 3650.0, 60.0)
        assert 0.0 <= r.max_eccentricity <= 1.0

    def test_cycle_period_positive(self) -> None:
        r = compute_kozai_lidov(10.0, 3650.0, 60.0)
        assert r.kozai_cycle_period_yr > 0.0

    def test_longer_outer_orbit_longer_cycle(self) -> None:
        r1 = compute_kozai_lidov(10.0, 1000.0, 60.0)
        r2 = compute_kozai_lidov(10.0, 5000.0, 60.0)
        assert r2.kozai_cycle_period_yr > r1.kozai_cycle_period_yr

    def test_high_inclination_max_eccentricity(self) -> None:
        r = compute_kozai_lidov(10.0, 3650.0, 90.0)
        assert r.max_eccentricity > 0.5

    def test_invalid_inner_period(self) -> None:
        r = compute_kozai_lidov(0.0, 3650.0, 60.0)
        assert r.flag == "INVALID_INNER_PERIOD"

    def test_outer_not_longer(self) -> None:
        r = compute_kozai_lidov(100.0, 10.0, 60.0)
        assert r.flag == "OUTER_NOT_LONGER"

    def test_invalid_inclination(self) -> None:
        r = compute_kozai_lidov(10.0, 3650.0, 200.0)
        assert r.flag == "INVALID_INCLINATION"

    def test_result_frozen(self) -> None:
        r = compute_kozai_lidov(10.0, 3650.0, 60.0)
        try:
            r.max_eccentricity = 0.0  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except (AttributeError, TypeError):
            pass

    def test_format_returns_string(self) -> None:
        r = compute_kozai_lidov(10.0, 3650.0, 60.0)
        s = format_kozai_lidov_result(r)
        assert isinstance(s, str)
        assert r.flag in s
