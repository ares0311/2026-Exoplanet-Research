"""Tests for Skills/planet_oblateness_estimator.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from planet_oblateness_estimator import (
    compute_planet_oblateness,
    format_planet_oblateness_result,
)


class TestComputePlanetOblateness:
    def test_ok_flag(self) -> None:
        r = compute_planet_oblateness(1.0, 1.0, 10.0)
        assert r.flag == "OK"

    def test_oblateness_non_negative(self) -> None:
        r = compute_planet_oblateness(1.0, 1.0, 10.0)
        assert r.oblateness >= 0.0

    def test_oblateness_less_than_half(self) -> None:
        r = compute_planet_oblateness(1.0, 1.0, 10.0)
        assert r.oblateness <= 0.5

    def test_equatorial_radius_ge_mean(self) -> None:
        r = compute_planet_oblateness(1.0, 1.0, 10.0)
        assert r.equatorial_radius_factor >= 1.0

    def test_polar_radius_le_mean(self) -> None:
        r = compute_planet_oblateness(1.0, 1.0, 10.0)
        assert r.polar_radius_factor <= 1.0

    def test_faster_rotation_more_oblate(self) -> None:
        r_slow = compute_planet_oblateness(1.0, 1.0, 500.0)
        r_fast = compute_planet_oblateness(1.0, 1.0, 5.0)
        assert r_fast.oblateness > r_slow.oblateness

    def test_jupiter_like_oblateness(self) -> None:
        r = compute_planet_oblateness(1.0, 1.0, 9.925)
        assert r.oblateness > 0.0

    def test_depth_difference_non_negative(self) -> None:
        r = compute_planet_oblateness(1.0, 1.0, 10.0)
        assert r.transit_depth_difference_ppm >= 0.0

    def test_invalid_mass(self) -> None:
        r = compute_planet_oblateness(0.0, 1.0, 10.0)
        assert r.flag == "INVALID_MASS"

    def test_invalid_radius(self) -> None:
        r = compute_planet_oblateness(1.0, 0.0, 10.0)
        assert r.flag == "INVALID_RADIUS"

    def test_invalid_rotation_period(self) -> None:
        r = compute_planet_oblateness(1.0, 1.0, 0.0)
        assert r.flag == "INVALID_ROTATION_PERIOD"

    def test_result_frozen(self) -> None:
        r = compute_planet_oblateness(1.0, 1.0, 10.0)
        try:
            r.oblateness = 0.0  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except (AttributeError, TypeError):
            pass

    def test_format_returns_string(self) -> None:
        r = compute_planet_oblateness(1.0, 1.0, 10.0)
        s = format_planet_oblateness_result(r)
        assert isinstance(s, str)
        assert r.flag in s
