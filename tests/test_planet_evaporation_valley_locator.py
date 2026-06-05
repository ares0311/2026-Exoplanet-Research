"""Tests for Skills/planet_evaporation_valley_locator.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from planet_evaporation_valley_locator import (
    format_evaporation_valley_result,
    locate_evaporation_valley,
)


class TestLocateEvaporationValley:
    def test_ok_flag(self) -> None:
        r = locate_evaporation_valley(2.5, 10.0)
        assert r.flag == "OK"

    def test_gap_radius_near_1p9_at_10d(self) -> None:
        r = locate_evaporation_valley(2.5, 10.0)
        assert abs(r.gap_radius_rearth - 1.9) < 0.1

    def test_large_planet_above_gap(self) -> None:
        r = locate_evaporation_valley(4.0, 10.0)
        assert r.valley_position == "ABOVE_GAP"
        assert r.composition_hint == "GAS_ENVELOPE"

    def test_small_planet_below_gap(self) -> None:
        r = locate_evaporation_valley(1.0, 10.0)
        assert r.valley_position == "BELOW_GAP"
        assert r.composition_hint == "ROCKY_CORE"

    def test_gap_planet_in_gap(self) -> None:
        r = locate_evaporation_valley(1.9, 10.0)
        assert r.valley_position == "IN_GAP"

    def test_radius_ratio_positive(self) -> None:
        r = locate_evaporation_valley(2.5, 10.0)
        assert r.radius_ratio > 0.0

    def test_longer_period_lower_gap_radius(self) -> None:
        r_short = locate_evaporation_valley(2.0, 3.0)
        r_long = locate_evaporation_valley(2.0, 30.0)
        assert r_short.gap_radius_rearth > r_long.gap_radius_rearth

    def test_more_massive_star_larger_gap(self) -> None:
        r1 = locate_evaporation_valley(2.0, 10.0, stellar_mass_msun=0.5)
        r2 = locate_evaporation_valley(2.0, 10.0, stellar_mass_msun=1.5)
        assert r2.gap_radius_rearth > r1.gap_radius_rearth

    def test_invalid_radius(self) -> None:
        r = locate_evaporation_valley(0.0, 10.0)
        assert r.flag == "INVALID_RADIUS"

    def test_invalid_period(self) -> None:
        r = locate_evaporation_valley(2.5, 0.0)
        assert r.flag == "INVALID_PERIOD"

    def test_invalid_stellar_mass(self) -> None:
        r = locate_evaporation_valley(2.5, 10.0, stellar_mass_msun=0.0)
        assert r.flag == "INVALID_STELLAR_MASS"

    def test_result_frozen(self) -> None:
        r = locate_evaporation_valley(2.5, 10.0)
        try:
            r.valley_position = "X"  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except (AttributeError, TypeError):
            pass

    def test_format_returns_string(self) -> None:
        r = locate_evaporation_valley(2.5, 10.0)
        s = format_evaporation_valley_result(r)
        assert isinstance(s, str)
        assert r.flag in s
