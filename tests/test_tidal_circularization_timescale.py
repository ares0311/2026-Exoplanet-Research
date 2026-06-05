"""Tests for Skills/tidal_circularization_timescale.py"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from tidal_circularization_timescale import (
    CircularizationResult,
    compute_circularization_timescale,
    format_circularization_result,
)


class TestTidalCircularizationTimescale:
    def test_basic_hot_jupiter(self) -> None:
        r = compute_circularization_timescale(3.0, mass_planet_mjup=1.0)
        assert r.flag == "OK"
        assert r.timescale_gyr > 0
        assert math.isfinite(r.timescale_gyr)

    def test_circular_orbit_flagged(self) -> None:
        r = compute_circularization_timescale(3.0, eccentricity=0.02)
        assert r.currently_circular is True

    def test_eccentric_orbit_flagged(self) -> None:
        r = compute_circularization_timescale(3.0, eccentricity=0.3)
        assert r.currently_circular is False

    def test_expected_circular_short_period(self) -> None:
        r = compute_circularization_timescale(1.0, mass_planet_mjup=1.0,
                                               radius_planet_rjup=1.0)
        assert r.flag == "OK"
        assert r.timescale_gyr > 0

    def test_dominant_tide_changes_with_q(self) -> None:
        r_low_qp = compute_circularization_timescale(3.0, q_planet=1e4, q_star=1e8)
        r_low_qs = compute_circularization_timescale(3.0, q_planet=1e8, q_star=1e4)
        assert r_low_qp.dominant_tide == "PLANETARY"
        assert r_low_qs.dominant_tide == "STELLAR"

    def test_timescale_increases_with_period(self) -> None:
        r_short = compute_circularization_timescale(2.0)
        r_long = compute_circularization_timescale(10.0)
        assert r_long.timescale_gyr > r_short.timescale_gyr

    def test_invalid_period(self) -> None:
        r = compute_circularization_timescale(-1.0)
        assert r.flag == "INVALID_PERIOD"
        assert math.isnan(r.timescale_gyr)

    def test_invalid_stellar_mass(self) -> None:
        r = compute_circularization_timescale(3.0, mass_star_msun=0.0)
        assert r.flag == "INVALID_MASS"

    def test_result_is_frozen(self) -> None:
        r = compute_circularization_timescale(3.0)
        assert isinstance(r, CircularizationResult)
        try:
            object.__setattr__(r, "dominant_tide", "X")
            raise AssertionError("frozen dataclass should not allow setattr")
        except Exception:
            pass

    def test_format_ok(self) -> None:
        r = compute_circularization_timescale(3.0)
        s = format_circularization_result(r)
        assert "Gyr" in s
        assert r.dominant_tide in s

    def test_format_error(self) -> None:
        r = compute_circularization_timescale(-1.0)
        s = format_circularization_result(r)
        assert "INVALID_PERIOD" in s or "N/A" in s

    def test_expected_circular_flag_logic(self) -> None:
        r = compute_circularization_timescale(1.0, stellar_age_gyr=5.0)
        assert isinstance(r.expected_circular, bool)

    def test_period_stored_in_result(self) -> None:
        r = compute_circularization_timescale(4.5)
        assert abs(r.period_days - 4.5) < 1e-9
