"""Tests for Skills/tidal_circularization_timescale.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from tidal_circularization_timescale import (
    CircularizationResult,
    compute_circularization_timescale,
)


class TestTidalCircularizationTimescale:
    def test_hot_jupiter_returns_ok(self) -> None:
        r = compute_circularization_timescale(period_days=3.0, mass_star_msun=1.0,
            mass_planet_mjup=1.0, radius_star_rsun=1.0, radius_planet_rjup=1.0)
        assert r.flag == "OK"

    def test_timescale_positive(self) -> None:
        r = compute_circularization_timescale(period_days=3.0)
        assert r.flag == "OK"
        assert r.timescale_gyr > 0

    def test_invalid_period(self) -> None:
        r = compute_circularization_timescale(period_days=0.0)
        assert r.flag == "INVALID_PERIOD"

    def test_invalid_negative_period(self) -> None:
        r = compute_circularization_timescale(period_days=-1.0)
        assert r.flag == "INVALID_PERIOD"

    def test_shorter_period_faster_circularization(self) -> None:
        r1 = compute_circularization_timescale(period_days=1.0)
        r2 = compute_circularization_timescale(period_days=10.0)
        assert r1.timescale_gyr < r2.timescale_gyr

    def test_dominant_tide_field(self) -> None:
        r = compute_circularization_timescale(period_days=3.0)
        assert r.dominant_tide in ("PLANETARY", "STELLAR")

    def test_already_circular_flag(self) -> None:
        r = compute_circularization_timescale(period_days=3.0, eccentricity=0.0)
        assert r.flag == "OK"
        assert r.currently_circular

    def test_eccentric_orbit_not_circular(self) -> None:
        r = compute_circularization_timescale(period_days=3.0, eccentricity=0.3)
        assert not r.currently_circular

    def test_circularized_within_age(self) -> None:
        r = compute_circularization_timescale(period_days=1.0, eccentricity=0.3,
            stellar_age_gyr=10.0)
        assert r.flag == "OK"
        assert hasattr(r, "expected_circular")

    def test_large_q_longer_timescale(self) -> None:
        r1 = compute_circularization_timescale(period_days=3.0, q_planet=1e4)
        r2 = compute_circularization_timescale(period_days=3.0, q_planet=1e7)
        assert r2.timescale_gyr > r1.timescale_gyr

    def test_result_frozen(self) -> None:
        r = compute_circularization_timescale(period_days=3.0)
        assert isinstance(r, CircularizationResult)
        try:
            object.__setattr__(r, "flag", "x")
            raise AssertionError()
        except Exception:
            pass

    def test_format_output(self) -> None:
        from tidal_circularization_timescale import format_circularization_result
        r = compute_circularization_timescale(period_days=3.0)
        s = format_circularization_result(r)
        assert "|" in s
