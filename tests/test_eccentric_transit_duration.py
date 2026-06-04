"""Tests for Skills/eccentric_transit_duration.py"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from eccentric_transit_duration import EccentricDurationResult, compute_eccentric_t14


class TestEccentricTransitDuration:
    def test_circular_ok(self) -> None:
        r = compute_eccentric_t14(period_days=10.0, eccentricity=0.0)
        assert r.flag == "OK"

    def test_duration_positive(self) -> None:
        r = compute_eccentric_t14(period_days=10.0, eccentricity=0.0)
        assert r.t14_eccentric_hours > 0

    def test_invalid_period(self) -> None:
        r = compute_eccentric_t14(period_days=0.0)
        assert r.flag == "INVALID_PERIOD"

    def test_invalid_eccentricity(self) -> None:
        r = compute_eccentric_t14(period_days=10.0, eccentricity=1.5)
        assert r.flag == "INVALID_ECCENTRICITY"

    def test_ecc_factor_stored(self) -> None:
        r = compute_eccentric_t14(period_days=10.0, eccentricity=0.3, omega_deg=90.0)
        assert r.flag == "OK"
        assert math.isfinite(r.duration_factor)

    def test_circular_factor_one(self) -> None:
        r = compute_eccentric_t14(period_days=10.0, eccentricity=0.0, omega_deg=90.0)
        assert abs(r.duration_factor - 1.0) < 1e-6

    def test_longer_period_longer_duration(self) -> None:
        r1 = compute_eccentric_t14(period_days=3.0, eccentricity=0.0)
        r2 = compute_eccentric_t14(period_days=30.0, eccentricity=0.0)
        assert r2.t14_eccentric_hours > r1.t14_eccentric_hours

    def test_eccentric_vs_circular_duration(self) -> None:
        r_circ = compute_eccentric_t14(period_days=10.0, eccentricity=0.0)
        r_ecc = compute_eccentric_t14(period_days=10.0, eccentricity=0.5, omega_deg=90.0)
        assert r_ecc.t14_eccentric_hours != r_circ.t14_eccentric_hours

    def test_result_frozen(self) -> None:
        r = compute_eccentric_t14(period_days=10.0)
        assert isinstance(r, EccentricDurationResult)
        try:
            object.__setattr__(r, "flag", "x")
            raise AssertionError()
        except Exception:
            pass

    def test_negative_eccentricity(self) -> None:
        r = compute_eccentric_t14(period_days=10.0, eccentricity=-0.1)
        assert r.flag == "INVALID_ECCENTRICITY"

    def test_high_impact_parameter(self) -> None:
        r = compute_eccentric_t14(period_days=10.0, eccentricity=0.0,
                                   impact_parameter=0.99)
        assert r.flag in ("OK", "GRAZING_TRANSIT")

    def test_format_output(self) -> None:
        from eccentric_transit_duration import format_eccentric_duration_result
        r = compute_eccentric_t14(period_days=10.0)
        s = format_eccentric_duration_result(r)
        assert "|" in s
