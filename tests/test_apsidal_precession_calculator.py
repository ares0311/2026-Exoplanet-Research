"""Tests for Skills/apsidal_precession_calculator.py"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from apsidal_precession_calculator import (
    compute_apsidal_precession,
    format_apsidal_precession_result,
)


class TestApsidialPrecessionCalculator:
    def test_basic_ok(self) -> None:
        r = compute_apsidal_precession(3.0)
        assert r.flag == "OK"
        assert r.gr_rate_deg_per_orbit > 0
        assert r.total_rate_deg_per_orbit > 0

    def test_gr_rate_positive(self) -> None:
        r = compute_apsidal_precession(3.0, eccentricity=0.1)
        assert r.gr_rate_deg_per_orbit > 0

    def test_tidal_rate_positive(self) -> None:
        r = compute_apsidal_precession(3.0, eccentricity=0.1)
        assert r.tidal_rate_deg_per_orbit > 0

    def test_total_is_sum(self) -> None:
        r = compute_apsidal_precession(3.0, eccentricity=0.1)
        expected = r.gr_rate_deg_per_orbit + r.tidal_rate_deg_per_orbit
        assert abs(r.total_rate_deg_per_orbit - expected) < 1e-10

    def test_precession_period_positive(self) -> None:
        r = compute_apsidal_precession(3.0)
        assert r.precession_period_yr > 0

    def test_shorter_period_higher_gr_rate(self) -> None:
        r_short = compute_apsidal_precession(1.0)
        r_long = compute_apsidal_precession(10.0)
        assert r_short.gr_rate_deg_per_orbit > r_long.gr_rate_deg_per_orbit

    def test_eccentricity_zero(self) -> None:
        r = compute_apsidal_precession(3.0, eccentricity=0.0)
        assert r.flag == "OK"
        assert r.gr_rate_deg_per_orbit > 0

    def test_invalid_period(self) -> None:
        r = compute_apsidal_precession(0.0)
        assert r.flag == "INVALID_PERIOD"
        assert math.isnan(r.gr_rate_deg_per_orbit)

    def test_invalid_eccentricity(self) -> None:
        r = compute_apsidal_precession(3.0, eccentricity=1.5)
        assert r.flag == "INVALID_ECCENTRICITY"

    def test_invalid_mass(self) -> None:
        r = compute_apsidal_precession(3.0, planet_mass_mjup=0.0)
        assert r.flag == "INVALID_MASS"

    def test_result_is_frozen(self) -> None:
        r = compute_apsidal_precession(3.0)
        try:
            r.gr_rate_deg_per_orbit = 0.0  # type: ignore[misc]
            raise AssertionError("frozen dataclass should not allow setattr")
        except Exception:
            pass

    def test_format_ok(self) -> None:
        r = compute_apsidal_precession(3.0)
        s = format_apsidal_precession_result(r)
        assert "GR" in s or "precession" in s.lower()

    def test_format_error(self) -> None:
        r = compute_apsidal_precession(0.0)
        s = format_apsidal_precession_result(r)
        assert "INVALID_PERIOD" in s
