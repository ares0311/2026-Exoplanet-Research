"""Tests for Skills/tidal_heating_estimator.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from tidal_heating_estimator import compute_tidal_heating, format_tidal_heating_result


class TestComputeTidalHeating:
    def test_ok_flag(self) -> None:
        r = compute_tidal_heating(3.5, eccentricity=0.1)
        assert r.flag == "OK"

    def test_zero_eccentricity_zero_heating(self) -> None:
        r = compute_tidal_heating(3.5, eccentricity=0.0)
        assert r.heating_rate_w == 0.0

    def test_heating_scales_with_eccentricity_squared(self) -> None:
        r1 = compute_tidal_heating(3.5, eccentricity=0.1)
        r2 = compute_tidal_heating(3.5, eccentricity=0.2)
        assert abs(r2.heating_rate_w / r1.heating_rate_w - 4.0) < 0.01

    def test_heating_decreases_with_period(self) -> None:
        r_short = compute_tidal_heating(1.0, eccentricity=0.1)
        r_long = compute_tidal_heating(10.0, eccentricity=0.1)
        assert r_short.heating_rate_w > r_long.heating_rate_w

    def test_heating_increases_with_q_decrease(self) -> None:
        r_lowq = compute_tidal_heating(3.5, eccentricity=0.1, q_planet=1e4)
        r_highq = compute_tidal_heating(3.5, eccentricity=0.1, q_planet=1e6)
        assert r_lowq.heating_rate_w > r_highq.heating_rate_w

    def test_heat_flux_positive(self) -> None:
        r = compute_tidal_heating(3.5, eccentricity=0.1)
        assert r.heating_flux_wm2 >= 0

    def test_dominant_source_eccentricity(self) -> None:
        r = compute_tidal_heating(3.5, eccentricity=0.1)
        assert r.dominant_source == "ECCENTRICITY"

    def test_invalid_period(self) -> None:
        r = compute_tidal_heating(0.0, eccentricity=0.1)
        assert r.flag == "INVALID_PERIOD"

    def test_invalid_eccentricity_negative(self) -> None:
        r = compute_tidal_heating(3.5, eccentricity=-0.1)
        assert r.flag == "INVALID_ECCENTRICITY"

    def test_invalid_eccentricity_one(self) -> None:
        r = compute_tidal_heating(3.5, eccentricity=1.0)
        assert r.flag == "INVALID_ECCENTRICITY"

    def test_result_frozen(self) -> None:
        r = compute_tidal_heating(3.5, eccentricity=0.1)
        try:
            r.heating_rate_w = 0.0  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except (AttributeError, TypeError):
            pass

    def test_format_returns_string(self) -> None:
        r = compute_tidal_heating(3.5, eccentricity=0.1)
        s = format_tidal_heating_result(r)
        assert isinstance(s, str)
        assert r.flag in s
