"""Tests for Skills/kepler_third_law_checker.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from kepler_third_law_checker import check_kepler_third_law, format_kepler_check_result

# Earth: period=365.25 days, a=1.0 AU, M=1.0 M_sun
_EARTH_P = 365.25
_EARTH_A = 1.0
_EARTH_M = 1.0


class TestCheckKeplerThirdLaw:
    def test_earth_consistent(self) -> None:
        r = check_kepler_third_law(_EARTH_P, _EARTH_A, _EARTH_M)
        assert r.consistent is True
        assert r.flag == "CONSISTENT"

    def test_predicted_period_close(self) -> None:
        r = check_kepler_third_law(_EARTH_P, _EARTH_A, _EARTH_M)
        assert abs(r.predicted_period_days - _EARTH_P) < 1.0

    def test_predicted_sma_close(self) -> None:
        r = check_kepler_third_law(_EARTH_P, _EARTH_A, _EARTH_M)
        assert abs(r.predicted_sma_au - _EARTH_A) < 0.01

    def test_inconsistent_wrong_mass(self) -> None:
        # period and SMA consistent for 1 M_sun, but claim 10 M_sun
        r = check_kepler_third_law(_EARTH_P, _EARTH_A, 10.0)
        assert r.consistent is False

    def test_invalid_period(self) -> None:
        r = check_kepler_third_law(0.0, 1.0, 1.0)
        assert r.flag == "INVALID_PERIOD"

    def test_invalid_sma(self) -> None:
        r = check_kepler_third_law(365.25, 0.0, 1.0)
        assert r.flag == "INVALID_SMA"

    def test_invalid_stellar_mass(self) -> None:
        r = check_kepler_third_law(365.25, 1.0, 0.0)
        assert r.flag == "INVALID_STELLAR_MASS"

    def test_hot_jupiter_consistent(self) -> None:
        # P=3d, a~0.04 AU for M=1 Msun
        r = check_kepler_third_law(3.0, 0.0407, 1.0)
        assert r.consistent is True

    def test_period_residual_float(self) -> None:
        r = check_kepler_third_law(_EARTH_P, _EARTH_A, _EARTH_M)
        assert isinstance(r.period_residual_days, float)

    def test_sma_residual_float(self) -> None:
        r = check_kepler_third_law(_EARTH_P, _EARTH_A, _EARTH_M)
        assert isinstance(r.sma_residual_au, float)

    def test_format_returns_string(self) -> None:
        r = check_kepler_third_law(_EARTH_P, _EARTH_A, _EARTH_M)
        s = format_kepler_check_result(r)
        assert isinstance(s, str)
        assert "Kepler" in s

    def test_negative_period_invalid(self) -> None:
        r = check_kepler_third_law(-10.0, 1.0, 1.0)
        assert r.flag == "INVALID_PERIOD"
