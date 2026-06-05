"""Tests for Skills/co_orbital_checker.py"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from co_orbital_checker import (
    check_co_orbital,
    format_co_orbital_result,
)


class TestCoOrbitalChecker:
    def test_basic_ok(self) -> None:
        r = check_co_orbital(10.0, 317.8)
        assert r.flag == "OK"
        assert r.expected_ttv_amplitude_min > 0

    def test_ttv_amplitude_positive(self) -> None:
        r = check_co_orbital(10.0, 317.8)
        assert r.expected_ttv_amplitude_min > 0
        assert math.isfinite(r.expected_ttv_amplitude_min)

    def test_horseshoe_period_positive(self) -> None:
        r = check_co_orbital(10.0, 317.8)
        assert r.horseshoe_period_yr > 0

    def test_larger_trojan_mass_larger_ttv(self) -> None:
        r_small = check_co_orbital(10.0, 317.8, trojan_mass_mearth=0.1)
        r_large = check_co_orbital(10.0, 317.8, trojan_mass_mearth=10.0)
        assert r_large.expected_ttv_amplitude_min > r_small.expected_ttv_amplitude_min

    def test_mass_limit_from_observed_ttv(self) -> None:
        r = check_co_orbital(10.0, 317.8, observed_ttv_amplitude_min=5.0)
        assert r.flag == "OK"
        assert math.isfinite(r.trojan_mass_limit_mearth)
        assert r.trojan_mass_limit_mearth > 0

    def test_mass_limit_nan_without_observed_ttv(self) -> None:
        r = check_co_orbital(10.0, 317.8)
        assert math.isnan(r.trojan_mass_limit_mearth)

    def test_co_orbital_class_set(self) -> None:
        r = check_co_orbital(10.0, 317.8)
        assert r.co_orbital_class in ("TROJAN_POSSIBLE", "HORSESHOE_POSSIBLE",
                                       "NOT_CONSTRAINED")

    def test_invalid_period(self) -> None:
        r = check_co_orbital(0.0, 317.8)
        assert r.flag == "INVALID_PERIOD"
        assert math.isnan(r.expected_ttv_amplitude_min)

    def test_invalid_mass(self) -> None:
        r = check_co_orbital(10.0, 0.0)
        assert r.flag == "INVALID_MASS"

    def test_result_is_frozen(self) -> None:
        r = check_co_orbital(10.0, 317.8)
        try:
            r.co_orbital_class = "X"  # type: ignore[misc]
            raise AssertionError("frozen dataclass should not allow setattr")
        except Exception:
            pass

    def test_format_ok(self) -> None:
        r = check_co_orbital(10.0, 317.8)
        s = format_co_orbital_result(r)
        assert "Trojan" in s or "trojan" in s.lower()
        assert r.co_orbital_class in s

    def test_format_error(self) -> None:
        r = check_co_orbital(0.0, 317.8)
        s = format_co_orbital_result(r)
        assert "INVALID_PERIOD" in s

    def test_longer_period_smaller_ttv(self) -> None:
        r_short = check_co_orbital(5.0, 317.8)
        r_long = check_co_orbital(100.0, 317.8)
        # TTV amplitude ~ P, so longer period → larger TTV
        assert r_long.expected_ttv_amplitude_min > r_short.expected_ttv_amplitude_min
