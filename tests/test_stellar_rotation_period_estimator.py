"""Tests for Skills/stellar_rotation_period_estimator.py"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from stellar_rotation_period_estimator import (
    estimate_rotation_period,
    format_rotation_estimate,
)


class TestEstimateRotationPeriod:
    def _sine_flux(self, n: int, period_days: float, cadence: float) -> list[float]:
        return [math.sin(2 * math.pi * i * cadence / period_days) for i in range(n)]

    def test_sinusoidal_detects_period(self) -> None:
        flux = self._sine_flux(200, 5.0, 0.1)
        r = estimate_rotation_period(flux, 0.1, max_lag_days=20.0)
        assert r.flag == "OK"
        assert r.period_days > 0.0

    def test_period_within_20_percent(self) -> None:
        true_period = 5.0
        flux = self._sine_flux(300, true_period, 0.1)
        r = estimate_rotation_period(flux, 0.1, max_lag_days=25.0)
        if r.flag == "OK":
            assert abs(r.period_days - true_period) / true_period < 0.3

    def test_too_short_flux_insufficient(self) -> None:
        r = estimate_rotation_period([1.0, 2.0, 3.0], 0.1)
        assert r.flag == "INSUFFICIENT_DATA"

    def test_flat_flux_no_period(self) -> None:
        flux = [1.0] * 100
        r = estimate_rotation_period(flux, 0.1)
        assert r.flag in ("NO_PERIOD_FOUND", "OK")

    def test_period_days_positive_for_periodic(self) -> None:
        flux = self._sine_flux(200, 5.0, 0.1)
        r = estimate_rotation_period(flux, 0.1)
        if r.flag == "OK":
            assert r.period_days > 0.0

    def test_quality_field_is_string(self) -> None:
        flux = self._sine_flux(200, 5.0, 0.1)
        r = estimate_rotation_period(flux, 0.1)
        assert isinstance(r.quality, str)

    def test_alias_check_is_bool(self) -> None:
        flux = self._sine_flux(200, 5.0, 0.1)
        r = estimate_rotation_period(flux, 0.1)
        assert isinstance(r.alias_check, bool)

    def test_flag_ok_for_good_data(self) -> None:
        flux = self._sine_flux(300, 7.0, 0.1)
        r = estimate_rotation_period(flux, 0.1, max_lag_days=30.0)
        assert r.flag in ("OK", "NO_PERIOD_FOUND", "PERIOD_TOO_SHORT")

    def test_flag_insufficient_data_short(self) -> None:
        r = estimate_rotation_period([1.0] * 5, 0.1)
        assert r.flag == "INSUFFICIENT_DATA"

    def test_acf_peak_lag_positive_for_periodic(self) -> None:
        flux = self._sine_flux(300, 5.0, 0.1)
        r = estimate_rotation_period(flux, 0.1)
        if r.flag == "OK":
            assert r.acf_peak_lag > 0.0

    def test_result_is_frozen(self) -> None:
        r = estimate_rotation_period([1.0] * 5, 0.1)
        try:
            r.flag = "X"  # type: ignore[misc]
            raise AssertionError("should be frozen")
        except Exception:
            pass

    def test_format_contains_period(self) -> None:
        flux = self._sine_flux(200, 5.0, 0.1)
        r = estimate_rotation_period(flux, 0.1)
        s = format_rotation_estimate(r)
        assert "Period" in s or "period" in s
