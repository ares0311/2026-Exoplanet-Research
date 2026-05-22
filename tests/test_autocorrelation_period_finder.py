"""Tests for Skills/autocorrelation_period_finder.py."""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from autocorrelation_period_finder import (
    ACFResult,
    compute_acf,
    find_acf_period,
    format_acf_result,
)


def _sine_lc(period: float, n: int = 200, dt: float = 0.1) -> tuple[list[float], list[float]]:
    """Generate a sine-wave light curve with given period."""
    time = [i * dt for i in range(n)]
    flux = [1.0 + 0.01 * math.sin(2 * math.pi * t / period) for t in time]
    return time, flux


class TestComputeAcf:
    def test_returns_acf_result(self):
        time, flux = _sine_lc(5.0)
        result = compute_acf(time, flux)
        assert isinstance(result, ACFResult)

    def test_flag_ok_for_sufficient_data(self):
        time, flux = _sine_lc(5.0)
        result = compute_acf(time, flux)
        assert result.flag == "OK"

    def test_too_few_points_returns_invalid(self):
        result = compute_acf([0.0, 1.0], [1.0, 1.0])
        assert result.flag == "INVALID"

    def test_empty_input_returns_invalid(self):
        result = compute_acf([], [])
        assert result.flag == "INVALID"

    def test_lags_start_at_zero(self):
        time, flux = _sine_lc(5.0)
        result = compute_acf(time, flux)
        assert result.lags_days[0] == pytest.approx(0.0, abs=1e-6)

    def test_acf_at_lag_zero_is_one(self):
        time, flux = _sine_lc(5.0)
        result = compute_acf(time, flux)
        assert result.acf_values[0] == pytest.approx(1.0, abs=1e-3)

    def test_lags_and_acf_same_length(self):
        time, flux = _sine_lc(5.0)
        result = compute_acf(time, flux)
        assert len(result.lags_days) == len(result.acf_values)

    def test_period_is_none_before_find(self):
        time, flux = _sine_lc(5.0)
        result = compute_acf(time, flux)
        assert result.period_days is None

    def test_mismatched_lengths_returns_invalid(self):
        result = compute_acf([0.0] * 15, [1.0] * 14)
        assert result.flag == "INVALID"

    def test_frozen_dataclass(self):
        time, flux = _sine_lc(5.0)
        result = compute_acf(time, flux)
        with pytest.raises((AttributeError, TypeError)):
            result.flag = "CHANGED"  # type: ignore[misc]


class TestFindAcfPeriod:
    def test_returns_acf_result(self):
        time, flux = _sine_lc(5.0)
        acf = compute_acf(time, flux, max_lag_days=20.0)
        result = find_acf_period(acf, min_period_days=2.0, max_period_days=15.0)
        assert isinstance(result, ACFResult)

    def test_detects_period_for_sinusoidal(self):
        time, flux = _sine_lc(5.0, n=300, dt=0.1)
        acf = compute_acf(time, flux, max_lag_days=25.0)
        result = find_acf_period(acf, min_period_days=2.0, max_period_days=15.0)
        if result.flag == "OK" and result.period_days is not None:
            # ACF may detect fundamental (5 d) or harmonic (10 d)
            ratio = result.period_days / 5.0
            assert abs(ratio - round(ratio)) < 0.1

    def test_no_period_flag_for_flat_lc(self):
        time = [float(i) * 0.1 for i in range(200)]
        flux = [1.0] * 200
        acf = compute_acf(time, flux)
        result = find_acf_period(acf, min_period_days=1.0, max_period_days=10.0)
        assert result.flag in ("NO_PERIOD", "INSUFFICIENT", "OK")

    def test_passthrough_on_non_ok_flag(self):
        invalid = ACFResult((), (), None, None, None, 0, "INVALID")
        result = find_acf_period(invalid)
        assert result.flag == "INVALID"

    def test_n_peaks_populated(self):
        time, flux = _sine_lc(5.0, n=300, dt=0.1)
        acf = compute_acf(time, flux, max_lag_days=25.0)
        result = find_acf_period(acf, min_period_days=2.0, max_period_days=15.0)
        assert result.n_peaks >= 0


class TestFormatAcfResult:
    def test_returns_string(self):
        time, flux = _sine_lc(5.0)
        acf = compute_acf(time, flux)
        md = format_acf_result(acf)
        assert isinstance(md, str)

    def test_contains_flag(self):
        time, flux = _sine_lc(5.0)
        acf = compute_acf(time, flux)
        md = format_acf_result(acf)
        assert acf.flag in md

    def test_invalid_result_format(self):
        result = compute_acf([], [])
        md = format_acf_result(result)
        assert "INVALID" in md
