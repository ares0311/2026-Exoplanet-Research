"""Tests for rolling_bls_periodogram.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
import math

from rolling_bls_periodogram import (
    RollingBLSResult,
    format_rolling_bls_result,
    run_rolling_bls,
)


def _sinusoidal_lc(period=5.0, n=300, amp=0.005):
    time = [i * 0.1 for i in range(n)]
    flux = [1.0 - amp * (0.5 + 0.5 * math.cos(2 * math.pi * t / period)) for t in time]
    return time, flux


class TestRunRollingBLS:
    def test_invalid_empty(self):
        r = run_rolling_bls([], [], 5.0)
        assert r.flag == "INVALID"

    def test_invalid_period_zero(self):
        time = [float(i) for i in range(100)]
        flux = [1.0] * 100
        r = run_rolling_bls(time, flux, 0.0)
        assert r.flag == "INVALID"

    def test_insufficient_windows(self):
        # Short LC: not enough windows
        time = [float(i) * 0.1 for i in range(20)]
        flux = [1.0] * 20
        r = run_rolling_bls(time, flux, 5.0, window_days=14.0, min_windows=3)
        assert r.flag in ("INSUFFICIENT", "INVALID")

    def test_returns_result(self):
        time, flux = _sinusoidal_lc()
        r = run_rolling_bls(time, flux, 5.0, window_days=10.0, step_days=5.0)
        assert isinstance(r, RollingBLSResult)

    def test_n_windows_positive(self):
        time, flux = _sinusoidal_lc(n=500)
        r = run_rolling_bls(time, flux, 5.0, window_days=10.0, step_days=5.0)
        if r.flag == "OK":
            assert r.n_windows > 0

    def test_recovery_fraction_in_range(self):
        time, flux = _sinusoidal_lc(n=500)
        r = run_rolling_bls(time, flux, 5.0, window_days=10.0, step_days=5.0)
        if r.flag == "OK":
            assert 0.0 <= r.recovery_fraction <= 1.0

    def test_recovered_periods_length(self):
        time, flux = _sinusoidal_lc(n=400)
        r = run_rolling_bls(time, flux, 5.0, window_days=10.0, step_days=5.0)
        if r.flag == "OK":
            assert len(r.recovered_periods) == r.n_windows

    def test_result_frozen(self):
        time, flux = _sinusoidal_lc()
        r = run_rolling_bls(time, flux, 5.0)
        try:
            r.n_windows = 99  # type: ignore[misc]
            raise AssertionError()
        except (AttributeError, TypeError):
            pass

    def test_period_days_preserved(self):
        time, flux = _sinusoidal_lc()
        r = run_rolling_bls(time, flux, 7.3)
        assert r.period_days == 7.3


class TestFormatRollingBLS:
    def test_returns_string(self):
        time, flux = _sinusoidal_lc()
        r = run_rolling_bls(time, flux, 5.0)
        assert isinstance(format_rolling_bls_result(r), str)

    def test_contains_flag(self):
        time, flux = _sinusoidal_lc()
        r = run_rolling_bls(time, flux, 5.0)
        assert r.flag in format_rolling_bls_result(r)
