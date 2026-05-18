"""Tests for Skills.period_doubling_checker."""
from __future__ import annotations

from Skills.period_doubling_checker import (
    PeriodDoublingResult,
    check_period_doubling,
    format_period_doubling_result,
)


def _make_lc_with_half_period(period=10.0, epoch=2458000.0, depth=0.005, n_points=2000):
    """LC with transit at P/2 in addition to primary."""
    dt = 2.0 / 1440.0
    time = [epoch + i * dt for i in range(n_points)]
    flux = []
    half = period / 2.0
    for t in time:
        ph = (t - epoch) % period
        if ph < 0.05 or abs(ph - half) < 0.05:
            flux.append(1.0 - depth)
        else:
            flux.append(1.0)
    return time, flux


def _make_clean_lc(period=10.0, epoch=2458000.0, depth=0.005, n_points=2000):
    """LC with transit only at P."""
    dt = 2.0 / 1440.0
    time = [epoch + i * dt for i in range(n_points)]
    flux = [1.0 - depth if (t - epoch) % period < 0.05 else 1.0 for t in time]
    return time, flux


class TestCheckPeriodDoubling:
    def test_returns_result(self) -> None:
        t, f = _make_clean_lc()
        r = check_period_doubling(t, f, 10.0, 2458000.0)
        assert isinstance(r, PeriodDoublingResult)

    def test_empty_returns_insufficient(self) -> None:
        r = check_period_doubling([], [], 10.0, 2458000.0)
        assert r.flag == "INSUFFICIENT"

    def test_zero_period_returns_insufficient(self) -> None:
        t, f = _make_clean_lc()
        r = check_period_doubling(t, f, 0.0, 2458000.0)
        assert r.flag == "INSUFFICIENT"

    def test_half_period_stored(self) -> None:
        r = check_period_doubling([], [], 10.0, 2458000.0)
        assert abs(r.half_period_days - 5.0) < 1e-9

    def test_no_doubling_flagged_ok(self) -> None:
        t, f = _make_clean_lc(depth=0.01)
        r = check_period_doubling(t, f, 10.0, 2458000.0, duration_days=0.08, snr_threshold=3.0)
        assert r.flag in {"OK", "INSUFFICIENT"}

    def test_flag_values_valid(self) -> None:
        t, f = _make_clean_lc()
        r = check_period_doubling(t, f, 10.0, 2458000.0)
        assert r.flag in {"OK", "POSSIBLE_DOUBLING", "INSUFFICIENT"}


class TestFormatPeriodDoubling:
    def test_returns_string(self) -> None:
        t, f = _make_clean_lc()
        r = check_period_doubling(t, f, 10.0, 2458000.0)
        assert isinstance(format_period_doubling_result(r), str)

    def test_insufficient_handled(self) -> None:
        r = check_period_doubling([], [], 10.0, 2458000.0)
        out = format_period_doubling_result(r)
        assert "INSUFFICIENT" in out or "insufficient" in out.lower()


