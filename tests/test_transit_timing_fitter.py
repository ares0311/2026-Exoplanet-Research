"""Tests for Skills.transit_timing_fitter."""
from __future__ import annotations

import math
import pytest
from Skills.transit_timing_fitter import (
    TransitTiming,
    TransitTimingResult,
    fit_transit_times,
    format_timing_result,
)


def _lc_with_transits(period=5.0, epoch=2458000.0, depth=0.01, n_points=500):
    dt = 2.0 / 1440.0  # 2-min cadence
    t0 = epoch - 5.0
    time = [t0 + i * dt for i in range(n_points)]
    flux = []
    for t in time:
        phase = (t - epoch) % period
        if phase > period / 2:
            phase -= period
        if abs(phase) < 0.05:
            flux.append(1.0 - depth)
        else:
            flux.append(1.0)
    return time, flux


class TestFitTransitTimes:
    def test_returns_result(self) -> None:
        t, f = _lc_with_transits()
        r = fit_transit_times(t, f, 5.0, 2458000.0)
        assert isinstance(r, TransitTimingResult)

    def test_empty_returns_empty(self) -> None:
        r = fit_transit_times([], [], 5.0, 2458000.0)
        assert r.n_transits == 0

    def test_zero_period_returns_empty(self) -> None:
        t, f = _lc_with_transits()
        r = fit_transit_times(t, f, 0.0, 2458000.0)
        assert r.n_transits == 0

    def test_period_stored(self) -> None:
        t, f = _lc_with_transits()
        r = fit_transit_times(t, f, 5.0, 2458000.0)
        assert r.period_days == pytest.approx(5.0)

    def test_epoch_stored(self) -> None:
        t, f = _lc_with_transits()
        r = fit_transit_times(t, f, 5.0, 2458000.0)
        assert r.epoch_bjd == pytest.approx(2458000.0)

    def test_finds_transits(self) -> None:
        t, f = _lc_with_transits(period=5.0, n_points=500)
        r = fit_transit_times(t, f, 5.0, 2458000.0, duration_days=0.1)
        assert r.n_transits >= 1

    def test_rms_oc_nonnegative(self) -> None:
        t, f = _lc_with_transits()
        r = fit_transit_times(t, f, 5.0, 2458000.0)
        assert r.rms_oc_minutes >= 0.0

    def test_timing_transit_number_int(self) -> None:
        t, f = _lc_with_transits()
        r = fit_transit_times(t, f, 5.0, 2458000.0, duration_days=0.1)
        for timing in r.timings:
            assert isinstance(timing.transit_number, int)

    def test_timing_snr_positive(self) -> None:
        t, f = _lc_with_transits()
        r = fit_transit_times(t, f, 5.0, 2458000.0, duration_days=0.1)
        for timing in r.timings:
            assert timing.snr > 0.0

    def test_min_snr_filters(self) -> None:
        t, f = _lc_with_transits(depth=0.0001)  # very shallow
        r_strict = fit_transit_times(t, f, 5.0, 2458000.0, min_snr=100.0)
        r_loose = fit_transit_times(t, f, 5.0, 2458000.0, min_snr=0.0)
        assert r_strict.n_transits <= r_loose.n_transits


class TestFormatTimingResult:
    def test_returns_string(self) -> None:
        t, f = _lc_with_transits()
        r = fit_transit_times(t, f, 5.0, 2458000.0)
        assert isinstance(format_timing_result(r), str)

    def test_contains_period(self) -> None:
        t, f = _lc_with_transits()
        r = fit_transit_times(t, f, 5.0, 2458000.0)
        assert "5.0" in format_timing_result(r)
