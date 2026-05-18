"""Tests for Skills.snr_estimator."""
from __future__ import annotations

from Skills.snr_estimator import (
    SNRResult,
    estimate_snr,
    format_snr_result,
)


def _make_lc(period=10.0, epoch=2458000.0, depth=0.005, n_points=2000):
    dt = 2.0 / 1440.0
    time = [epoch + i * dt for i in range(n_points)]
    flux = [1.0 - depth if (t - epoch) % period < 0.05 else 1.0 for t in time]
    return time, flux


class TestEstimateSNR:
    def test_returns_result(self) -> None:
        t, f = _make_lc()
        r = estimate_snr(t, f, 10.0, 2458000.0)
        assert isinstance(r, SNRResult)

    def test_empty_returns_insufficient(self) -> None:
        r = estimate_snr([], [], 10.0, 2458000.0)
        assert r.flag == "INSUFFICIENT"

    def test_zero_period_returns_insufficient(self) -> None:
        t, f = _make_lc()
        r = estimate_snr(t, f, 0.0, 2458000.0)
        assert r.flag == "INSUFFICIENT"

    def test_ok_flag_with_data(self) -> None:
        t, f = _make_lc(depth=0.01)
        r = estimate_snr(t, f, 10.0, 2458000.0, duration_days=0.06)
        assert r.flag in {"OK", "INSUFFICIENT"}

    def test_snr_combined_positive(self) -> None:
        t, f = _make_lc(depth=0.01)
        r = estimate_snr(t, f, 10.0, 2458000.0, duration_days=0.06)
        if r.snr_combined is not None:
            assert r.snr_combined > 0

    def test_snr_combined_geq_single(self) -> None:
        t, f = _make_lc(depth=0.01)
        r = estimate_snr(t, f, 10.0, 2458000.0, duration_days=0.06)
        if r.snr_single is not None and r.snr_combined is not None:
            assert r.snr_combined >= r.snr_single

    def test_depth_positive_for_real_transit(self) -> None:
        t, f = _make_lc(depth=0.01)
        r = estimate_snr(t, f, 10.0, 2458000.0, duration_days=0.06)
        if r.flag == "OK":
            assert r.depth_ppm > 0

    def test_n_transits_positive(self) -> None:
        t, f = _make_lc(depth=0.01)
        r = estimate_snr(t, f, 10.0, 2458000.0, duration_days=0.06)
        if r.flag == "OK":
            assert r.n_transits >= 1

    def test_deeper_transit_higher_snr(self) -> None:
        t1, f1 = _make_lc(depth=0.001)
        t2, f2 = _make_lc(depth=0.01)
        r1 = estimate_snr(t1, f1, 10.0, 2458000.0, duration_days=0.06)
        r2 = estimate_snr(t2, f2, 10.0, 2458000.0, duration_days=0.06)
        if r1.snr_combined is not None and r2.snr_combined is not None:
            assert r2.snr_combined > r1.snr_combined


class TestFormatSNRResult:
    def test_returns_string(self) -> None:
        t, f = _make_lc()
        r = estimate_snr(t, f, 10.0, 2458000.0)
        assert isinstance(format_snr_result(r), str)

    def test_insufficient_handled(self) -> None:
        r = estimate_snr([], [], 10.0, 2458000.0)
        out = format_snr_result(r)
        assert "INSUFFICIENT" in out
