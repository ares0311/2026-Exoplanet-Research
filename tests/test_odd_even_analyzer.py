"""Tests for Skills.odd_even_analyzer."""
from __future__ import annotations

from Skills.odd_even_analyzer import (
    OddEvenResult,
    analyze_odd_even,
    format_odd_even_result,
)


def _make_lc(period=5.0, epoch=2458000.0, depth=0.01, n_cycles=20):
    dt = 2.0 / 1440.0
    t0 = epoch
    n_pts = int(n_cycles * period / dt) + 1
    time = [t0 + i * dt for i in range(n_pts)]
    flux = []
    for t in time:
        phase = (t - epoch) % period
        if phase > period / 2:
            phase -= period
        flux.append(1.0 - depth if abs(phase) < 0.05 else 1.0)
    return time, flux


class TestAnalyzeOddEven:
    def test_returns_result(self) -> None:
        t, f = _make_lc()
        r = analyze_odd_even(t, f, 5.0, 2458000.0)
        assert isinstance(r, OddEvenResult)

    def test_empty_returns_insufficient(self) -> None:
        r = analyze_odd_even([], [], 5.0, 2458000.0)
        assert r.flag == "INSUFFICIENT"

    def test_zero_period_returns_insufficient(self) -> None:
        t, f = _make_lc()
        r = analyze_odd_even(t, f, 0.0, 2458000.0)
        assert r.flag == "INSUFFICIENT"

    def test_finds_odd_and_even_transits(self) -> None:
        t, f = _make_lc(n_cycles=20)
        r = analyze_odd_even(t, f, 5.0, 2458000.0, duration_days=0.15)
        assert r.n_odd >= 2
        assert r.n_even >= 2

    def test_symmetric_lc_passes(self) -> None:
        t, f = _make_lc(n_cycles=20)
        r = analyze_odd_even(t, f, 5.0, 2458000.0, duration_days=0.15)
        if r.flag != "INSUFFICIENT":
            assert r.flag in {"PASS", "WARN"}

    def test_depths_positive(self) -> None:
        t, f = _make_lc(n_cycles=20)
        r = analyze_odd_even(t, f, 5.0, 2458000.0, duration_days=0.15)
        if r.depth_odd_ppm is not None:
            assert r.depth_odd_ppm >= 0
        if r.depth_even_ppm is not None:
            assert r.depth_even_ppm >= 0

    def test_sigma_asymmetry_nonnegative(self) -> None:
        t, f = _make_lc(n_cycles=20)
        r = analyze_odd_even(t, f, 5.0, 2458000.0, duration_days=0.15)
        if r.sigma_asymmetry is not None:
            assert r.sigma_asymmetry >= 0

    def test_asymmetric_lc_fails(self) -> None:
        t, f = _make_lc(n_cycles=20, depth=0.01)
        # Double depth on odd transits to create asymmetry
        epoch = 2458000.0
        period = 5.0
        modified = list(f)
        for i, tv in enumerate(t):
            n = round((tv - epoch) / period)
            t_mid = epoch + n * period
            if abs(tv - t_mid) < 0.05 and n % 2 == 1:
                modified[i] = 1.0 - 0.05  # much deeper odd transits
        r = analyze_odd_even(t, modified, period, epoch, duration_days=0.15)
        if r.flag != "INSUFFICIENT":
            assert r.flag in {"WARN", "FAIL"}

    def test_flag_values_valid(self) -> None:
        t, f = _make_lc(n_cycles=20)
        r = analyze_odd_even(t, f, 5.0, 2458000.0, duration_days=0.15)
        assert r.flag in {"PASS", "WARN", "FAIL", "INSUFFICIENT"}


class TestFormatOddEven:
    def test_returns_string(self) -> None:
        t, f = _make_lc(n_cycles=20)
        r = analyze_odd_even(t, f, 5.0, 2458000.0, duration_days=0.15)
        assert isinstance(format_odd_even_result(r), str)

    def test_insufficient_handled(self) -> None:
        r = analyze_odd_even([], [], 5.0, 2458000.0)
        out = format_odd_even_result(r)
        assert "Insufficient" in out or "insufficient" in out.lower()
