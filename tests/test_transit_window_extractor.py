"""Tests for Skills.transit_window_extractor."""
from __future__ import annotations

from Skills.transit_window_extractor import (
    TransitWindowResult,
    extract_transit_windows,
    format_window_result,
)


def _make_lc(period=5.0, epoch=2458000.0, depth=0.01, n_cycles=10):
    dt = 2.0 / 1440.0
    n = int(n_cycles * period / dt) + 1
    time = [epoch - 1.0 + i * dt for i in range(n)]
    flux = []
    for t in time:
        ph = (t - epoch) % period
        if ph > period / 2:
            ph -= period
        flux.append(1.0 - depth if abs(ph) < 0.05 else 1.0)
    return time, flux


class TestExtractTransitWindows:
    def test_returns_result(self) -> None:
        t, f = _make_lc()
        r = extract_transit_windows(t, f, 5.0, 2458000.0)
        assert isinstance(r, TransitWindowResult)

    def test_empty_returns_no_transits(self) -> None:
        r = extract_transit_windows([], [], 5.0, 2458000.0)
        assert r.flag == "NO_TRANSITS"

    def test_zero_period_returns_no_transits(self) -> None:
        t, f = _make_lc()
        r = extract_transit_windows(t, f, 0.0, 2458000.0)
        assert r.flag == "NO_TRANSITS"

    def test_finds_windows(self) -> None:
        t, f = _make_lc(n_cycles=10)
        r = extract_transit_windows(t, f, 5.0, 2458000.0, duration_days=0.15)
        assert r.n_windows >= 1

    def test_window_has_points(self) -> None:
        t, f = _make_lc(n_cycles=10)
        r = extract_transit_windows(t, f, 5.0, 2458000.0, duration_days=0.15)
        for w in r.windows:
            assert len(w.time) > 0

    def test_oot_cadences_present(self) -> None:
        t, f = _make_lc(n_cycles=10)
        r = extract_transit_windows(t, f, 5.0, 2458000.0, duration_days=0.15)
        assert len(r.time_oot) > 0

    def test_flux_err_propagated(self) -> None:
        t, f = _make_lc(n_cycles=5)
        e = [1e-4] * len(f)
        r = extract_transit_windows(t, f, 5.0, 2458000.0, flux_err=e, duration_days=0.15)
        for w in r.windows:
            assert w.flux_err is not None

    def test_transit_numbers_are_int(self) -> None:
        t, f = _make_lc(n_cycles=10)
        r = extract_transit_windows(t, f, 5.0, 2458000.0, duration_days=0.15)
        for w in r.windows:
            assert isinstance(w.transit_number, int)

    def test_flag_values_valid(self) -> None:
        t, f = _make_lc(n_cycles=10)
        r = extract_transit_windows(t, f, 5.0, 2458000.0, duration_days=0.15)
        assert r.flag in {"OK", "PARTIAL", "NO_TRANSITS"}


class TestFormatWindowResult:
    def test_returns_string(self) -> None:
        t, f = _make_lc(n_cycles=5)
        r = extract_transit_windows(t, f, 5.0, 2458000.0)
        assert isinstance(format_window_result(r), str)

    def test_contains_n_windows(self) -> None:
        t, f = _make_lc(n_cycles=10)
        r = extract_transit_windows(t, f, 5.0, 2458000.0, duration_days=0.15)
        assert str(r.n_windows) in format_window_result(r)
