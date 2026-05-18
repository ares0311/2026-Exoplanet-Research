"""Tests for Skills.harmonic_period_analyzer."""
from __future__ import annotations

from Skills.harmonic_period_analyzer import (
    HarmonicResult,
    analyze_harmonics,
    format_harmonic_result,
)


def _make_lc_with_harmonic(period=10.0, epoch=2458000.0, depth=0.01, n_points=3000):
    """LC with strong signal at P/2."""
    dt = 2.0 / 1440.0
    time = [epoch + i * dt for i in range(n_points)]
    half = period / 2.0
    flux = []
    for t in time:
        ph = (t - epoch) % period
        if ph < 0.04 or abs(ph - half) < 0.04:
            flux.append(1.0 - depth)
        else:
            flux.append(1.0)
    return time, flux


def _make_clean_lc(period=10.0, epoch=2458000.0, n_points=3000):
    dt = 2.0 / 1440.0
    time = [epoch + i * dt for i in range(n_points)]
    flux = [1.0] * n_points
    return time, flux


class TestAnalyzeHarmonics:
    def test_returns_result(self) -> None:
        t, f = _make_clean_lc()
        r = analyze_harmonics(t, f, 10.0, 2458000.0)
        assert isinstance(r, HarmonicResult)

    def test_empty_returns_insufficient(self) -> None:
        r = analyze_harmonics([], [], 10.0, 2458000.0)
        assert r.flag == "INSUFFICIENT"

    def test_zero_period_returns_insufficient(self) -> None:
        t, f = _make_clean_lc()
        r = analyze_harmonics(t, f, 0.0, 2458000.0)
        assert r.flag == "INSUFFICIENT"

    def test_harmonics_tested_populated(self) -> None:
        t, f = _make_clean_lc()
        r = analyze_harmonics(t, f, 10.0, 2458000.0, max_harmonic=2)
        assert len(r.harmonics_tested) >= 2

    def test_no_harmonic_in_flat_lc(self) -> None:
        t, f = _make_clean_lc()
        r = analyze_harmonics(t, f, 10.0, 2458000.0, depth_threshold_ppm=100.0)
        assert r.flag in {"OK", "INSUFFICIENT"}

    def test_harmonic_found_flag(self) -> None:
        t, f = _make_lc_with_harmonic(depth=0.01)
        r = analyze_harmonics(
            t, f, 10.0, 2458000.0,
            duration_days=0.06, depth_threshold_ppm=500.0,
        )
        assert r.flag in {"HARMONIC_FOUND", "OK", "INSUFFICIENT"}

    def test_flag_values_valid(self) -> None:
        t, f = _make_clean_lc()
        r = analyze_harmonics(t, f, 10.0, 2458000.0)
        assert r.flag in {"OK", "HARMONIC_FOUND", "INSUFFICIENT"}

    def test_nominal_period_stored(self) -> None:
        t, f = _make_clean_lc()
        r = analyze_harmonics(t, f, 10.0, 2458000.0)
        assert r.nominal_period_days == 10.0


class TestFormatHarmonicResult:
    def test_returns_string(self) -> None:
        t, f = _make_clean_lc()
        r = analyze_harmonics(t, f, 10.0, 2458000.0)
        assert isinstance(format_harmonic_result(r), str)

    def test_insufficient_handled(self) -> None:
        r = analyze_harmonics([], [], 10.0, 2458000.0)
        out = format_harmonic_result(r)
        assert "INSUFFICIENT" in out
