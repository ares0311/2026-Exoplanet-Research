"""Tests for Skills.phase_coverage_checker."""
from __future__ import annotations

from Skills.phase_coverage_checker import (
    PhaseCoverageResult,
    check_phase_coverage,
    format_phase_coverage_result,
)


def _make_time(period=10.0, epoch=2458000.0, n_days=30.0):
    dt = 2.0 / 1440.0
    n = int(n_days / dt)
    return [epoch + i * dt for i in range(n)]


class TestCheckPhaseCoverage:
    def test_returns_result(self) -> None:
        t = _make_time()
        r = check_phase_coverage(t, 10.0, 2458000.0)
        assert isinstance(r, PhaseCoverageResult)

    def test_empty_returns_insufficient(self) -> None:
        r = check_phase_coverage([], 10.0, 2458000.0)
        assert r.flag == "INSUFFICIENT"

    def test_zero_period_returns_insufficient(self) -> None:
        t = _make_time()
        r = check_phase_coverage(t, 0.0, 2458000.0)
        assert r.flag == "INSUFFICIENT"

    def test_full_coverage_ok(self) -> None:
        # Dense LC → should have near-complete coverage
        t = _make_time(period=2.0, n_days=30.0)
        r = check_phase_coverage(t, 2.0, 2458000.0, n_bins=50)
        assert r.flag == "OK"

    def test_coverage_fraction_in_range(self) -> None:
        t = _make_time()
        r = check_phase_coverage(t, 10.0, 2458000.0)
        assert 0.0 <= r.coverage_fraction <= 1.0

    def test_n_covered_leq_n_bins(self) -> None:
        t = _make_time()
        r = check_phase_coverage(t, 10.0, 2458000.0, n_bins=100)
        assert r.n_covered <= r.n_bins

    def test_gap_phases_subset_of_bins(self) -> None:
        t = _make_time()
        r = check_phase_coverage(t, 10.0, 2458000.0, n_bins=100)
        for ph in r.gap_phases:
            assert 0.0 <= ph <= 1.0

    def test_poor_coverage_flag(self) -> None:
        # Only 1 data point → very low coverage
        r = check_phase_coverage([2458000.0], 10.0, 2458000.0, n_bins=100, min_coverage=0.80)
        assert r.flag == "POOR_COVERAGE"

    def test_flag_values_valid(self) -> None:
        t = _make_time()
        r = check_phase_coverage(t, 10.0, 2458000.0)
        assert r.flag in {"OK", "POOR_COVERAGE", "INSUFFICIENT"}


class TestFormatPhaseCoverage:
    def test_returns_string(self) -> None:
        t = _make_time()
        r = check_phase_coverage(t, 10.0, 2458000.0)
        assert isinstance(format_phase_coverage_result(r), str)

    def test_contains_coverage(self) -> None:
        t = _make_time()
        r = check_phase_coverage(t, 10.0, 2458000.0)
        out = format_phase_coverage_result(r)
        assert "Coverage" in out or "coverage" in out.lower()

    def test_insufficient_handled(self) -> None:
        r = check_phase_coverage([], 10.0, 2458000.0)
        out = format_phase_coverage_result(r)
        assert "INSUFFICIENT" in out
