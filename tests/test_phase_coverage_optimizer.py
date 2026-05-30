"""Tests for Skills/phase_coverage_optimizer.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from phase_coverage_optimizer import (
    format_phase_coverage,
    optimize_phase_coverage,
)


class TestOptimizePhaseCoverage:
    def test_dense_sampling_high_coverage(self) -> None:
        # 200 evenly spaced points over one period
        ts = [i * 0.05 for i in range(200)]
        r = optimize_phase_coverage(ts, 10.0, 0.0, n_bins=20)
        assert r.coverage_fraction == 1.0
        assert r.n_bins_covered == 20

    def test_single_point_one_bin(self) -> None:
        r = optimize_phase_coverage([0.0], 10.0, 0.0, n_bins=20)
        assert r.n_bins_covered == 1
        assert r.coverage_fraction == 1.0 / 20.0

    def test_invalid_period(self) -> None:
        r = optimize_phase_coverage([1.0, 2.0], 0.0, 0.0)
        assert r.flag == "INVALID_PERIOD"
        assert r.coverage_fraction == 0.0

    def test_no_data(self) -> None:
        r = optimize_phase_coverage([], 10.0, 0.0)
        assert r.flag == "NO_DATA"
        assert r.n_bins_covered == 0

    def test_coverage_fraction_in_range(self) -> None:
        ts = [i * 0.5 for i in range(50)]
        r = optimize_phase_coverage(ts, 10.0, 0.0)
        assert 0.0 <= r.coverage_fraction <= 1.0

    def test_gap_phases_length(self) -> None:
        r = optimize_phase_coverage([0.0], 10.0, 0.0, n_bins=20)
        assert len(r.gap_phases) == 20 - r.n_bins_covered

    def test_gap_phases_in_zero_one(self) -> None:
        r = optimize_phase_coverage([0.0], 10.0, 0.0, n_bins=10)
        for p in r.gap_phases:
            assert 0.0 <= p < 1.0

    def test_bins_covered_plus_gaps_equals_total(self) -> None:
        ts = [i * 1.0 for i in range(10)]
        r = optimize_phase_coverage(ts, 7.0, 0.0, n_bins=20)
        assert r.n_bins_covered + len(r.gap_phases) == r.n_bins_total

    def test_flag_ok_for_valid(self) -> None:
        r = optimize_phase_coverage([0.0, 2.5, 5.0, 7.5], 10.0, 0.0)
        assert r.flag == "OK"

    def test_full_coverage_no_gaps(self) -> None:
        ts = [i * 0.1 for i in range(100)]
        r = optimize_phase_coverage(ts, 10.0, 0.0, n_bins=20)
        assert r.gap_phases == []

    def test_n_bins_total_stored(self) -> None:
        r = optimize_phase_coverage([0.0], 10.0, 0.0, n_bins=15)
        assert r.n_bins_total == 15

    def test_format_returns_string(self) -> None:
        r = optimize_phase_coverage([0.0, 2.5], 10.0, 0.0)
        s = format_phase_coverage(r)
        assert isinstance(s, str)
        assert "Coverage" in s
