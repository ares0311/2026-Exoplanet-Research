"""Tests for Skills/transit_gap_coverage_estimator.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from transit_gap_coverage_estimator import (
    compute_transit_gap_coverage,
    format_transit_gap_coverage_result,
)


def _uniform_timestamps(n: int = 100, baseline_days: float = 27.0) -> list[float]:
    return [2457000.0 + i * baseline_days / (n - 1) for i in range(n)]


class TestComputeTransitGapCoverage:
    def test_ok_flag(self) -> None:
        r = compute_transit_gap_coverage(_uniform_timestamps(), 3.0)
        assert r.flag == "OK"

    def test_coverage_between_zero_and_one(self) -> None:
        r = compute_transit_gap_coverage(_uniform_timestamps(), 3.0)
        assert 0.0 <= r.phase_coverage_fraction <= 1.0

    def test_uniform_sampling_high_coverage(self) -> None:
        r = compute_transit_gap_coverage(_uniform_timestamps(200), 3.0)
        assert r.phase_coverage_fraction > 0.8

    def test_n_timestamps_correct(self) -> None:
        ts = _uniform_timestamps(50)
        r = compute_transit_gap_coverage(ts, 3.0)
        assert r.n_timestamps == 50

    def test_baseline_correct(self) -> None:
        ts = _uniform_timestamps(100, baseline_days=27.0)
        r = compute_transit_gap_coverage(ts, 3.0)
        assert abs(r.baseline_days - 27.0) < 0.1

    def test_largest_gap_between_zero_and_one(self) -> None:
        r = compute_transit_gap_coverage(_uniform_timestamps(), 3.0)
        assert 0.0 <= r.largest_gap_phase <= 1.0

    def test_prob_missed_between_zero_and_one(self) -> None:
        r = compute_transit_gap_coverage(_uniform_timestamps(), 3.0)
        assert 0.0 <= r.prob_transit_missed <= 1.0

    def test_sparse_sampling_lower_coverage(self) -> None:
        r_dense = compute_transit_gap_coverage(_uniform_timestamps(200), 3.0)
        r_sparse = compute_transit_gap_coverage(_uniform_timestamps(10), 3.0)
        assert r_dense.phase_coverage_fraction >= r_sparse.phase_coverage_fraction

    def test_insufficient_timestamps(self) -> None:
        r = compute_transit_gap_coverage([2457000.0], 3.0)
        assert r.flag == "INSUFFICIENT_TIMESTAMPS"

    def test_invalid_period(self) -> None:
        r = compute_transit_gap_coverage(_uniform_timestamps(), 0.0)
        assert r.flag == "INVALID_PERIOD"

    def test_bins_covered_le_total(self) -> None:
        r = compute_transit_gap_coverage(_uniform_timestamps(), 3.0)
        assert r.n_phase_bins_covered <= r.n_phase_bins_total

    def test_format_returns_string(self) -> None:
        r = compute_transit_gap_coverage(_uniform_timestamps(), 3.0)
        s = format_transit_gap_coverage_result(r)
        assert isinstance(s, str)
        assert r.flag in s
