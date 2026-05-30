"""Tests for Skills/period_window_coverage_checker.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from period_window_coverage_checker import (
    check_period_window_coverage,
    format_window_coverage,
)


class TestCheckPeriodWindowCoverage:
    def test_perfect_coverage(self):
        # 5 windows, one observation per window
        epoch = 0.0
        period = 2.0
        obs = [0.0, 2.0, 4.0, 6.0, 8.0]
        result = check_period_window_coverage(obs, period, epoch, 10.0)
        assert result.n_windows_observed == result.n_windows_total
        assert result.coverage_fraction == 1.0

    def test_partial_coverage(self):
        epoch = 0.0
        period = 2.0
        # Only cover windows 0, 2 — miss windows 1, 3, 4
        obs = [0.0, 4.0]
        result = check_period_window_coverage(obs, period, epoch, 10.0)
        assert result.n_windows_observed == 2
        assert result.coverage_fraction < 1.0

    def test_no_observations(self):
        result = check_period_window_coverage([], 2.0, 0.0, 10.0)
        assert result.flag == "NO_OBSERVATIONS"
        assert result.n_windows_observed == 0
        assert result.coverage_fraction == 0.0

    def test_invalid_period(self):
        result = check_period_window_coverage([1.0, 2.0], 0.0, 0.0, 10.0)
        assert result.flag == "INVALID_PERIOD"

    def test_invalid_baseline(self):
        result = check_period_window_coverage([1.0, 2.0], 2.0, 0.0, -1.0)
        assert result.flag == "INVALID_BASELINE"

    def test_coverage_fraction_in_unit_interval(self):
        obs = [0.0, 2.5, 5.0]
        result = check_period_window_coverage(obs, 2.0, 0.0, 10.0)
        assert 0.0 <= result.coverage_fraction <= 1.0

    def test_missed_windows_correct(self):
        epoch = 0.0
        period = 2.0
        # Only observe window 0 (t=0) and window 2 (t=4)
        obs = [0.0, 4.0]
        result = check_period_window_coverage(obs, period, epoch, 10.0)
        # n_windows_total = int(10/2) = 5; windows 0..4
        assert 1 in result.missed_windows
        assert 3 in result.missed_windows
        assert 0 not in result.missed_windows

    def test_n_windows_total_correct(self):
        result = check_period_window_coverage([0.0], 3.0, 0.0, 9.0)
        assert result.n_windows_total == 3

    def test_single_window_covered(self):
        result = check_period_window_coverage([0.5], 10.0, 0.0, 5.0)
        # n_windows_total = max(1, int(5/10)) = 1; window center at 0.0
        # obs at 0.5; half_window = 2.5; |0.5 - 0.0| = 0.5 <= 2.5 → covered
        assert result.n_windows_observed == 1
        assert result.coverage_fraction == 1.0

    def test_format_returns_string(self):
        obs = [0.0, 2.0, 4.0]
        result = check_period_window_coverage(obs, 2.0, 0.0, 6.0)
        md = format_window_coverage(result)
        assert isinstance(md, str)

    def test_flag_ok_for_valid_inputs(self):
        obs = [0.0, 2.0, 4.0]
        result = check_period_window_coverage(obs, 2.0, 0.0, 6.0)
        assert result.flag == "OK"

    def test_n_observed_plus_missed_equals_total(self):
        obs = [0.0, 4.0]
        result = check_period_window_coverage(obs, 2.0, 0.0, 10.0)
        assert result.n_windows_observed + len(result.missed_windows) == result.n_windows_total
