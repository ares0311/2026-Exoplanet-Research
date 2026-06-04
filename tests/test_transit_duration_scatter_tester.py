"""Tests for Skills/transit_duration_scatter_tester.py."""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from transit_duration_scatter_tester import (
    format_duration_scatter_result,
)
from transit_duration_scatter_tester import (
    test_duration_scatter as run_scatter_test,
)


class TestDurationScatterTester:
    def test_consistent_durations(self) -> None:
        durations = [3.0, 3.01, 2.99, 3.02, 2.98]
        r = run_scatter_test(durations)
        assert r.flag == "OK"
        assert not r.scatter_significant

    def test_insufficient_transits(self) -> None:
        r = run_scatter_test([3.0, 3.1])
        assert r.flag == "INSUFFICIENT_TRANSITS"

    def test_high_scatter_detected(self) -> None:
        durations = [1.0, 5.0, 1.0, 5.0, 1.0]
        errors = [0.01] * 5
        r = run_scatter_test(durations, errors)
        assert r.scatter_significant or r.chi2_reduced > 1.0

    def test_mean_duration_correct(self) -> None:
        durations = [3.0, 3.0, 3.0, 3.0, 3.0]
        r = run_scatter_test(durations)
        assert abs(r.mean_duration_hours - 3.0) < 1e-6

    def test_rms_scatter_zero_for_identical(self) -> None:
        durations = [3.0, 3.0, 3.0, 3.0, 3.0]
        r = run_scatter_test(durations)
        assert abs(r.rms_scatter_hours) < 1e-9

    def test_with_errors(self) -> None:
        durations = [3.0, 3.02, 2.98, 3.01, 2.99]
        errors = [0.05] * 5
        r = run_scatter_test(durations, errors)
        assert math.isfinite(r.chi2_reduced)

    def test_n_transits_count(self) -> None:
        durations = [3.0, 3.1, 2.9, 3.0]
        r = run_scatter_test(durations)
        assert r.n_transits == 4

    def test_p_value_in_range(self) -> None:
        durations = [3.0, 3.0, 3.0, 3.0, 3.0]
        r = run_scatter_test(durations)
        if math.isfinite(r.p_value_approx):
            assert 0.0 <= r.p_value_approx <= 1.0

    def test_chi2_nonneg(self) -> None:
        durations = [3.0, 3.1, 2.9]
        r = run_scatter_test(durations)
        assert r.chi2_reduced >= 0

    def test_custom_threshold(self) -> None:
        durations = [1.0, 2.0, 3.0, 4.0, 5.0]
        r_strict = run_scatter_test(durations, significance_threshold=0.1)
        r_loose = run_scatter_test(durations, significance_threshold=1000.0)
        assert r_strict.scatter_significant
        assert not r_loose.scatter_significant

    def test_format_output(self) -> None:
        r = run_scatter_test([3.0, 3.01, 2.99, 3.0])
        s = format_duration_scatter_result(r)
        assert "|" in s
        assert "scatter" in s.lower() or "T14" in s

    def test_empty_input(self) -> None:
        r = run_scatter_test([])
        assert r.flag == "INSUFFICIENT_TRANSITS"
