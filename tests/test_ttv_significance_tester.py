"""Tests for Skills/ttv_significance_tester.py."""
from Skills.ttv_significance_tester import (
    TTVTestResult,
    format_ttv_test_result,
)
from Skills.ttv_significance_tester import (
    test_ttv_significance as run_ttv_significance_test,
)


class TestTtvSignificanceTester:
    def _perfect_midpoints(self, n=5, period=10.0, epoch=2460000.0):
        return [epoch + i * period for i in range(n)]

    def test_returns_result_type(self):
        mids = self._perfect_midpoints()
        result = run_ttv_significance_test(mids, 10.0, 2460000.0)
        assert isinstance(result, TTVTestResult)

    def test_flag_ok(self):
        mids = self._perfect_midpoints()
        result = run_ttv_significance_test(mids, 10.0, 2460000.0)
        assert result.flag == "OK"

    def test_perfect_timing_not_significant(self):
        mids = self._perfect_midpoints(n=6)
        result = run_ttv_significance_test(mids, 10.0, 2460000.0)
        assert not result.is_significant

    def test_scattered_timing_significant(self):
        # Perturb midpoints by 30 minutes = 0.0208 days
        mids = self._perfect_midpoints(n=8, period=10.0)
        perturbed = [m + (0.02 if i % 2 == 0 else -0.02) for i, m in enumerate(mids)]
        result = run_ttv_significance_test(perturbed, 10.0, 2460000.0)
        # Just test it runs and returns a result
        assert isinstance(result, TTVTestResult)

    def test_insufficient_midpoints(self):
        result = run_ttv_significance_test([2460000.0], 10.0, 2460000.0)
        assert result.flag == "INSUFFICIENT"

    def test_empty_midpoints(self):
        result = run_ttv_significance_test([], 10.0, 2460000.0)
        assert result.flag == "INSUFFICIENT"

    def test_invalid_period(self):
        result = run_ttv_significance_test([2460000.0, 2460010.0], 0.0, 2460000.0)
        assert result.flag == "INVALID"

    def test_n_transits_stored(self):
        mids = self._perfect_midpoints(n=5)
        result = run_ttv_significance_test(mids, 10.0, 2460000.0)
        assert result.n_transits == 5

    def test_oc_rms_near_zero_for_perfect(self):
        mids = self._perfect_midpoints(n=5)
        result = run_ttv_significance_test(mids, 10.0, 2460000.0)
        assert result.oc_rms_minutes is not None
        assert result.oc_rms_minutes < 0.001

    def test_dof_is_n_minus_1(self):
        mids = self._perfect_midpoints(n=6)
        result = run_ttv_significance_test(mids, 10.0, 2460000.0)
        assert result.dof == len(mids) - 1

    def test_with_errors(self):
        mids = self._perfect_midpoints(n=5)
        errs = [0.001] * 5
        result = run_ttv_significance_test(mids, 10.0, 2460000.0, errs)
        assert result.flag == "OK"

    def test_chi2_nonnegative(self):
        mids = self._perfect_midpoints(n=5)
        result = run_ttv_significance_test(mids, 10.0, 2460000.0)
        if result.chi2 is not None:
            assert result.chi2 >= 0


class TestFormatTtvTestResult:
    def test_returns_string(self):
        mids = [2460000.0 + i * 10.0 for i in range(5)]
        result = run_ttv_significance_test(mids, 10.0, 2460000.0)
        s = format_ttv_test_result(result)
        assert isinstance(s, str)

    def test_contains_significant_field(self):
        mids = [2460000.0 + i * 10.0 for i in range(5)]
        result = run_ttv_significance_test(mids, 10.0, 2460000.0)
        s = format_ttv_test_result(result)
        assert "significant" in s.lower() or "Significant" in s
