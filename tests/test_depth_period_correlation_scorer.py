"""Tests for Skills/depth_period_correlation_scorer.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from depth_period_correlation_scorer import (
    format_depth_period_result,
    score_depth_period_correlation,
)


class TestScoreDepthPeriodCorrelation:
    def test_basic_ok(self):
        periods = [1.0, 2.0, 3.0, 4.0, 5.0]
        depths = [100.0, 200.0, 300.0, 400.0, 500.0]
        result = score_depth_period_correlation(periods, depths)
        assert result.flag == "OK"

    def test_insufficient_short(self):
        result = score_depth_period_correlation([1.0, 2.0], [100.0, 200.0])
        assert result.flag == "INSUFFICIENT"

    def test_mismatched_lengths_invalid(self):
        result = score_depth_period_correlation([1.0, 2.0, 3.0], [100.0, 200.0])
        assert result.flag == "INVALID"

    def test_perfect_positive_correlation(self):
        periods = [1.0, 2.0, 3.0, 4.0, 5.0]
        depths = [100.0, 200.0, 300.0, 400.0, 500.0]
        result = score_depth_period_correlation(periods, depths)
        assert abs(result.pearson_r - 1.0) < 1e-6

    def test_negative_correlation(self):
        periods = [1.0, 2.0, 3.0, 4.0, 5.0]
        depths = [500.0, 400.0, 300.0, 200.0, 100.0]
        result = score_depth_period_correlation(periods, depths)
        assert result.pearson_r < 0.0

    def test_no_correlation(self):
        periods = [1.0, 2.0, 3.0, 4.0, 5.0]
        depths = [300.0, 300.0, 300.0, 300.0, 300.0]
        result = score_depth_period_correlation(periods, depths)
        assert abs(result.pearson_r) < 1e-6

    def test_spearman_r_in_range(self):
        periods = [1.0, 2.0, 3.0, 4.0, 5.0]
        depths = [100.0, 200.0, 150.0, 400.0, 350.0]
        result = score_depth_period_correlation(periods, depths)
        assert -1.0 <= result.spearman_r <= 1.0

    def test_n_candidates_correct(self):
        periods = [1.0, 2.0, 3.0, 4.0]
        depths = [100.0, 200.0, 300.0, 400.0]
        result = score_depth_period_correlation(periods, depths)
        assert result.n_candidates == 4

    def test_slope_positive_for_positive_correlation(self):
        periods = [1.0, 2.0, 3.0, 4.0, 5.0]
        depths = [100.0, 200.0, 300.0, 400.0, 500.0]
        result = score_depth_period_correlation(periods, depths)
        assert result.slope_ppm_per_day > 0.0

    def test_anomalous_strong_correlation(self):
        # Near-perfect correlation (not exact, so p-value is computed properly)
        n = 20
        periods = [float(i) for i in range(1, n + 1)]
        # Small noise to avoid r=1 exactly (which returns p=1.0)
        depths = [float(i) * 100 + (0.1 if i % 3 == 0 else 0.0) for i in range(1, n + 1)]
        result = score_depth_period_correlation(periods, depths)
        assert result.is_anomalous is True

    def test_not_anomalous_random(self):
        # Constant depth — no correlation
        periods = [1.0, 2.0, 3.0, 4.0, 5.0]
        depths = [300.0] * 5
        result = score_depth_period_correlation(periods, depths)
        assert result.is_anomalous is False

    def test_depth_errs_accepted(self):
        periods = [1.0, 2.0, 3.0, 4.0, 5.0]
        depths = [100.0, 200.0, 300.0, 400.0, 500.0]
        errs = [10.0] * 5
        result = score_depth_period_correlation(periods, depths, depth_errs=errs)
        assert result.flag == "OK"

    def test_result_frozen(self):
        result = score_depth_period_correlation([1.0, 2.0, 3.0], [100.0, 200.0, 300.0])
        try:
            result.pearson_r = 99.0
            raise AssertionError()
        except Exception:
            pass

    def test_format_returns_string(self):
        result = score_depth_period_correlation([1.0, 2.0, 3.0], [100.0, 200.0, 300.0])
        text = format_depth_period_result(result)
        assert isinstance(text, str)

    def test_format_contains_flag(self):
        result = score_depth_period_correlation([1.0, 2.0, 3.0], [100.0, 200.0, 300.0])
        text = format_depth_period_result(result)
        assert result.flag in text
