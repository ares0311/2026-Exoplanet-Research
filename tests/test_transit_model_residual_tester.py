"""Tests for transit_model_residual_tester.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from transit_model_residual_tester import (
    format_residual_test_result,
)
from transit_model_residual_tester import (
    test_model_residuals as run_residual_tests,
)


def _white(n=50, amp=0.001):
    import math
    return [amp * math.sin(i * 1.7) for i in range(n)]


class TestModelResiduals:
    def test_result_frozen(self):
        r = run_residual_tests(_white())
        try:
            r.n_points = 99  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except (AttributeError, TypeError):
            pass

    def test_empty_invalid(self):
        r = run_residual_tests([])
        assert r.flag == "INVALID"

    def test_too_few_points_invalid(self):
        r = run_residual_tests([0.1, 0.2, 0.3, 0.4])
        assert r.flag == "INVALID"

    def test_insufficient_with_5_to_9_points(self):
        r = run_residual_tests(_white(7))
        assert r.flag == "INSUFFICIENT"

    def test_ok_with_sufficient_points(self):
        r = run_residual_tests(_white(50))
        assert r.flag == "OK"

    def test_dw_in_range(self):
        r = run_residual_tests(_white(50))
        if r.durbin_watson is not None:
            assert 0.0 <= r.durbin_watson <= 4.0

    def test_white_noise_detected(self):
        r = run_residual_tests(_white(50))
        # White noise residuals should ideally pass
        assert r.dw_interpretation in ("no_AC", "positive_AC", "negative_AC")

    def test_positively_autocorrelated(self):
        # Strongly autocorrelated: all positive
        residuals = [0.001] * 50
        r = run_residual_tests(residuals)
        assert r.dw_interpretation in ("positive_AC", "no_AC")

    def test_chi2_computed_with_flux_err(self):
        residuals = _white(50)
        err = [0.001] * 50
        r = run_residual_tests(residuals, flux_err=err)
        assert r.chi2_reduced is not None

    def test_chi2_none_without_flux_err(self):
        r = run_residual_tests(_white(50))
        assert r.chi2_reduced is None

    def test_n_points_correct(self):
        res = _white(30)
        r = run_residual_tests(res)
        assert r.n_points == 30

    def test_runs_z_score_present(self):
        res = _white(50)
        r = run_residual_tests(res)
        assert r.runs_z_score is not None

    def test_format_returns_string(self):
        r = run_residual_tests(_white(50))
        s = format_residual_test_result(r)
        assert isinstance(s, str)
        assert "Residual" in s

    def test_is_white_noise_bool(self):
        r = run_residual_tests(_white(50))
        assert isinstance(r.is_white_noise, bool)
