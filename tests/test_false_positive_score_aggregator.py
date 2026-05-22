"""Tests for false_positive_score_aggregator.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from false_positive_score_aggregator import (
    FPDiagnostic,
    aggregate_fp_scores,
    format_fp_aggregate_result,
)


def _diag(name, score, weight=1.0, direction="fp_if_high"):
    return FPDiagnostic(name=name, score=score, weight=weight, direction=direction)


class TestAggregateFPScores:
    def test_result_frozen(self):
        r = aggregate_fp_scores([_diag("odd_even", 0.5)])
        try:
            r.composite_fp_prob = 99.0  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except (AttributeError, TypeError):
            pass

    def test_diag_frozen(self):
        d = _diag("test", 0.5)
        try:
            d.score = 0.9  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except (AttributeError, TypeError):
            pass

    def test_empty_diagnostics_invalid(self):
        r = aggregate_fp_scores([])
        assert r.flag == "INVALID"

    def test_all_none_scores_insufficient(self):
        r = aggregate_fp_scores([_diag("a", None), _diag("b", None)])
        assert r.flag == "INSUFFICIENT"

    def test_clean_planet_low_fp(self):
        diags = [
            _diag("odd_even", 0.05, 1.0, "fp_if_high"),
            _diag("secondary", 0.05, 1.0, "fp_if_high"),
        ]
        r = aggregate_fp_scores(diags)
        assert r.composite_fp_prob < 0.3
        assert r.flag == "OK"

    def test_clear_fp_high_score(self):
        diags = [
            _diag("odd_even", 0.95, 1.0, "fp_if_high"),
            _diag("secondary", 0.90, 1.0, "fp_if_high"),
        ]
        r = aggregate_fp_scores(diags)
        assert r.composite_fp_prob > 0.7

    def test_fp_if_low_direction(self):
        # High planet score → direction fp_if_low → low fp_score
        diags = [_diag("planet_prob", 0.9, 1.0, "fp_if_low")]
        r = aggregate_fp_scores(diags)
        assert r.composite_fp_prob < 0.5

    def test_mixed_none_and_real(self):
        diags = [_diag("a", None), _diag("b", 0.7, 1.0, "fp_if_high")]
        r = aggregate_fp_scores(diags)
        assert r.flag == "OK"
        assert r.n_active == 1

    def test_n_diagnostics_correct(self):
        diags = [_diag("a", 0.5), _diag("b", 0.3), _diag("c", None)]
        r = aggregate_fp_scores(diags)
        assert r.n_diagnostics == 3

    def test_n_active_correct(self):
        diags = [_diag("a", 0.5), _diag("b", None)]
        r = aggregate_fp_scores(diags)
        assert r.n_active == 1

    def test_invalid_score_out_of_range(self):
        r = aggregate_fp_scores([_diag("a", 1.5)])
        assert r.flag == "INVALID"

    def test_invalid_negative_weight(self):
        r = aggregate_fp_scores([FPDiagnostic("a", 0.5, -1.0, "fp_if_high")])
        assert r.flag == "INVALID"

    def test_composite_in_0_1(self):
        diags = [_diag("a", 0.3), _diag("b", 0.7)]
        r = aggregate_fp_scores(diags)
        assert 0.0 <= r.composite_fp_prob <= 1.0

    def test_dominant_diagnostic_is_highest_weight(self):
        diags = [_diag("low", 0.5, 0.1), _diag("high", 0.5, 2.0)]
        r = aggregate_fp_scores(diags)
        assert r.dominant_diagnostic == "high"

    def test_format_returns_string(self):
        r = aggregate_fp_scores([_diag("test", 0.5)])
        s = format_fp_aggregate_result(r)
        assert isinstance(s, str)
        assert "False-Positive" in s or "FP" in s or "False" in s
