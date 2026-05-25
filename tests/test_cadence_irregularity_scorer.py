"""Tests for Skills/cadence_irregularity_scorer.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from cadence_irregularity_scorer import (
    format_cadence_irregularity_result,
    score_cadence_irregularity,
)


def _uniform_time(n=100, cadence=0.02083):
    return [i * cadence for i in range(n)]


class TestScoreCadenceIrregularity:
    def test_perfectly_uniform(self):
        t = _uniform_time(100)
        result = score_cadence_irregularity(t)
        assert result.flag == "OK"
        assert result.irregularity_score < 0.01

    def test_too_short_insufficient(self):
        result = score_cadence_irregularity([0.0, 1.0])
        assert result.flag == "INSUFFICIENT"

    def test_n_cadences_correct(self):
        t = _uniform_time(50)
        result = score_cadence_irregularity(t)
        assert result.n_cadences == 50

    def test_median_gap_positive(self):
        t = _uniform_time(100)
        result = score_cadence_irregularity(t)
        assert result.median_gap_min > 0.0

    def test_gap_std_near_zero_for_uniform(self):
        t = _uniform_time(100)
        result = score_cadence_irregularity(t)
        assert result.gap_std_min < 1.0

    def test_outlier_gap_detected(self):
        t = list(range(50))
        t[25] = t[24] + 100  # large gap
        for i in range(26, 50):
            t[i] = t[25] + (i - 25)
        result = score_cadence_irregularity([float(x) for x in t])
        assert result.n_outlier_gaps >= 1

    def test_custom_expected_cadence(self):
        t = _uniform_time(100, cadence=0.5)
        result = score_cadence_irregularity(t, expected_cadence_min=30.0)
        assert result.flag == "OK"

    def test_negative_gap_invalid(self):
        result = score_cadence_irregularity([1.0, 0.5, 2.0, 3.0])
        assert result.flag == "INVALID"

    def test_monotone_increases_ok(self):
        t = [0.0, 1.0, 2.0, 3.0, 4.0]
        result = score_cadence_irregularity(t)
        assert result.flag == "OK"

    def test_irregularity_score_nonneg(self):
        t = _uniform_time(100)
        result = score_cadence_irregularity(t)
        assert result.irregularity_score >= 0.0

    def test_result_frozen(self):
        t = _uniform_time(100)
        result = score_cadence_irregularity(t)
        try:
            result.irregularity_score = 99.0
            raise AssertionError()
        except Exception:
            pass

    def test_format_returns_string(self):
        t = _uniform_time(100)
        result = score_cadence_irregularity(t)
        text = format_cadence_irregularity_result(result)
        assert isinstance(text, str)

    def test_format_contains_flag(self):
        t = _uniform_time(100)
        result = score_cadence_irregularity(t)
        text = format_cadence_irregularity_result(result)
        assert result.flag in text
