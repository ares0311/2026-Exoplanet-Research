"""Tests for Skills/false_negative_rate_estimator.py"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from false_negative_rate_estimator import (
    FNRResult,
    estimate_false_negative_rate,
    format_fnr_result,
)


class TestEstimateFalseNegativeRate:
    def test_basic_ok(self):
        scores = [0.8, 0.3, 0.9, 0.2, 0.7]
        labels = [1, 0, 1, 0, 1]
        result = estimate_false_negative_rate(scores, labels)
        assert result.flag == "OK"

    def test_perfect_classifier(self):
        scores = [0.9, 0.9, 0.9, 0.1, 0.1]
        labels = [1, 1, 1, 0, 0]
        result = estimate_false_negative_rate(scores, labels, threshold=0.5)
        assert result.false_negative_rate == 0.0
        assert result.true_positive_rate == 1.0

    def test_worst_classifier(self):
        scores = [0.1, 0.1, 0.1]
        labels = [1, 1, 1]
        result = estimate_false_negative_rate(scores, labels, threshold=0.5)
        assert result.false_negative_rate == 1.0
        assert result.true_positive_rate == 0.0

    def test_fnr_plus_tpr_equals_one(self):
        scores = [0.6, 0.4, 0.8, 0.2, 0.7]
        labels = [1, 1, 1, 0, 0]
        result = estimate_false_negative_rate(scores, labels)
        if result.n_positives > 0:
            assert abs(result.false_negative_rate + result.true_positive_rate - 1.0) < 1e-9

    def test_no_positives_insufficient(self):
        scores = [0.1, 0.2]
        labels = [0, 0]
        result = estimate_false_negative_rate(scores, labels)
        assert result.flag == "INSUFFICIENT"

    def test_mismatched_lengths_invalid(self):
        result = estimate_false_negative_rate([0.5, 0.6], [1])
        assert result.flag == "INVALID"

    def test_empty_insufficient(self):
        result = estimate_false_negative_rate([], [])
        assert result.flag == "INSUFFICIENT"

    def test_threshold_stored(self):
        result = estimate_false_negative_rate([0.6, 0.4], [1, 0], threshold=0.7)
        assert result.threshold == 0.7

    def test_n_positives_counted(self):
        labels = [1, 0, 1, 1, 0]
        result = estimate_false_negative_rate([0.8] * 5, labels)
        assert result.n_positives == 3

    def test_n_negatives_counted(self):
        labels = [1, 0, 1, 1, 0]
        result = estimate_false_negative_rate([0.8] * 5, labels)
        assert result.n_negatives == 2

    def test_result_frozen(self):
        result = estimate_false_negative_rate([0.8, 0.2], [1, 0])
        try:
            result.false_negative_rate = 99.0
            assert False
        except Exception:
            pass

    def test_custom_threshold_high(self):
        scores = [0.9, 0.7, 0.4]
        labels = [1, 1, 1]
        result = estimate_false_negative_rate(scores, labels, threshold=0.8)
        assert result.n_missed == 2
        assert result.false_negative_rate == pytest.approx(2 / 3)

    def test_format_returns_string(self):
        result = estimate_false_negative_rate([0.8, 0.2], [1, 0])
        text = format_fnr_result(result)
        assert isinstance(text, str)

    def test_format_contains_flag(self):
        result = estimate_false_negative_rate([0.8, 0.2], [1, 0])
        text = format_fnr_result(result)
        assert result.flag in text
