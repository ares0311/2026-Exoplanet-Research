"""Tests for Skills/cnn_threshold_optimizer.py"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from cnn_threshold_optimizer import (
    format_threshold_result,
    optimize_threshold,
)

_Y_TRUE = [1, 1, 0, 0, 1, 0, 1, 0]
_Y_SCORE = [0.9, 0.8, 0.2, 0.1, 0.7, 0.3, 0.85, 0.15]


def test_flag_ok():
    r = optimize_threshold(_Y_TRUE, _Y_SCORE)
    assert r.flag == "OK"


def test_threshold_in_range():
    r = optimize_threshold(_Y_TRUE, _Y_SCORE)
    assert 0.0 <= r.threshold <= 1.0


def test_f1_at_good_threshold():
    # Perfect classifier → optimal threshold → F1 close to 1.0
    r = optimize_threshold(_Y_TRUE, _Y_SCORE, objective="f1")
    assert r.f1 > 0.8


def test_balanced_accuracy_objective():
    r = optimize_threshold(_Y_TRUE, _Y_SCORE, objective="balanced_accuracy")
    assert r.flag == "OK"
    assert 0.0 <= r.balanced_accuracy <= 1.0


def test_youden_j_objective():
    r = optimize_threshold(_Y_TRUE, _Y_SCORE, objective="youden_j")
    assert r.flag == "OK"
    assert -1.0 <= r.youden_j <= 1.0


def test_empty_invalid():
    r = optimize_threshold([], [])
    assert r.flag == "INVALID"


def test_mismatched_lengths_invalid():
    r = optimize_threshold([1, 0], [0.9])
    assert r.flag == "INVALID"


def test_unknown_objective_invalid():
    r = optimize_threshold(_Y_TRUE, _Y_SCORE, objective="accuracy")
    assert r.flag == "INVALID"


def test_zero_steps_invalid():
    r = optimize_threshold(_Y_TRUE, _Y_SCORE, n_steps=0)
    assert r.flag == "INVALID"


def test_negative_steps_invalid():
    r = optimize_threshold(_Y_TRUE, _Y_SCORE, n_steps=-1)
    assert r.flag == "INVALID"


def test_non_binary_label_invalid():
    r = optimize_threshold([1, 2, 0], [0.9, 0.8, 0.1])
    assert r.flag == "INVALID"


def test_bool_label_invalid():
    r = optimize_threshold([True, 0, 1], [0.9, 0.2, 0.8])  # type: ignore[list-item]
    assert r.flag == "INVALID"


def test_nan_score_invalid():
    r = optimize_threshold([1, 0], [math.nan, 0.2])
    assert r.flag == "INVALID"


def test_out_of_range_score_invalid():
    r = optimize_threshold([1, 0], [1.1, 0.2])
    assert r.flag == "INVALID"


def test_string_score_invalid():
    r = optimize_threshold([1, 0], ["0.9", 0.2])  # type: ignore[list-item]
    assert r.flag == "INVALID"


def test_degenerate_all_positive():
    y_true = [1, 1, 1, 1]
    y_score = [0.9, 0.8, 0.7, 0.6]
    r = optimize_threshold(y_true, y_score)
    assert r.flag == "DEGENERATE"


def test_precision_recall_f1_relation():
    r = optimize_threshold(_Y_TRUE, _Y_SCORE)
    if r.precision + r.recall > 0:
        expected_f1 = 2 * r.precision * r.recall / (r.precision + r.recall)
        assert abs(r.f1 - expected_f1) < 1e-6


def test_n_positive_correct():
    r = optimize_threshold(_Y_TRUE, _Y_SCORE)
    assert r.n_positive == sum(_Y_TRUE)


def test_n_negative_correct():
    r = optimize_threshold(_Y_TRUE, _Y_SCORE)
    assert r.n_negative == len(_Y_TRUE) - sum(_Y_TRUE)


def test_format_returns_string():
    r = optimize_threshold(_Y_TRUE, _Y_SCORE)
    s = format_threshold_result(r)
    assert isinstance(s, str)
    assert "Threshold" in s


def test_result_frozen():
    r = optimize_threshold(_Y_TRUE, _Y_SCORE)
    try:
        r.threshold = 0.9  # type: ignore[misc]
        raise AssertionError("Should be frozen")
    except Exception:
        pass
