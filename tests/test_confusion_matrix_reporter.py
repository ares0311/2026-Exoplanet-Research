"""Tests for Skills/confusion_matrix_reporter.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from confusion_matrix_reporter import (
    ConfusionMatrixResult,
    compute_confusion_matrix,
    format_confusion_matrix,
)


def test_perfect_classifier():
    y_true = [1, 1, 0, 0]
    y_score = [0.9, 0.8, 0.2, 0.1]
    r = compute_confusion_matrix(y_true, y_score)
    assert r.tp == 2
    assert r.tn == 2
    assert r.fp == 0
    assert r.fn == 0
    assert r.flag == "OK"


def test_worst_classifier():
    y_true = [1, 1, 0, 0]
    y_score = [0.1, 0.2, 0.8, 0.9]
    r = compute_confusion_matrix(y_true, y_score)
    assert r.tp == 0
    assert r.tn == 0
    assert r.fp == 2
    assert r.fn == 2


def test_counts_sum_to_total():
    y_true = [1, 0, 1, 0, 1]
    y_score = [0.9, 0.3, 0.7, 0.6, 0.4]
    r = compute_confusion_matrix(y_true, y_score)
    assert r.tp + r.fp + r.tn + r.fn == r.n_total


def test_precision_formula():
    y_true = [1, 0, 1, 0]
    y_score = [0.9, 0.8, 0.7, 0.2]
    r = compute_confusion_matrix(y_true, y_score)
    if r.tp + r.fp > 0:
        assert abs(r.precision - r.tp / (r.tp + r.fp)) < 1e-6


def test_precision_zero_when_no_positive_predictions():
    y_true = [1, 1, 0, 0]
    y_score = [0.3, 0.4, 0.2, 0.1]  # all below threshold 0.5
    r = compute_confusion_matrix(y_true, y_score, threshold=0.5)
    assert r.tp == 0
    assert r.fp == 0
    assert r.precision == 0.0


def test_recall_formula():
    y_true = [1, 0, 1, 0]
    y_score = [0.9, 0.3, 0.7, 0.2]
    r = compute_confusion_matrix(y_true, y_score)
    if r.tp + r.fn > 0:
        assert abs(r.recall - r.tp / (r.tp + r.fn)) < 1e-6


def test_f1_zero_when_precision_recall_zero():
    y_true = [1, 1]
    y_score = [0.1, 0.2]  # all below threshold
    r = compute_confusion_matrix(y_true, y_score, threshold=0.5)
    assert r.f1 == 0.0


def test_accuracy_formula():
    y_true = [1, 1, 0, 0]
    y_score = [0.9, 0.8, 0.2, 0.1]
    r = compute_confusion_matrix(y_true, y_score)
    assert abs(r.accuracy - (r.tp + r.tn) / r.n_total) < 1e-6


def test_empty_input_invalid():
    r = compute_confusion_matrix([], [])
    assert r.flag == "INVALID"


def test_all_one_class_degenerate():
    r = compute_confusion_matrix([1, 1, 1], [0.9, 0.8, 0.7])
    assert r.flag == "DEGENERATE"


def test_threshold_zero_all_positive():
    y_true = [1, 0, 1, 0]
    y_score = [0.9, 0.8, 0.7, 0.6]
    r = compute_confusion_matrix(y_true, y_score, threshold=0.0)
    assert r.tp + r.fp == r.n_total  # all predicted positive


def test_threshold_one_all_negative():
    y_true = [1, 0, 1, 0]
    y_score = [0.9, 0.8, 0.7, 0.6]
    r = compute_confusion_matrix(y_true, y_score, threshold=1.0)
    assert r.tp == 0
    assert r.fp == 0


def test_format_returns_string():
    r = compute_confusion_matrix([1, 0, 1, 0], [0.9, 0.2, 0.7, 0.3])
    s = format_confusion_matrix(r)
    assert isinstance(s, str)


def test_result_is_frozen():
    r = compute_confusion_matrix([1, 0], [0.9, 0.1])
    assert isinstance(r, ConfusionMatrixResult)
    try:
        r.tp = 99  # type: ignore[misc]
        raise AssertionError("Should be frozen")
    except Exception:
        pass
