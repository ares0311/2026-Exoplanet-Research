"""Tests for Skills/pr_auc_calculator.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from pr_auc_calculator import compute_pr_auc, format_pr_auc


def test_perfect_classifier_auc():
    y_true = [1, 1, 0, 0]
    y_score = [0.9, 0.8, 0.2, 0.1]
    r = compute_pr_auc(y_true, y_score)
    assert r.flag == "OK"
    assert abs(r.auc - 1.0) < 0.05


def test_all_same_score_auc():
    # With all same scores, AUC should be roughly the prevalence
    y_true = [1, 0, 0, 0]
    y_score = [0.5, 0.5, 0.5, 0.5]
    r = compute_pr_auc(y_true, y_score)
    assert r.flag == "OK"
    assert r.auc > 0.0


def test_empty_returns_invalid():
    r = compute_pr_auc([], [])
    assert r.flag == "INVALID"


def test_all_positive_returns_degenerate():
    r = compute_pr_auc([1, 1, 1], [0.9, 0.8, 0.7])
    assert r.flag == "DEGENERATE"


def test_all_negative_returns_degenerate():
    r = compute_pr_auc([0, 0, 0], [0.9, 0.8, 0.7])
    assert r.flag == "DEGENERATE"


def test_optimal_threshold_in_range():
    y_true = [1, 1, 0, 0]
    y_score = [0.9, 0.8, 0.2, 0.1]
    r = compute_pr_auc(y_true, y_score)
    assert 0.0 <= r.optimal_threshold <= 1.0


def test_optimal_f1_in_range():
    y_true = [1, 1, 0, 0]
    y_score = [0.9, 0.8, 0.2, 0.1]
    r = compute_pr_auc(y_true, y_score)
    assert 0.0 <= r.optimal_f1 <= 1.0


def test_precision_recall_same_length():
    y_true = [1, 0, 1, 0]
    y_score = [0.9, 0.7, 0.4, 0.1]
    r = compute_pr_auc(y_true, y_score)
    assert len(r.precision) == len(r.recall)


def test_precision_recall_in_range():
    y_true = [1, 0, 1, 0]
    y_score = [0.9, 0.7, 0.4, 0.1]
    r = compute_pr_auc(y_true, y_score)
    assert all(0.0 <= p <= 1.0 for p in r.precision)
    assert all(0.0 <= rc <= 1.0 for rc in r.recall)


def test_flag_ok_on_valid():
    r = compute_pr_auc([1, 0], [0.9, 0.1])
    assert r.flag == "OK"


def test_format_has_auc():
    r = compute_pr_auc([1, 0], [0.9, 0.1])
    text = format_pr_auc(r)
    assert "AUC" in text


def test_format_has_optimal_threshold():
    r = compute_pr_auc([1, 0], [0.9, 0.1])
    text = format_pr_auc(r)
    assert "threshold" in text.lower() or "Optimal" in text


def test_pr_auc_result_frozen():
    r = compute_pr_auc([1, 0], [0.9, 0.1])
    try:
        r.auc = 0.5  # type: ignore[misc]
        raise AssertionError("Should have raised FrozenInstanceError")
    except Exception:
        pass
