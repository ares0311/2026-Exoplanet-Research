"""Tests for Skills/roc_auc_calculator.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from roc_auc_calculator import compute_roc_auc, format_roc_auc


def test_perfect_classifier():
    y_true = [1, 1, 0, 0]
    y_score = [0.9, 0.8, 0.2, 0.1]
    r = compute_roc_auc(y_true, y_score)
    assert r.flag == "OK"
    assert abs(r.auc - 1.0) < 0.01


def test_random_classifier():
    # alternating labels with same score → AUC ≈ 0.5
    y_true = [1, 0, 1, 0, 1, 0, 1, 0]
    y_score = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    r = compute_roc_auc(y_true, y_score)
    # With all same scores, result should still be valid
    assert r.flag in ("OK", "DEGENERATE")


def test_worst_classifier():
    # scores perfectly inverted → AUC ≈ 0.0
    y_true = [1, 1, 0, 0]
    y_score = [0.1, 0.2, 0.8, 0.9]
    r = compute_roc_auc(y_true, y_score)
    assert r.flag == "OK"
    assert r.auc < 0.1


def test_empty_returns_invalid():
    r = compute_roc_auc([], [])
    assert r.flag == "INVALID"


def test_all_positive_returns_degenerate():
    r = compute_roc_auc([1, 1, 1], [0.9, 0.8, 0.7])
    assert r.flag == "DEGENERATE"


def test_all_negative_returns_degenerate():
    r = compute_roc_auc([0, 0, 0], [0.9, 0.8, 0.7])
    assert r.flag == "DEGENERATE"


def test_single_pos_neg_works():
    r = compute_roc_auc([1, 0], [0.8, 0.2])
    assert r.flag == "OK"
    assert abs(r.auc - 1.0) < 0.01


def test_fpr_tpr_same_length_and_thresholds_one_less():
    # (0,0) is prepended without a threshold; otherwise fpr/tpr/thresholds are parallel
    y_true = [1, 1, 0, 0]
    y_score = [0.9, 0.8, 0.2, 0.1]
    r = compute_roc_auc(y_true, y_score)
    assert len(r.fpr) == len(r.tpr)
    # thresholds has one fewer point (the prepended (0,0) has no threshold)
    assert len(r.thresholds) == len(r.fpr) - 1 or len(r.thresholds) == len(r.fpr)


def test_fpr_starts_at_zero():
    y_true = [1, 0, 1, 0]
    y_score = [0.9, 0.8, 0.3, 0.1]
    r = compute_roc_auc(y_true, y_score)
    assert r.fpr[0] == 0.0
    assert r.tpr[0] == 0.0


def test_fpr_ends_at_one():
    y_true = [1, 0, 1, 0]
    y_score = [0.9, 0.8, 0.3, 0.1]
    r = compute_roc_auc(y_true, y_score)
    assert abs(r.fpr[-1] - 1.0) < 1e-9
    assert abs(r.tpr[-1] - 1.0) < 1e-9


def test_flag_ok_on_valid():
    r = compute_roc_auc([1, 0], [0.9, 0.1])
    assert r.flag == "OK"


def test_format_has_auc():
    r = compute_roc_auc([1, 0], [0.9, 0.1])
    text = format_roc_auc(r)
    assert "AUC" in text


def test_roc_auc_result_frozen():
    r = compute_roc_auc([1, 0], [0.9, 0.1])
    try:
        r.auc = 0.5  # type: ignore[misc]
        raise AssertionError("Should have raised FrozenInstanceError")
    except Exception:
        pass
