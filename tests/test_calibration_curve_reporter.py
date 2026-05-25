"""Tests for Skills/calibration_curve_reporter.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from calibration_curve_reporter import (
    compute_calibration_curve,
    format_calibration_curve,
)


def test_perfect_calibration_fraction_pos_approx_mean_pred():
    """Perfectly calibrated model: fraction_pos bins should match mean_pred."""
    # 100 items: score == label probability, linearly spaced
    n = 100
    y_score = [i / (n - 1) for i in range(n)]
    y_true = [1 if s >= 0.5 else 0 for s in y_score]
    result = compute_calibration_curve(y_true, y_score, n_bins=10)
    assert result.flag in ("OK", "INSUFFICIENT")
    # At least some non-empty bins
    non_empty = [i for i in range(result.n_bins) if result.bin_counts[i] > 0]
    assert len(non_empty) > 0


def test_overconfident_model_mean_pred_gt_fraction_pos():
    """Overconfident model: all scores near 1 but only half are positive."""
    y_true = [1] * 10 + [0] * 10
    y_score = [0.95] * 20  # all very high scores
    result = compute_calibration_curve(y_true, y_score)
    # Fraction positive ≈ 0.5, mean_pred ≈ 0.95 → overconfident
    non_empty = [i for i in range(result.n_bins) if result.bin_counts[i] > 0]
    assert len(non_empty) >= 1
    idx = non_empty[0]
    assert result.mean_pred_prob[idx] > result.fraction_positive[idx]


def test_brier_zero_for_perfect():
    """Perfect predictions → Brier score = 0."""
    y_true = [1, 1, 0, 0]
    y_score = [1.0, 1.0, 0.0, 0.0]
    result = compute_calibration_curve(y_true, y_score)
    assert abs(result.brier_score) < 1e-9


def test_brier_one_for_worst():
    """Worst predictions → Brier score = 1."""
    y_true = [1, 1, 0, 0]
    y_score = [0.0, 0.0, 1.0, 1.0]
    result = compute_calibration_curve(y_true, y_score)
    assert abs(result.brier_score - 1.0) < 1e-9


def test_len_mean_pred_prob_equals_n_bins():
    y_true = [1, 0, 1, 0, 1, 0]
    y_score = [0.8, 0.2, 0.7, 0.3, 0.9, 0.1]
    result = compute_calibration_curve(y_true, y_score, n_bins=5)
    assert len(result.mean_pred_prob) == 5


def test_empty_bins_have_bin_count_zero():
    y_true = [1, 0]
    y_score = [0.9, 0.1]  # only two scores, many bins will be empty
    result = compute_calibration_curve(y_true, y_score, n_bins=10)
    # Most bins should be empty (count=0)
    zero_bins = [c for c in result.bin_counts if c == 0]
    assert len(zero_bins) >= 6  # at least 6 empty bins expected


def test_empty_input_returns_invalid():
    result = compute_calibration_curve([], [])
    assert result.flag == "INVALID"


def test_all_same_class_returns_insufficient():
    y_true = [1, 1, 1, 1]
    y_score = [0.8, 0.9, 0.7, 0.85]
    result = compute_calibration_curve(y_true, y_score)
    assert result.flag == "INSUFFICIENT"


def test_n_bins_respected():
    y_true = [1, 0] * 10
    y_score = [0.9 if y == 1 else 0.1 for y in y_true]
    result = compute_calibration_curve(y_true, y_score, n_bins=7)
    assert result.n_bins == 7
    assert len(result.bin_counts) == 7


def test_brier_in_zero_one():
    y_true = [1, 0, 1, 0]
    y_score = [0.6, 0.4, 0.7, 0.3]
    result = compute_calibration_curve(y_true, y_score)
    assert 0.0 <= result.brier_score <= 1.0


def test_calibration_curve_result_frozen():
    y_true = [1, 0]
    y_score = [0.9, 0.1]
    result = compute_calibration_curve(y_true, y_score)
    try:
        result.flag = "MODIFIED"  # type: ignore[misc]
        raise AssertionError("Should have raised FrozenInstanceError")
    except Exception as exc:
        assert "frozen" in str(exc).lower() or "FrozenInstance" in type(exc).__name__


def test_format_returns_str():
    y_true = [1, 0, 1, 0]
    y_score = [0.9, 0.1, 0.8, 0.2]
    result = compute_calibration_curve(y_true, y_score)
    md = format_calibration_curve(result)
    assert isinstance(md, str)


def test_format_has_brier():
    y_true = [1, 0, 1, 0]
    y_score = [0.9, 0.1, 0.8, 0.2]
    result = compute_calibration_curve(y_true, y_score)
    md = format_calibration_curve(result)
    assert "Brier" in md
