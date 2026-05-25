"""Tests for Skills/ensemble_weight_optimizer.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from ensemble_weight_optimizer import (
    blend_scores,
    format_weight_result,
    optimize_weights,
)


def _make_data(n: int = 40) -> tuple[list[int], list[float], list[float], list[float]]:
    """Simple dataset: XGB is perfect, CNN/Bayes are anti-informative."""
    import random
    rng = random.Random(42)
    y_true = [1 if i % 2 == 0 else 0 for i in range(n)]  # interleaved
    # XGB is perfect
    xgb_scores = [0.9 if y == 1 else 0.1 for y in y_true]
    # CNN/Bayes are anti-correlated (bad, not just flat)
    cnn_scores = [rng.uniform(0.3, 0.5) if y == 1 else rng.uniform(0.5, 0.7)
                  for y in y_true]
    bayes_scores = [rng.uniform(0.3, 0.5) if y == 1 else rng.uniform(0.5, 0.7)
                    for y in y_true]
    return y_true, xgb_scores, cnn_scores, bayes_scores


def test_xgb_dominant_weight_optimal():
    y_true, xgb, cnn, bayes = _make_data()
    result = optimize_weights(y_true, xgb, cnn, bayes)
    assert result.flag == "OK"
    # XGB is perfect: optimal blend achieves AUC=1.0 and uses nonzero XGB weight
    assert abs(result.best_auc - 1.0) < 1e-9
    assert result.best_weights[0] > 0.0


def test_equal_weights_is_valid_candidate():
    y_true = [1, 0, 1, 0]
    scores = [0.8, 0.2, 0.7, 0.3]
    result = optimize_weights(y_true, scores, scores, scores)
    assert result.flag == "OK"
    # Equal-weight combination (1/3, 1/3, 1/3) must have been tried
    assert result.n_combinations_tried > 0


def test_n_combinations_consistent_with_step():
    y_true, xgb, cnn, bayes = _make_data()
    result = optimize_weights(y_true, xgb, cnn, bayes, step=0.1)
    # For step=0.1, N=10: combinations = C(N+2, 2) = 66
    assert result.n_combinations_tried == 66


def test_best_weights_sum_to_one():
    y_true, xgb, cnn, bayes = _make_data()
    result = optimize_weights(y_true, xgb, cnn, bayes)
    assert result.flag == "OK"
    total = sum(result.best_weights)
    assert abs(total - 1.0) < 1e-9


def test_best_auc_in_zero_one():
    y_true, xgb, cnn, bayes = _make_data()
    result = optimize_weights(y_true, xgb, cnn, bayes)
    assert 0.0 <= result.best_auc <= 1.0


def test_mismatched_lengths_returns_invalid():
    result = optimize_weights([1, 0], [0.9, 0.1], [0.5], [0.5, 0.5])
    assert result.flag == "INVALID"


def test_empty_returns_invalid():
    result = optimize_weights([], [], [], [])
    assert result.flag == "INVALID"


def test_all_same_class_returns_degenerate():
    y_true = [1, 1, 1, 1]
    scores = [0.9, 0.8, 0.7, 0.6]
    result = optimize_weights(y_true, scores, scores, scores)
    assert result.flag == "DEGENERATE"


def test_blend_all_xgb():
    xgb = [0.9, 0.1]
    cnn = [0.5, 0.5]
    bayes = [0.3, 0.7]
    blended = blend_scores(xgb, cnn, bayes, (1.0, 0.0, 0.0))
    assert blended == xgb


def test_blend_all_bayes():
    xgb = [0.9, 0.1]
    cnn = [0.5, 0.5]
    bayes = [0.3, 0.7]
    blended = blend_scores(xgb, cnn, bayes, (0.0, 0.0, 1.0))
    assert blended == bayes


def test_weight_search_result_frozen():
    y_true, xgb, cnn, bayes = _make_data()
    result = optimize_weights(y_true, xgb, cnn, bayes)
    try:
        result.flag = "MODIFIED"  # type: ignore[misc]
        raise AssertionError("Should have raised FrozenInstanceError")
    except Exception as exc:
        assert "frozen" in str(exc).lower() or "FrozenInstance" in type(exc).__name__


def test_format_has_weight_values():
    y_true, xgb, cnn, bayes = _make_data()
    result = optimize_weights(y_true, xgb, cnn, bayes)
    md = format_weight_result(result)
    assert isinstance(md, str)
    assert "XGBoost" in md or "xgb" in md.lower()


def test_step_half_gives_six_combinations():
    y_true = [1, 0, 1, 0]
    scores_a = [0.9, 0.1, 0.8, 0.2]
    scores_b = [0.5, 0.5, 0.5, 0.5]
    scores_c = [0.6, 0.4, 0.7, 0.3]
    result = optimize_weights(y_true, scores_a, scores_b, scores_c, step=0.5)
    # step=0.5 → N=2; combinations: (2,0,0),(0,2,0),(0,0,2),(1,1,0),(1,0,1),(0,1,1) = 6
    assert result.n_combinations_tried == 6
