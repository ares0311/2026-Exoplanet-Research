import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from learning_curve_analyzer import (
    LearningCurveResult,
    analyze_learning_curve,
    format_learning_curve,
)

# --- happy path ---

def test_converged_good_model():
    # val_score > 0.7, gap < 0.1, last improvement < 0.01
    points = [
        {"n_samples": 100, "train_score": 0.85, "val_score": 0.80},
        {"n_samples": 200, "train_score": 0.86, "val_score": 0.81},
        {"n_samples": 400, "train_score": 0.86, "val_score": 0.815},
    ]
    result = analyze_learning_curve(points)
    assert result.flag == "OK"
    assert result.converged is True


def test_underfitting_low_val_score():
    points = [
        {"n_samples": 100, "train_score": 0.65, "val_score": 0.60},
        {"n_samples": 200, "train_score": 0.65, "val_score": 0.62},
        {"n_samples": 400, "train_score": 0.66, "val_score": 0.63},
    ]
    result = analyze_learning_curve(points)
    assert result.flag == "UNDERFITTING"


def test_overfitting_large_gap():
    points = [
        {"n_samples": 100, "train_score": 0.98, "val_score": 0.78},
        {"n_samples": 200, "train_score": 0.97, "val_score": 0.80},
        {"n_samples": 400, "train_score": 0.97, "val_score": 0.82},
    ]
    result = analyze_learning_curve(points)
    assert result.flag == "OVERFITTING"


def test_final_val_and_train_scores_from_last_point():
    points = [
        {"n_samples": 100, "train_score": 0.80, "val_score": 0.75},
        {"n_samples": 200, "train_score": 0.85, "val_score": 0.80},
    ]
    result = analyze_learning_curve(points)
    assert abs(result.final_val_score - 0.80) < 1e-9
    assert abs(result.final_train_score - 0.85) < 1e-9


# --- converged flag ---

def test_not_converged_when_large_improvement():
    points = [
        {"n_samples": 100, "train_score": 0.80, "val_score": 0.70},
        {"n_samples": 200, "train_score": 0.88, "val_score": 0.80},
    ]
    result = analyze_learning_curve(points)
    assert result.converged is False


def test_converged_when_tiny_improvement():
    points = [
        {"n_samples": 100, "train_score": 0.85, "val_score": 0.82},
        {"n_samples": 200, "train_score": 0.855, "val_score": 0.8205},
    ]
    result = analyze_learning_curve(points)
    assert result.converged is True


# --- edge cases ---

def test_empty_points():
    result = analyze_learning_curve([])
    assert isinstance(result, LearningCurveResult)


def test_single_point():
    result = analyze_learning_curve([{"n_samples": 100, "train_score": 0.80, "val_score": 0.75}])
    assert isinstance(result, LearningCurveResult)


def test_exactly_at_underfitting_boundary():
    # final val exactly 0.7 → NOT underfitting (< 0.7 threshold)
    points = [
        {"n_samples": 100, "train_score": 0.72, "val_score": 0.70},
        {"n_samples": 200, "train_score": 0.72, "val_score": 0.70},
    ]
    result = analyze_learning_curve(points)
    assert result.flag != "UNDERFITTING"


# --- return type ---

def test_returns_learning_curve_result():
    result = analyze_learning_curve([{"n_samples": 100, "train_score": 0.85, "val_score": 0.80}])
    assert isinstance(result, LearningCurveResult)


def test_result_is_frozen():
    result = analyze_learning_curve([{"n_samples": 100, "train_score": 0.85, "val_score": 0.80}])
    with pytest.raises((AttributeError, TypeError)):
        result.flag = "CHANGED"


# --- format output ---

def test_format_contains_header():
    result = analyze_learning_curve([{"n_samples": 100, "train_score": 0.85, "val_score": 0.80}])
    text = format_learning_curve(result)
    assert "##" in text


def test_format_contains_flag():
    result = analyze_learning_curve([{"n_samples": 100, "train_score": 0.85, "val_score": 0.80}])
    text = format_learning_curve(result)
    assert result.flag in text
