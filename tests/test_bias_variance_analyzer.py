import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from bias_variance_analyzer import BiasVarianceResult, analyze_bias_variance, format_bias_variance


def test_basic_low_variance():
    preds = [0.82, 0.83, 0.81, 0.84, 0.82]
    result = analyze_bias_variance(true_label=1, fold_predictions=preds)
    assert isinstance(result, BiasVarianceResult)
    assert result.flag == "OK"


def test_mean_pred_correct():
    preds = [0.6, 0.8]
    result = analyze_bias_variance(true_label=1, fold_predictions=preds)
    assert abs(result.mean_pred - 0.7) < 1e-9


def test_bias_sq_computed_correctly():
    preds = [0.6, 0.6, 0.6]
    result = analyze_bias_variance(true_label=1, fold_predictions=preds)
    # bias = 1.0 - 0.6 = 0.4; bias_sq = 0.16
    assert abs(result.bias_sq - 0.16) < 1e-9


def test_variance_computed_correctly():
    preds = [0.5, 0.7]
    result = analyze_bias_variance(true_label=1, fold_predictions=preds)
    # sample variance of [0.5, 0.7] = 0.02
    assert abs(result.variance - 0.02) < 1e-9


def test_flag_high_variance_above_threshold():
    preds = [0.0, 0.8, 0.0, 0.8, 0.0, 0.8]
    result = analyze_bias_variance(true_label=1, fold_predictions=preds)
    assert result.flag == "HIGH_VARIANCE"


def test_flag_ok_variance_below_threshold():
    preds = [0.75, 0.76, 0.74, 0.75, 0.76]
    result = analyze_bias_variance(true_label=1, fold_predictions=preds)
    assert result.flag == "OK"


def test_true_label_zero_bias_sq_non_negative():
    preds = [0.1, 0.2, 0.15]
    result = analyze_bias_variance(true_label=0, fold_predictions=preds)
    assert result.bias_sq >= 0.0


def test_single_prediction_zero_variance():
    result = analyze_bias_variance(true_label=1, fold_predictions=[0.7])
    assert result.variance == 0.0
    assert result.flag == "OK"


def test_empty_predictions():
    result = analyze_bias_variance(true_label=1, fold_predictions=[])
    assert result.mean_pred == 0.0
    assert result.variance == 0.0


def test_perfect_predictions():
    preds = [1.0, 1.0, 1.0]
    result = analyze_bias_variance(true_label=1, fold_predictions=preds)
    assert result.bias_sq == 0.0
    assert result.variance == 0.0
    assert result.flag == "OK"


def test_total_error_equals_bias_sq_plus_variance():
    preds = [0.6, 0.8, 0.7]
    result = analyze_bias_variance(true_label=1, fold_predictions=preds)
    assert abs(result.total_error - (result.bias_sq + result.variance)) < 1e-9


def test_result_is_frozen():
    result = analyze_bias_variance(true_label=1, fold_predictions=[0.8])
    with pytest.raises((AttributeError, TypeError)):
        result.flag = "CHANGED"


def test_format_contains_header():
    result = analyze_bias_variance(true_label=1, fold_predictions=[0.8, 0.85])
    text = format_bias_variance(result)
    assert "##" in text


def test_format_contains_flag():
    result = analyze_bias_variance(true_label=1, fold_predictions=[0.8, 0.85])
    text = format_bias_variance(result)
    assert result.flag in text
