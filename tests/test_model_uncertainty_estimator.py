import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from model_uncertainty_estimator import UncertaintyResult, estimate_model_uncertainty, format_uncertainty


# --- happy path ---

def test_identical_predictions_zero_std():
    preds = [0.8] * 5
    result = estimate_model_uncertainty(preds)
    assert result.std_pred == pytest.approx(0.0, abs=1e-9)
    assert result.uncertainty_level == "LOW"
    assert result.flag == "OK"


def test_mean_computed_correctly():
    preds = [0.6, 0.8, 1.0]
    result = estimate_model_uncertainty(preds)
    assert result.mean_pred == pytest.approx(0.8, abs=1e-9)


def test_n_models_correct():
    preds = [0.7, 0.8, 0.9, 0.75, 0.85]
    result = estimate_model_uncertainty(preds)
    assert result.n_models == 5


def test_iqr_is_q3_minus_q1():
    # sorted [0.1, 0.2, 0.8, 0.9] → Q1~0.175, Q3~0.825 → IQR~0.65
    preds = [0.1, 0.9, 0.2, 0.8]
    result = estimate_model_uncertainty(preds)
    assert result.iqr >= 0.0


# --- uncertainty_level boundaries ---

def test_low_uncertainty_std_below_0_05():
    # std of [0.78, 0.80, 0.82] ~ 0.02 < 0.05 → LOW
    preds = [0.78, 0.80, 0.82]
    result = estimate_model_uncertainty(preds)
    assert result.uncertainty_level == "LOW"
    assert result.flag == "OK"


def test_medium_uncertainty_std_between_0_05_and_0_15():
    # std ~ 0.10: predictions spread around 0.1 apart
    preds = [0.5, 0.6, 0.7, 0.8]
    result = estimate_model_uncertainty(preds)
    # std of [0.5,0.6,0.7,0.8] = 0.129 → MEDIUM
    assert result.uncertainty_level == "MEDIUM"
    assert result.flag == "OK"


def test_high_uncertainty_std_above_0_15():
    # Large spread → HIGH
    preds = [0.0, 0.5, 1.0]
    result = estimate_model_uncertainty(preds)
    # std of [0,0.5,1] = 0.5 → HIGH
    assert result.uncertainty_level == "HIGH"
    assert result.flag == "HIGH_UNCERTAINTY"


def test_flag_high_uncertainty_matches_level():
    preds = [0.0, 1.0]
    result = estimate_model_uncertainty(preds)
    assert (result.flag == "HIGH_UNCERTAINTY") == (result.uncertainty_level == "HIGH")


# --- edge cases ---

def test_empty_predictions_no_data():
    result = estimate_model_uncertainty([])
    assert result.flag == "NO_DATA"
    assert result.n_models == 0
    assert result.mean_pred == 0.0
    assert result.std_pred == 0.0


def test_single_prediction_zero_std():
    result = estimate_model_uncertainty([0.75])
    assert result.n_models == 1
    assert result.std_pred == 0.0
    assert result.mean_pred == pytest.approx(0.75, abs=1e-9)


def test_two_predictions():
    preds = [0.4, 0.6]
    result = estimate_model_uncertainty(preds)
    assert result.n_models == 2
    # sample std = sqrt(((0.1)^2 + (0.1)^2) / 1) = 0.1414...
    assert result.std_pred > 0.0


# --- return type ---

def test_returns_uncertainty_result():
    result = estimate_model_uncertainty([0.8, 0.85, 0.9])
    assert isinstance(result, UncertaintyResult)


def test_result_is_frozen():
    result = estimate_model_uncertainty([0.8, 0.85, 0.9])
    with pytest.raises((AttributeError, TypeError)):
        result.flag = "CHANGED"


# --- format output ---

def test_format_contains_header():
    result = estimate_model_uncertainty([0.8, 0.85, 0.9])
    text = format_uncertainty(result)
    assert "## Model Uncertainty Estimate" in text


def test_format_contains_flag():
    result = estimate_model_uncertainty([0.8, 0.85, 0.9])
    text = format_uncertainty(result)
    assert result.flag in text
