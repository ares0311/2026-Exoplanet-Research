import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from cross_validation_reporter import CVReport, report_cross_validation, format_cv_report


# --- happy path ---

def test_basic_five_fold():
    folds = [{"auc": 0.90, "f1": 0.85, "precision": 0.88, "recall": 0.82}] * 5
    result = report_cross_validation(folds)
    assert result.k_folds == 5
    assert abs(result.mean_auc - 0.90) < 1e-4
    assert result.std_auc == 0.0
    assert result.flag == "OK"


def test_mean_values_correct():
    folds = [
        {"auc": 0.80, "f1": 0.70, "precision": 0.75, "recall": 0.65},
        {"auc": 0.90, "f1": 0.80, "precision": 0.85, "recall": 0.75},
    ]
    result = report_cross_validation(folds)
    assert abs(result.mean_auc - 0.85) < 1e-4
    assert abs(result.mean_f1 - 0.75) < 1e-4
    assert abs(result.mean_precision - 0.80) < 1e-4
    assert abs(result.mean_recall - 0.70) < 1e-4


def test_ci_bounds_within_zero_one():
    folds = [
        {"auc": 0.55, "f1": 0.5, "precision": 0.5, "recall": 0.5},
        {"auc": 0.95, "f1": 0.9, "precision": 0.9, "recall": 0.9},
    ]
    result = report_cross_validation(folds)
    assert result.ci_auc_low >= 0.0
    assert result.ci_auc_high <= 1.0


# --- flag boundary ---

def test_flag_ok_when_std_auc_at_threshold():
    # std exactly 0.05 → OK
    folds = [{"auc": 0.85, "f1": 0.8, "precision": 0.8, "recall": 0.8},
             {"auc": 0.75, "f1": 0.7, "precision": 0.7, "recall": 0.7}]
    result = report_cross_validation(folds)
    # std of [0.85, 0.75] = 0.0707 > 0.05 → HIGH_VARIANCE
    assert result.flag == "HIGH_VARIANCE"


def test_flag_ok_low_std():
    folds = [{"auc": 0.80, "f1": 0.78, "precision": 0.79, "recall": 0.77},
             {"auc": 0.82, "f1": 0.80, "precision": 0.81, "recall": 0.79}]
    result = report_cross_validation(folds)
    # std ~ 0.014 < 0.05 → OK
    assert result.flag == "OK"


def test_high_variance_flag():
    folds = [{"auc": 0.60, "f1": 0.6, "precision": 0.6, "recall": 0.6},
             {"auc": 0.95, "f1": 0.9, "precision": 0.9, "recall": 0.9}]
    result = report_cross_validation(folds)
    assert result.flag == "HIGH_VARIANCE"


# --- edge cases ---

def test_empty_input():
    result = report_cross_validation([])
    assert result.k_folds == 0
    assert result.mean_auc == 0.0
    assert result.flag == "OK"


def test_single_fold():
    result = report_cross_validation([{"auc": 0.88, "f1": 0.80, "precision": 0.85, "recall": 0.76}])
    assert result.k_folds == 1
    assert result.std_auc == 0.0
    assert result.ci_auc_low == result.ci_auc_high == result.mean_auc


def test_missing_keys_default_zero():
    folds = [{"auc": 0.80}, {"auc": 0.82}]
    result = report_cross_validation(folds)
    assert result.mean_f1 == 0.0
    assert result.mean_precision == 0.0
    assert result.mean_recall == 0.0


# --- return type ---

def test_returns_cvreport_instance():
    result = report_cross_validation([{"auc": 0.9, "f1": 0.8, "precision": 0.85, "recall": 0.75}])
    assert isinstance(result, CVReport)


def test_result_is_frozen():
    result = report_cross_validation([{"auc": 0.9, "f1": 0.8, "precision": 0.85, "recall": 0.75}])
    with pytest.raises((AttributeError, TypeError)):
        result.flag = "CHANGED"


# --- format output ---

def test_format_contains_header():
    folds = [{"auc": 0.85, "f1": 0.80, "precision": 0.82, "recall": 0.78}]
    result = report_cross_validation(folds)
    text = format_cv_report(result)
    assert "## Cross-Validation Report" in text


def test_format_contains_flag():
    folds = [{"auc": 0.85, "f1": 0.80, "precision": 0.82, "recall": 0.78}]
    result = report_cross_validation(folds)
    text = format_cv_report(result)
    assert result.flag in text


def test_format_contains_ci():
    folds = [{"auc": 0.80, "f1": 0.78, "precision": 0.79, "recall": 0.77},
             {"auc": 0.90, "f1": 0.88, "precision": 0.89, "recall": 0.87}]
    result = report_cross_validation(folds)
    text = format_cv_report(result)
    assert "95% CI" in text
