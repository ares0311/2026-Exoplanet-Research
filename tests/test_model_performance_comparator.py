"""Tests for Skills/model_performance_comparator.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from model_performance_comparator import (
    compare_models,
    format_comparison,
)

_TWO_MODELS = [
    {"model_id": "xgb", "auc": 0.92, "f1": 0.88, "brier": 0.08, "n_train": 1000},
    {"model_id": "bayes", "auc": 0.85, "f1": 0.80, "brier": 0.12, "n_train": 1000},
]


def test_two_models_n_models_equals_2():
    result = compare_models(_TWO_MODELS)
    assert result.n_models == 2


def test_best_by_auc_is_highest():
    result = compare_models(_TWO_MODELS)
    assert result.best_by_auc == "xgb"


def test_best_by_brier_is_lowest():
    result = compare_models(_TWO_MODELS)
    assert result.best_by_brier == "xgb"


def test_none_auc_excluded_from_best():
    models = [
        {"model_id": "a", "auc": None},
        {"model_id": "b", "auc": 0.70},
    ]
    result = compare_models(models)
    assert result.best_by_auc == "b"


def test_empty_list_returns_empty():
    result = compare_models([])
    assert result.flag == "EMPTY"
    assert result.n_models == 0


def test_single_model_ok_and_best_is_itself():
    result = compare_models([{"model_id": "solo", "auc": 0.88}])
    assert result.flag == "OK"
    assert result.best_by_auc == "solo"


def test_format_returns_str_with_pipe():
    result = compare_models(_TWO_MODELS)
    md = format_comparison(result)
    assert isinstance(md, str)
    assert "|" in md


def test_format_has_all_model_ids():
    result = compare_models(_TWO_MODELS)
    md = format_comparison(result)
    assert "xgb" in md
    assert "bayes" in md


def test_comparison_result_frozen():
    result = compare_models(_TWO_MODELS)
    try:
        result.flag = "MODIFIED"  # type: ignore[misc]
        raise AssertionError("Should have raised FrozenInstanceError")
    except Exception as exc:
        assert "frozen" in str(exc).lower() or "FrozenInstance" in type(exc).__name__


def test_missing_optional_fields_dont_crash():
    result = compare_models([{"model_id": "minimal"}])
    assert result.flag == "OK"


def test_notes_preserved_in_model_metrics():
    models = [{"model_id": "x", "notes": "experimental run"}]
    result = compare_models(models)
    assert result.models[0].notes == "experimental run"


def test_n_models_matches_input():
    models = [{"model_id": f"m{i}"} for i in range(5)]
    result = compare_models(models)
    assert result.n_models == 5


def test_flag_ok_on_valid():
    result = compare_models(_TWO_MODELS)
    assert result.flag == "OK"
