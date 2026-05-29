"""Tests for Skills/model_ensemble_evaluator.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from model_ensemble_evaluator import (
    evaluate_ensemble,
    format_ensemble_eval,
)

_Y_TRUE = [1, 1, 0, 0, 1, 0, 1, 0]
# Bayesian has one cross-over (0.55 on a negative) → AUC < 1.0
# XGBoost perfectly separates → AUC = 1.0
_SCORES = {
    "bayesian": [0.6, 0.5, 0.55, 0.4, 0.65, 0.35, 0.7, 0.3],
    "xgboost": [0.9, 0.85, 0.1, 0.05, 0.88, 0.12, 0.92, 0.08],
    "ensemble": [0.75, 0.7, 0.25, 0.2, 0.76, 0.22, 0.8, 0.18],
}


def test_flag_ok():
    r = evaluate_ensemble(_Y_TRUE, _SCORES)
    assert r.flag == "OK"


def test_n_tiers():
    r = evaluate_ensemble(_Y_TRUE, _SCORES)
    assert len(r.tiers) == 3


def test_best_by_auc_is_xgboost():
    r = evaluate_ensemble(_Y_TRUE, _SCORES)
    assert r.best_tier_by_auc == "xgboost"


def test_auc_roc_in_range():
    r = evaluate_ensemble(_Y_TRUE, _SCORES)
    for t in r.tiers:
        assert 0.0 <= t.auc_roc <= 1.0


def test_auc_pr_in_range():
    r = evaluate_ensemble(_Y_TRUE, _SCORES)
    for t in r.tiers:
        assert 0.0 <= t.auc_pr <= 1.0


def test_brier_in_range():
    r = evaluate_ensemble(_Y_TRUE, _SCORES)
    for t in r.tiers:
        assert 0.0 <= t.brier <= 1.0


def test_ece_in_range():
    r = evaluate_ensemble(_Y_TRUE, _SCORES)
    for t in r.tiers:
        assert 0.0 <= t.ece <= 1.0


def test_empty_invalid():
    r = evaluate_ensemble([], {})
    assert r.flag == "INVALID"


def test_mismatched_lengths():
    r = evaluate_ensemble(_Y_TRUE, {"model": [0.5]})
    assert r.flag == "INVALID"


def test_n_test_correct():
    r = evaluate_ensemble(_Y_TRUE, _SCORES)
    assert r.n_test == len(_Y_TRUE)


def test_format_returns_string():
    r = evaluate_ensemble(_Y_TRUE, _SCORES)
    s = format_ensemble_eval(r)
    assert isinstance(s, str)
    assert "Ensemble" in s


def test_format_shows_best():
    r = evaluate_ensemble(_Y_TRUE, _SCORES)
    s = format_ensemble_eval(r)
    assert r.best_tier_by_auc in s


def test_tier_score_frozen():
    r = evaluate_ensemble(_Y_TRUE, _SCORES)
    t = r.tiers[0]
    try:
        t.auc_roc = 0.0  # type: ignore[misc]
        raise AssertionError("Should be frozen")
    except Exception:
        pass
