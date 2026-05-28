"""Tests for Skills/cnn_prediction_uncertainty.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from cnn_prediction_uncertainty import (
    UncertaintyResult,
    batch_uncertainty,
    estimate_uncertainty,
    format_uncertainty_report,
)

_SNIPPET = [1.0] * 64


def _deterministic_model(snippet: list[float]) -> float:
    return 0.8


def _noisy_model(snippet: list[float]) -> float:
    import random
    return 0.5 + random.Random().gauss(0, 0.2)


def test_flag_ok():
    r = estimate_uncertainty(100, 5.0, _SNIPPET, model_fn=_deterministic_model)
    assert r.flag == "OK"


def test_deterministic_std_zero():
    r = estimate_uncertainty(100, 5.0, _SNIPPET, model_fn=_deterministic_model, n_samples=20)
    assert r.std_score < 1e-9


def test_mean_close_to_true():
    r = estimate_uncertainty(100, 5.0, _SNIPPET, model_fn=_deterministic_model, n_samples=20)
    assert abs(r.mean_score - 0.8) < 1e-9


def test_noisy_model_positive_std():
    import random
    rng = random.Random(42)

    def noisy(s: list[float]) -> float:
        return 0.5 + rng.gauss(0, 0.2)

    r = estimate_uncertainty(100, 5.0, _SNIPPET, model_fn=noisy, n_samples=50)
    assert r.std_score > 0.0


def test_is_uncertain_flag():
    import random
    rng = random.Random(0)

    def very_noisy(s: list[float]) -> float:
        return max(0.0, min(1.0, 0.5 + rng.gauss(0, 0.3)))

    r = estimate_uncertainty(100, 5.0, _SNIPPET, model_fn=very_noisy,
                             n_samples=50, uncertainty_threshold=0.01)
    assert r.is_uncertain


def test_not_uncertain_when_deterministic():
    r = estimate_uncertainty(100, 5.0, _SNIPPET, model_fn=_deterministic_model,
                             n_samples=20, uncertainty_threshold=0.1)
    assert not r.is_uncertain


def test_empty_snippet_invalid():
    r = estimate_uncertainty(100, 5.0, [], model_fn=_deterministic_model)
    assert r.flag == "INVALID"


def test_zero_samples_invalid():
    r = estimate_uncertainty(100, 5.0, _SNIPPET, model_fn=_deterministic_model, n_samples=0)
    assert r.flag == "INVALID"


def test_model_exception_invalid():
    def bad_model(s: list[float]) -> float:
        raise RuntimeError("model offline")
    r = estimate_uncertainty(100, 5.0, _SNIPPET, model_fn=bad_model)
    assert r.flag == "INVALID"


def test_n_samples_stored():
    r = estimate_uncertainty(100, 5.0, _SNIPPET, model_fn=_deterministic_model, n_samples=15)
    assert r.n_samples == 15


def test_batch_ok():
    rows = [{"tic_id": i, "period_days": 5.0, "snippet": _SNIPPET} for i in range(3)]
    results = batch_uncertainty(rows, model_fn=_deterministic_model)
    assert all(r.flag == "OK" for r in results)


def test_batch_invalid_row():
    rows = [{"tic_id": 1, "period_days": 5.0}]  # missing snippet
    results = batch_uncertainty(rows, model_fn=_deterministic_model)
    assert results[0].flag == "INVALID"


def test_format_returns_string():
    rows = [{"tic_id": 1, "period_days": 5.0, "snippet": _SNIPPET}]
    results = batch_uncertainty(rows, model_fn=_deterministic_model)
    s = format_uncertainty_report(results)
    assert isinstance(s, str)
    assert "Uncertainty" in s
