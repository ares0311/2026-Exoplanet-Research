"""Tests for src/exo_toolkit/ml/stacking_scorer.py."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from exo_toolkit.ml.stacking_scorer import StackingScorer
from exo_toolkit.ml.xgboost_scorer import FEATURE_NAMES, XGBoostScorer
from exo_toolkit.schemas import CandidateFeatures

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _all_zero_features() -> CandidateFeatures:
    return CandidateFeatures(**dict.fromkeys(FEATURE_NAMES, 0.0))


def _all_one_features() -> CandidateFeatures:
    return CandidateFeatures(**dict.fromkeys(FEATURE_NAMES, 1.0))


def _mock_xgb(return_value: float = 0.8) -> MagicMock:
    scorer = MagicMock(spec=XGBoostScorer)
    scorer.predict_proba.return_value = return_value
    scorer.predict_proba_batch.return_value = np.full(1, return_value, dtype=np.float64)
    return scorer


def _mock_xgb_batch(values: list[float]) -> MagicMock:
    scorer = MagicMock(spec=XGBoostScorer)
    scorer.predict_proba_batch.return_value = np.array(values, dtype=np.float64)
    return scorer


def _make_training_data(n: int = 40, seed: int = 0) -> tuple[list[CandidateFeatures], list[int]]:
    rng = np.random.default_rng(seed)
    features_list = []
    labels = []
    for i in range(n):
        label = i % 2
        vals = {name: float(rng.uniform(0.3 if label else 0.7, 0.7 if label else 1.0))
                for name in FEATURE_NAMES}
        features_list.append(CandidateFeatures(**vals))
        labels.append(label)
    return features_list, labels


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestStackingScorerInit:
    def test_default_weight_is_half(self) -> None:
        s = StackingScorer()
        assert s.xgb_weight == pytest.approx(0.5)

    def test_custom_weight(self) -> None:
        s = StackingScorer(xgb_weight=0.7)
        assert s.xgb_weight == pytest.approx(0.7)

    def test_invalid_weight_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="xgb_weight"):
            StackingScorer(xgb_weight=1.1)

    def test_invalid_weight_below_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="xgb_weight"):
            StackingScorer(xgb_weight=-0.1)

    def test_boundary_weights_ok(self) -> None:
        StackingScorer(xgb_weight=0.0)
        StackingScorer(xgb_weight=1.0)

    def test_has_xgb_false_without_scorer(self) -> None:
        assert StackingScorer().has_xgb is False

    def test_has_xgb_true_with_scorer(self) -> None:
        assert StackingScorer(xgb_scorer=_mock_xgb()).has_xgb is True


# ---------------------------------------------------------------------------
# predict_proba
# ---------------------------------------------------------------------------


class TestStackingScorerPredict:
    def test_bayesian_fallback_when_no_xgb(self) -> None:
        s = StackingScorer()
        p = s.predict_proba(_all_zero_features(), bayesian_planet_prob=0.3)
        assert p == pytest.approx(0.3)

    def test_blends_correctly_equal_weight(self) -> None:
        xgb = _mock_xgb(return_value=0.8)
        s = StackingScorer(xgb_scorer=xgb, xgb_weight=0.5)
        p = s.predict_proba(_all_zero_features(), bayesian_planet_prob=0.4)
        assert p == pytest.approx(0.5 * 0.8 + 0.5 * 0.4)

    def test_weight_zero_returns_bayesian(self) -> None:
        xgb = _mock_xgb(return_value=0.9)
        s = StackingScorer(xgb_scorer=xgb, xgb_weight=0.0)
        p = s.predict_proba(_all_zero_features(), bayesian_planet_prob=0.25)
        assert p == pytest.approx(0.25)

    def test_weight_one_returns_xgb(self) -> None:
        xgb = _mock_xgb(return_value=0.9)
        s = StackingScorer(xgb_scorer=xgb, xgb_weight=1.0)
        p = s.predict_proba(_all_zero_features(), bayesian_planet_prob=0.1)
        assert p == pytest.approx(0.9)

    def test_result_in_range(self) -> None:
        xgb = _mock_xgb(return_value=0.6)
        s = StackingScorer(xgb_scorer=xgb, xgb_weight=0.4)
        p = s.predict_proba(_all_one_features(), bayesian_planet_prob=0.7)
        assert 0.0 <= p <= 1.0

    def test_returns_float(self) -> None:
        s = StackingScorer()
        p = s.predict_proba(_all_zero_features(), bayesian_planet_prob=0.5)
        assert isinstance(p, float)


# ---------------------------------------------------------------------------
# predict_proba_batch
# ---------------------------------------------------------------------------


class TestStackingScorerBatch:
    def test_batch_bayesian_fallback_shape(self) -> None:
        s = StackingScorer()
        probs = np.array([0.2, 0.5, 0.8])
        result = s.predict_proba_batch(
            [_all_zero_features()] * 3, bayesian_planet_probs=probs
        )
        assert result.shape == (3,)

    def test_batch_bayesian_fallback_values(self) -> None:
        s = StackingScorer()
        probs = np.array([0.1, 0.9])
        result = s.predict_proba_batch(
            [_all_zero_features()] * 2, bayesian_planet_probs=probs
        )
        np.testing.assert_allclose(result, probs)

    def test_batch_blends_correctly(self) -> None:
        xgb_vals = [0.8, 0.6]
        xgb = _mock_xgb_batch(xgb_vals)
        s = StackingScorer(xgb_scorer=xgb, xgb_weight=0.5)
        bayes = np.array([0.4, 0.2])
        result = s.predict_proba_batch(
            [_all_zero_features()] * 2, bayesian_planet_probs=bayes
        )
        expected = np.array([0.5 * 0.8 + 0.5 * 0.4, 0.5 * 0.6 + 0.5 * 0.2])
        np.testing.assert_allclose(result, expected)

    def test_batch_consistent_with_single(self) -> None:
        xgb = _mock_xgb(return_value=0.7)
        xgb.predict_proba_batch.return_value = np.array([0.7], dtype=np.float64)
        s = StackingScorer(xgb_scorer=xgb, xgb_weight=0.5)
        f = _all_zero_features()
        p_single = s.predict_proba(f, bayesian_planet_prob=0.3)
        p_batch = s.predict_proba_batch([f], bayesian_planet_probs=np.array([0.3]))
        assert abs(p_single - p_batch[0]) < 1e-9

    def test_batch_dtype_float64(self) -> None:
        s = StackingScorer()
        result = s.predict_proba_batch(
            [_all_zero_features()], bayesian_planet_probs=np.array([0.5])
        )
        assert result.dtype == np.float64


# ---------------------------------------------------------------------------
# Factory methods
# ---------------------------------------------------------------------------


class TestStackingScorerFactory:
    def test_bayesian_only_has_no_xgb(self) -> None:
        s = StackingScorer.bayesian_only()
        assert s.has_xgb is False

    def test_bayesian_only_passthrough(self) -> None:
        s = StackingScorer.bayesian_only()
        assert s.predict_proba(_all_zero_features(), 0.42) == pytest.approx(0.42)

    def test_from_model_path_loads_and_predicts(self, tmp_path: Path) -> None:
        # Train a small real model and save it.
        features_list, labels = _make_training_data(n=40, seed=3)
        xgb_scorer = XGBoostScorer()
        xgb_scorer.fit(features_list, labels)
        meta_path = tmp_path / "model.json"
        xgb_scorer.save(meta_path)

        stacker = StackingScorer.from_model_path(meta_path, xgb_weight=0.6)
        assert stacker.has_xgb is True
        assert stacker.xgb_weight == pytest.approx(0.6)

        p = stacker.predict_proba(_all_one_features(), bayesian_planet_prob=0.5)
        assert 0.0 <= p <= 1.0

    def test_from_model_path_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            StackingScorer.from_model_path(tmp_path / "nonexistent.json")
