"""Tests for src/exo_toolkit/ml/xgboost_scorer.py."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from exo_toolkit.ml.xgboost_scorer import (
    FEATURE_NAMES,
    TrainingResult,
    XGBoostScorer,
    features_list_to_matrix,
    features_to_array,
)
from exo_toolkit.schemas import CandidateFeatures

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _all_none_features() -> CandidateFeatures:
    return CandidateFeatures()


def _all_zero_features() -> CandidateFeatures:
    return CandidateFeatures(**dict.fromkeys(FEATURE_NAMES, 0.0))


def _all_one_features() -> CandidateFeatures:
    return CandidateFeatures(**dict.fromkeys(FEATURE_NAMES, 1.0))


def _partial_features(n_set: int = 10) -> CandidateFeatures:
    kwargs = {name: float(i % 2) for i, name in enumerate(FEATURE_NAMES[:n_set])}
    return CandidateFeatures(**kwargs)


def _make_training_data(
    n: int = 40, seed: int = 0
) -> tuple[list[CandidateFeatures], list[int]]:
    """Generate synthetic labelled training data."""
    rng = np.random.default_rng(seed)
    features_list = []
    labels = []
    for i in range(n):
        label = i % 2
        vals = {}
        for _j, name in enumerate(FEATURE_NAMES):
            if rng.random() < 0.2:
                vals[name] = None
            else:
                base = 0.7 if label == 1 else 0.3
                vals[name] = float(np.clip(rng.normal(base, 0.15), 0.0, 1.0))
        features_list.append(CandidateFeatures(**vals))
        labels.append(label)
    return features_list, labels


# ---------------------------------------------------------------------------
# FEATURE_NAMES
# ---------------------------------------------------------------------------


class TestFeatureNames:
    def test_length(self) -> None:
        assert len(FEATURE_NAMES) == 35

    def test_all_in_schema(self) -> None:
        schema_fields = set(CandidateFeatures.model_fields.keys())
        for name in FEATURE_NAMES:
            assert name in schema_fields, f"{name} not in CandidateFeatures"

    def test_no_duplicates(self) -> None:
        assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))

    def test_all_end_with_score(self) -> None:
        for name in FEATURE_NAMES:
            assert name.endswith("_score"), f"{name} does not end with '_score'"


# ---------------------------------------------------------------------------
# features_to_array
# ---------------------------------------------------------------------------


class TestFeaturesToArray:
    def test_shape(self) -> None:
        arr = features_to_array(_all_none_features())
        assert arr.shape == (35,)

    def test_dtype(self) -> None:
        arr = features_to_array(_all_none_features())
        assert arr.dtype == np.float64

    def test_none_becomes_nan(self) -> None:
        arr = features_to_array(_all_none_features())
        assert np.all(np.isnan(arr))

    def test_zero_features_all_zero(self) -> None:
        arr = features_to_array(_all_zero_features())
        assert np.all(arr == 0.0)

    def test_one_features_all_one(self) -> None:
        arr = features_to_array(_all_one_features())
        assert np.all(arr == 1.0)

    def test_partial_none_mixed(self) -> None:
        f = _partial_features(n_set=10)
        arr = features_to_array(f)
        assert np.sum(np.isnan(arr)) == 25  # 35 - 10 set

    def test_column_order_matches_feature_names(self) -> None:
        f = CandidateFeatures(snr_score=0.42)
        arr = features_to_array(f)
        idx = FEATURE_NAMES.index("snr_score")
        assert arr[idx] == pytest.approx(0.42)
        non_snr = [i for i in range(35) if i != idx]
        assert np.all(np.isnan(arr[non_snr]))


# ---------------------------------------------------------------------------
# features_list_to_matrix
# ---------------------------------------------------------------------------


class TestFeaturesListToMatrix:
    def test_shape(self) -> None:
        flist = [_all_none_features(), _all_zero_features(), _all_one_features()]
        mat = features_list_to_matrix(flist)
        assert mat.shape == (3, 35)

    def test_single_row(self) -> None:
        mat = features_list_to_matrix([_all_one_features()])
        assert mat.shape == (1, 35)
        assert np.all(mat == 1.0)

    def test_rows_match_individual(self) -> None:
        f1 = _all_zero_features()
        f2 = _all_one_features()
        mat = features_list_to_matrix([f1, f2])
        np.testing.assert_array_equal(mat[0], features_to_array(f1))
        np.testing.assert_array_equal(mat[1], features_to_array(f2))


# ---------------------------------------------------------------------------
# XGBoostScorer — untrained state
# ---------------------------------------------------------------------------


class TestXGBoostScorerUntrained:
    def test_is_trained_false(self) -> None:
        scorer = XGBoostScorer()
        assert scorer.is_trained is False

    def test_training_result_none(self) -> None:
        assert XGBoostScorer().training_result is None

    def test_feature_names(self) -> None:
        assert XGBoostScorer().feature_names == FEATURE_NAMES

    def test_predict_proba_raises(self) -> None:
        with pytest.raises(RuntimeError, match="not trained"):
            XGBoostScorer().predict_proba(_all_none_features())

    def test_predict_proba_batch_raises(self) -> None:
        with pytest.raises(RuntimeError, match="not trained"):
            XGBoostScorer().predict_proba_batch([_all_none_features()])

    def test_save_raises(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="not trained"):
            XGBoostScorer().save(tmp_path / "model.json")


# ---------------------------------------------------------------------------
# XGBoostScorer.fit
# ---------------------------------------------------------------------------


class TestXGBoostScorerFit:
    def test_fit_returns_self(self) -> None:
        scorer = XGBoostScorer()
        features_list, labels = _make_training_data()
        result = scorer.fit(features_list, labels)
        assert result is scorer

    def test_is_trained_after_fit(self) -> None:
        scorer = XGBoostScorer()
        features_list, labels = _make_training_data()
        scorer.fit(features_list, labels)
        assert scorer.is_trained is True

    def test_training_result_populated(self) -> None:
        scorer = XGBoostScorer()
        features_list, labels = _make_training_data(n=40)
        scorer.fit(features_list, labels)
        tr = scorer.training_result
        assert tr is not None
        assert tr.n_samples == 40
        assert tr.n_positive == 20
        assert tr.n_negative == 20
        assert tr.n_features == 35

    def test_feature_importance_all_names(self) -> None:
        scorer = XGBoostScorer()
        features_list, labels = _make_training_data()
        scorer.fit(features_list, labels)
        tr = scorer.training_result
        assert tr is not None
        assert set(tr.feature_importance.keys()) == set(FEATURE_NAMES)

    def test_feature_importance_sum_to_one(self) -> None:
        scorer = XGBoostScorer()
        features_list, labels = _make_training_data()
        scorer.fit(features_list, labels)
        tr = scorer.training_result
        assert tr is not None
        total = sum(tr.feature_importance.values())
        assert abs(total - 1.0) < 0.01

    def test_mismatched_lengths_raises(self) -> None:
        scorer = XGBoostScorer()
        features_list, labels = _make_training_data(n=10)
        with pytest.raises(ValueError, match="length"):
            scorer.fit(features_list, labels[:5])

    def test_invalid_label_raises(self) -> None:
        scorer = XGBoostScorer()
        features_list, labels = _make_training_data(n=10)
        labels[0] = 2
        with pytest.raises(ValueError, match="0 and 1"):
            scorer.fit(features_list, labels)

    def test_empty_raises(self) -> None:
        scorer = XGBoostScorer()
        with pytest.raises(ValueError, match="empty"):
            scorer.fit([], [])

    def test_fit_with_nan_features(self) -> None:
        scorer = XGBoostScorer()
        features_list = [_all_none_features()] * 20 + [_all_one_features()] * 20
        labels = [0] * 20 + [1] * 20
        scorer.fit(features_list, labels)
        assert scorer.is_trained


# ---------------------------------------------------------------------------
# XGBoostScorer.predict_proba
# ---------------------------------------------------------------------------


class TestXGBoostScorerPredict:
    def _trained_scorer(self) -> XGBoostScorer:
        scorer = XGBoostScorer()
        features_list, labels = _make_training_data(n=60, seed=1)
        scorer.fit(features_list, labels)
        return scorer

    def test_returns_float(self) -> None:
        scorer = self._trained_scorer()
        p = scorer.predict_proba(_all_zero_features())
        assert isinstance(p, float)

    def test_probability_in_range(self) -> None:
        scorer = self._trained_scorer()
        p = scorer.predict_proba(_all_zero_features())
        assert 0.0 <= p <= 1.0

    def test_high_planet_features_higher_prob(self) -> None:
        scorer = self._trained_scorer()
        p_planet = scorer.predict_proba(_all_one_features())
        p_fp = scorer.predict_proba(_all_zero_features())
        assert p_planet > p_fp

    def test_batch_length_matches(self) -> None:
        scorer = self._trained_scorer()
        flist = [_all_zero_features(), _all_one_features(), _partial_features()]
        probs = scorer.predict_proba_batch(flist)
        assert probs.shape == (3,)

    def test_batch_probabilities_in_range(self) -> None:
        scorer = self._trained_scorer()
        flist = [_all_zero_features(), _all_one_features()]
        probs = scorer.predict_proba_batch(flist)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)

    def test_batch_consistent_with_single(self) -> None:
        scorer = self._trained_scorer()
        f = _partial_features(n_set=15)
        p_single = scorer.predict_proba(f)
        p_batch = scorer.predict_proba_batch([f])
        assert abs(p_single - p_batch[0]) < 1e-9

    def test_none_features_ok(self) -> None:
        scorer = self._trained_scorer()
        p = scorer.predict_proba(_all_none_features())
        assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# XGBoostScorer save / load
# ---------------------------------------------------------------------------


class TestXGBoostScorerSaveLoad:
    def _trained_scorer(self) -> XGBoostScorer:
        scorer = XGBoostScorer()
        features_list, labels = _make_training_data(n=40, seed=2)
        scorer.fit(features_list, labels)
        return scorer

    def test_save_creates_files(self, tmp_path: Path) -> None:
        scorer = self._trained_scorer()
        meta_path = tmp_path / "model.json"
        scorer.save(meta_path)
        assert meta_path.exists()
        xgb_path = meta_path.with_suffix(".xgb.json")
        assert xgb_path.exists()

    def test_save_meta_contains_feature_names(self, tmp_path: Path) -> None:
        scorer = self._trained_scorer()
        meta_path = tmp_path / "model.json"
        scorer.save(meta_path)
        meta = json.loads(meta_path.read_text())
        assert meta["feature_names"] == FEATURE_NAMES

    def test_load_returns_scorer(self, tmp_path: Path) -> None:
        scorer = self._trained_scorer()
        meta_path = tmp_path / "model.json"
        scorer.save(meta_path)
        loaded = XGBoostScorer.load(meta_path)
        assert isinstance(loaded, XGBoostScorer)
        assert loaded.is_trained

    def test_loaded_predictions_match(self, tmp_path: Path) -> None:
        scorer = self._trained_scorer()
        meta_path = tmp_path / "model.json"
        scorer.save(meta_path)
        loaded = XGBoostScorer.load(meta_path)
        f = _partial_features(n_set=20)
        p_orig = scorer.predict_proba(f)
        p_load = loaded.predict_proba(f)
        assert abs(p_orig - p_load) < 1e-6

    def test_loaded_training_result_preserved(self, tmp_path: Path) -> None:
        scorer = self._trained_scorer()
        meta_path = tmp_path / "model.json"
        scorer.save(meta_path)
        loaded = XGBoostScorer.load(meta_path)
        assert loaded.training_result is not None
        assert loaded.training_result.n_samples == 40

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            XGBoostScorer.load(tmp_path / "nonexistent.json")

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        scorer = self._trained_scorer()
        meta_path = tmp_path / "subdir" / "deep" / "model.json"
        scorer.save(meta_path)
        assert meta_path.exists()


# ---------------------------------------------------------------------------
# TrainingResult dataclass
# ---------------------------------------------------------------------------


class TestTrainingResult:
    def test_fields(self) -> None:
        tr = TrainingResult(
            n_samples=100,
            n_positive=40,
            n_negative=60,
            n_features=35,
            best_iteration=50,
            feature_importance={"snr_score": 0.5, "log_snr_score": 0.5},
        )
        assert tr.n_samples == 100
        assert tr.n_positive == 40

    def test_frozen(self) -> None:
        tr = TrainingResult(
            n_samples=10,
            n_positive=5,
            n_negative=5,
            n_features=35,
            best_iteration=10,
            feature_importance={},
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            tr.n_samples = 99  # type: ignore[misc]
