"""Tests for Skills/train_xgboost.py."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from Skills.train_xgboost import (
    _metrics,
    _roc_auc,
    _stratified_kfold_indices,
    load_training_data,
    train_and_evaluate,
)

from exo_toolkit.ml.xgboost_scorer import FEATURE_NAMES
from exo_toolkit.schemas import CandidateFeatures

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_features(n: int = 40, seed: int = 0) -> tuple[list[CandidateFeatures], list[int]]:
    rng = np.random.default_rng(seed)
    features_list, labels = [], []
    for i in range(n):
        label = i % 2
        vals = {name: float(rng.uniform(0.2, 0.8)) for name in FEATURE_NAMES}
        features_list.append(CandidateFeatures(**vals))
        labels.append(label)
    return features_list, labels


# ---------------------------------------------------------------------------
# _roc_auc
# ---------------------------------------------------------------------------


class TestRocAuc:
    def test_perfect_classifier(self) -> None:
        y_true = [1, 1, 0, 0]
        y_prob = np.array([0.9, 0.8, 0.2, 0.1])
        auc = _roc_auc(y_true, y_prob)
        assert auc == pytest.approx(1.0)

    def test_random_classifier_near_half(self) -> None:
        rng = np.random.default_rng(42)
        y_true = list(rng.integers(0, 2, 200))
        y_prob = rng.uniform(0, 1, 200)
        auc = _roc_auc(y_true, y_prob)
        assert 0.3 <= auc <= 0.7

    def test_in_range(self) -> None:
        y_true = [1, 0, 1, 0, 1]
        y_prob = np.array([0.7, 0.3, 0.6, 0.4, 0.8])
        auc = _roc_auc(y_true, y_prob)
        assert 0.0 <= auc <= 1.0

    def test_all_same_class_returns_nan(self) -> None:
        y_true = [1, 1, 1]
        y_prob = np.array([0.9, 0.8, 0.7])
        auc = _roc_auc(y_true, y_prob)
        assert np.isnan(auc)


# ---------------------------------------------------------------------------
# _metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_perfect_predictions(self) -> None:
        y_true = [1, 1, 0, 0]
        y_prob = np.array([0.9, 0.8, 0.1, 0.2])
        m = _metrics(y_true, y_prob)
        assert m["acc"] == pytest.approx(1.0)
        assert m["precision"] == pytest.approx(1.0)
        assert m["recall"] == pytest.approx(1.0)
        assert m["f1"] == pytest.approx(1.0)

    def test_keys_present(self) -> None:
        m = _metrics([1, 0], np.array([0.8, 0.2]))
        assert set(m.keys()) == {"auc", "acc", "precision", "recall", "f1"}

    def test_values_in_range(self) -> None:
        y_true = [1, 0, 1, 0]
        y_prob = np.array([0.7, 0.3, 0.6, 0.4])
        m = _metrics(y_true, y_prob)
        for v in m.values():
            assert 0.0 <= v <= 1.0 or np.isnan(v)

    def test_all_wrong(self) -> None:
        y_true = [1, 1, 0, 0]
        y_prob = np.array([0.1, 0.2, 0.9, 0.8])
        m = _metrics(y_true, y_prob)
        assert m["acc"] == pytest.approx(0.0)
        assert m["recall"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _stratified_kfold_indices
# ---------------------------------------------------------------------------


class TestStratifiedKfold:
    def test_k_splits(self) -> None:
        labels = [0] * 10 + [1] * 10
        splits = _stratified_kfold_indices(labels, k=5)
        assert len(splits) == 5

    def test_val_covers_all_samples(self) -> None:
        labels = [0] * 10 + [1] * 10
        splits = _stratified_kfold_indices(labels, k=5)
        all_val = sorted(i for _, val in splits for i in val)
        assert all_val == list(range(20))

    def test_no_overlap_within_fold(self) -> None:
        labels = [0] * 6 + [1] * 6
        for train_idx, val_idx in _stratified_kfold_indices(labels, k=3):
            assert set(train_idx) & set(val_idx) == set()

    def test_stratified_class_balance(self) -> None:
        labels = [0] * 20 + [1] * 20
        for _, val_idx in _stratified_kfold_indices(labels, k=5):
            val_labels = [labels[i] for i in val_idx]
            assert sum(val_labels) == len(val_labels) // 2


# ---------------------------------------------------------------------------
# train_and_evaluate
# ---------------------------------------------------------------------------


class TestTrainAndEvaluate:
    def test_returns_metrics_dict(self, tmp_path: Path) -> None:
        features_list, labels = _make_features(40)
        m = train_and_evaluate(
            features_list, labels, k_folds=2, output_path=tmp_path / "model.json"
        )
        assert isinstance(m, dict)
        assert "auc" in m and "f1" in m

    def test_model_saved(self, tmp_path: Path) -> None:
        features_list, labels = _make_features(40)
        out = tmp_path / "model.json"
        train_and_evaluate(features_list, labels, k_folds=2, output_path=out)
        assert out.exists()

    def test_metrics_in_range(self, tmp_path: Path) -> None:
        features_list, labels = _make_features(40)
        m = train_and_evaluate(
            features_list, labels, k_folds=2, output_path=tmp_path / "model.json"
        )
        for v in m.values():
            assert 0.0 <= v <= 1.0 or np.isnan(v)


# ---------------------------------------------------------------------------
# load_training_data
# ---------------------------------------------------------------------------


class TestLoadTrainingData:
    def test_roundtrip(self, tmp_path: Path) -> None:
        features_list, labels = _make_features(10)
        pkl = tmp_path / "data.pkl"
        with pkl.open("wb") as fh:
            pickle.dump({"features_list": features_list, "labels": labels}, fh)
        loaded_f, loaded_l = load_training_data(pkl)
        assert len(loaded_f) == 10
        assert loaded_l == labels

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_training_data(tmp_path / "nonexistent.pkl")
