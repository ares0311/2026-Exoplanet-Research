"""Tests for Skills/evaluate_scorer.py."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from Skills.evaluate_scorer import (
    _load_data,
    _roc_auc,
    _stratified_splits,
    _threshold_metrics,
    evaluate,
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
        assert _roc_auc(y_true, y_prob) == pytest.approx(1.0)

    def test_in_range(self) -> None:
        y_true = [1, 0, 1, 0, 1]
        y_prob = np.array([0.7, 0.3, 0.6, 0.4, 0.8])
        assert 0.0 <= _roc_auc(y_true, y_prob) <= 1.0

    def test_all_same_class_returns_nan(self) -> None:
        assert np.isnan(_roc_auc([1, 1, 1], np.array([0.9, 0.8, 0.7])))


# ---------------------------------------------------------------------------
# _threshold_metrics
# ---------------------------------------------------------------------------


class TestThresholdMetrics:
    def test_keys_present(self) -> None:
        m = _threshold_metrics([1, 0], np.array([0.8, 0.2]))
        assert {"auc", "acc", "precision", "recall", "f1"} <= m.keys()

    def test_perfect_predictions(self) -> None:
        m = _threshold_metrics([1, 1, 0, 0], np.array([0.9, 0.8, 0.1, 0.2]))
        assert m["acc"] == pytest.approx(1.0)
        assert m["f1"] == pytest.approx(1.0)

    def test_values_in_range(self) -> None:
        m = _threshold_metrics([1, 0, 1, 0], np.array([0.7, 0.3, 0.6, 0.4]))
        for k, v in m.items():
            assert 0.0 <= v <= 1.0 or np.isnan(v), f"{k}={v} out of range"


# ---------------------------------------------------------------------------
# _stratified_splits
# ---------------------------------------------------------------------------


class TestStratifiedSplits:
    def test_k_folds(self) -> None:
        labels = [0] * 10 + [1] * 10
        splits = _stratified_splits(labels, k=5)
        assert len(splits) == 5

    def test_val_covers_all(self) -> None:
        labels = [0] * 10 + [1] * 10
        splits = _stratified_splits(labels, k=5)
        all_val = sorted(i for _, val in splits for i in val)
        assert all_val == list(range(20))

    def test_no_overlap(self) -> None:
        labels = [0] * 6 + [1] * 6
        for train, val in _stratified_splits(labels, k=3):
            assert set(train) & set(val) == set()


# ---------------------------------------------------------------------------
# _load_data
# ---------------------------------------------------------------------------


class TestLoadData:
    def test_roundtrip(self, tmp_path: Path) -> None:
        fl, labels = _make_features(10)
        pkl = tmp_path / "data.pkl"
        with pkl.open("wb") as fh:
            pickle.dump({"features_list": fl, "labels": labels}, fh)
        fl2, l2 = _load_data(pkl)
        assert len(fl2) == 10
        assert l2 == labels

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            _load_data(tmp_path / "nonexistent.pkl")


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_returns_bayesian_key(self) -> None:
        fl, labels = _make_features(40)
        result = evaluate(fl, labels, k_folds=2)
        assert "bayesian" in result

    def test_xgboost_key_present(self) -> None:
        fl, labels = _make_features(40)
        result = evaluate(fl, labels, k_folds=2)
        assert "xgboost" in result

    def test_metrics_in_range(self) -> None:
        fl, labels = _make_features(40)
        result = evaluate(fl, labels, k_folds=2)
        for scorer_metrics in result.values():
            for v in scorer_metrics.values():
                assert 0.0 <= v <= 1.0 or np.isnan(v)

    def test_result_has_auc(self) -> None:
        fl, labels = _make_features(40)
        result = evaluate(fl, labels, k_folds=2)
        for metrics in result.values():
            assert "auc" in metrics
