"""Tests for Skills/build_combined_training_data.py."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from Skills.build_combined_training_data import (
    _stratified_subsample,
    build_combined,
    merge_datasets,
)

from exo_toolkit.ml.xgboost_scorer import FEATURE_NAMES
from exo_toolkit.schemas import CandidateFeatures

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pkl(path: Path, n_pos: int = 10, n_neg: int = 10) -> None:
    rng = np.random.default_rng(0)
    features_list = [
        CandidateFeatures(**{k: float(rng.uniform(0.1, 0.9)) for k in FEATURE_NAMES})
        for _ in range(n_pos + n_neg)
    ]
    labels = [1] * n_pos + [0] * n_neg
    with path.open("wb") as fh:
        pickle.dump({"features_list": features_list, "labels": labels}, fh)


# ---------------------------------------------------------------------------
# _stratified_subsample
# ---------------------------------------------------------------------------


class TestStratifiedSubsample:
    def test_size_capped(self) -> None:
        rng = np.random.default_rng(0)
        features = [CandidateFeatures() for _ in range(100)]
        labels = [1] * 50 + [0] * 50
        fl, lb = _stratified_subsample(features, labels, 20, rng)
        assert len(lb) <= 20

    def test_class_balance_preserved(self) -> None:
        rng = np.random.default_rng(0)
        features = [CandidateFeatures() for _ in range(100)]
        labels = [1] * 50 + [0] * 50
        _, lb = _stratified_subsample(features, labels, 40, rng)
        assert sum(lb) == len(lb) // 2

    def test_returns_correct_types(self) -> None:
        rng = np.random.default_rng(0)
        features = [CandidateFeatures() for _ in range(20)]
        labels = [1] * 10 + [0] * 10
        fl, lb = _stratified_subsample(features, labels, 10, rng)
        assert all(isinstance(f, CandidateFeatures) for f in fl)
        assert all(isinstance(lbl, (int, np.integer)) for lbl in lb)


# ---------------------------------------------------------------------------
# merge_datasets
# ---------------------------------------------------------------------------


class TestMergeDatasets:
    def test_concatenates(self, tmp_path: Path) -> None:
        p1 = tmp_path / "a.pkl"
        p2 = tmp_path / "b.pkl"
        _make_pkl(p1, 5, 5)
        _make_pkl(p2, 3, 3)
        fl, lb = merge_datasets(p1, p2)
        assert len(lb) == 16

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            merge_datasets(tmp_path / "nonexistent.pkl")

    def test_max_per_source_caps(self, tmp_path: Path) -> None:
        p1 = tmp_path / "a.pkl"
        _make_pkl(p1, 50, 50)
        _, lb = merge_datasets(p1, max_per_source=20)
        assert len(lb) <= 20

    def test_single_source(self, tmp_path: Path) -> None:
        p = tmp_path / "data.pkl"
        _make_pkl(p, 8, 8)
        fl, lb = merge_datasets(p)
        assert len(lb) == 16


# ---------------------------------------------------------------------------
# build_combined
# ---------------------------------------------------------------------------


class TestBuildCombined:
    def test_output_written(self, tmp_path: Path) -> None:
        k = tmp_path / "kepler.pkl"
        t = tmp_path / "tess.pkl"
        _make_pkl(k, 5, 5)
        _make_pkl(t, 3, 3)
        out = tmp_path / "combined.pkl"
        build_combined(k, t, out)
        assert out.exists()

    def test_combined_count(self, tmp_path: Path) -> None:
        k = tmp_path / "kepler.pkl"
        t = tmp_path / "tess.pkl"
        _make_pkl(k, 5, 5)
        _make_pkl(t, 3, 3)
        out = tmp_path / "combined.pkl"
        fl, lb = build_combined(k, t, out)
        assert len(lb) == 16

    def test_labels_binary(self, tmp_path: Path) -> None:
        k = tmp_path / "kepler.pkl"
        t = tmp_path / "tess.pkl"
        _make_pkl(k, 5, 5)
        _make_pkl(t, 3, 3)
        out = tmp_path / "combined.pkl"
        _, lb = build_combined(k, t, out)
        assert all(lbl in (0, 1) for lbl in lb)

    def test_pickle_loadable(self, tmp_path: Path) -> None:
        k = tmp_path / "kepler.pkl"
        t = tmp_path / "tess.pkl"
        _make_pkl(k, 4, 4)
        _make_pkl(t, 4, 4)
        out = tmp_path / "combined.pkl"
        build_combined(k, t, out)
        with out.open("rb") as fh:
            data = pickle.load(fh)
        assert "features_list" in data and "labels" in data

    def test_max_per_source(self, tmp_path: Path) -> None:
        k = tmp_path / "kepler.pkl"
        t = tmp_path / "tess.pkl"
        _make_pkl(k, 20, 20)
        _make_pkl(t, 20, 20)
        out = tmp_path / "combined.pkl"
        _, lb = build_combined(k, t, out, max_per_source=10)
        assert len(lb) <= 20

    def test_missing_kepler_raises(self, tmp_path: Path) -> None:
        t = tmp_path / "tess.pkl"
        _make_pkl(t, 3, 3)
        with pytest.raises(FileNotFoundError):
            build_combined(tmp_path / "missing.pkl", t, tmp_path / "out.pkl")
