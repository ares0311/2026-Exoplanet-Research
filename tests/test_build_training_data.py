"""Tests for Skills/build_training_data.py."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from Skills.build_training_data import (
    _clip01,
    _safe,
    _sigmoid,
    build_training_data,
    row_to_features,
)

from exo_toolkit.schemas import CandidateFeatures

# ---------------------------------------------------------------------------
# _safe
# ---------------------------------------------------------------------------


class TestSafe:
    def test_float_passthrough(self) -> None:
        assert _safe(3.14) == pytest.approx(3.14)

    def test_int_to_float(self) -> None:
        assert _safe(5) == pytest.approx(5.0)

    def test_nan_returns_none(self) -> None:
        assert _safe(float("nan")) is None

    def test_none_returns_none(self) -> None:
        assert _safe(None) is None

    def test_string_number(self) -> None:
        assert _safe("2.5") == pytest.approx(2.5)

    def test_non_numeric_string_returns_none(self) -> None:
        assert _safe("abc") is None


# ---------------------------------------------------------------------------
# _sigmoid
# ---------------------------------------------------------------------------


class TestSigmoid:
    def test_zero_at_mu(self) -> None:
        assert _sigmoid(0.0, mu=0.0, sigma=1.0) == pytest.approx(0.5)

    def test_approaches_one_high_x(self) -> None:
        assert _sigmoid(100.0) > 0.99

    def test_approaches_zero_low_x(self) -> None:
        assert _sigmoid(-100.0) < 0.01

    def test_custom_mu(self) -> None:
        assert _sigmoid(10.0, mu=10.0, sigma=1.0) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# _clip01
# ---------------------------------------------------------------------------


class TestClip01:
    def test_interior_value_unchanged(self) -> None:
        assert _clip01(0.5) == pytest.approx(0.5)

    def test_clips_above_one(self) -> None:
        assert _clip01(1.5) == pytest.approx(1.0)

    def test_clips_below_zero(self) -> None:
        assert _clip01(-0.3) == pytest.approx(0.0)

    def test_boundary_zero(self) -> None:
        assert _clip01(0.0) == pytest.approx(0.0)

    def test_boundary_one(self) -> None:
        assert _clip01(1.0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# row_to_features
# ---------------------------------------------------------------------------


def _full_row() -> pd.Series:
    return pd.Series({
        "koi_model_snr": 15.0,
        "koi_count": 10.0,
        "koi_period": 5.0,
        "koi_duration": 2.0,
        "koi_depth": 5000.0,
        "koi_prad": 2.5,
        "koi_dikco_msky": 0.5,
    })


class TestRowToFeatures:
    def test_returns_candidate_features(self) -> None:
        f = row_to_features(_full_row())
        assert isinstance(f, CandidateFeatures)

    def test_snr_score_in_range(self) -> None:
        f = row_to_features(_full_row())
        assert f.snr_score is not None
        assert 0.0 <= f.snr_score <= 1.0

    def test_log_snr_score_in_range(self) -> None:
        f = row_to_features(_full_row())
        assert f.log_snr_score is not None
        assert 0.0 <= f.log_snr_score <= 1.0

    def test_transit_count_score_in_range(self) -> None:
        f = row_to_features(_full_row())
        assert f.transit_count_score is not None
        assert 0.0 <= f.transit_count_score <= 1.0

    def test_duration_scores_set(self) -> None:
        f = row_to_features(_full_row())
        assert f.duration_plausibility_score is not None
        assert f.duration_implausibility_score is not None

    def test_large_depth_score_in_range(self) -> None:
        f = row_to_features(_full_row())
        assert f.large_depth_score is not None
        assert 0.0 <= f.large_depth_score <= 1.0

    def test_centroid_offset_score_in_range(self) -> None:
        f = row_to_features(_full_row())
        assert f.centroid_offset_score is not None
        assert 0.0 <= f.centroid_offset_score <= 1.0

    def test_unmapped_fields_are_none(self) -> None:
        f = row_to_features(_full_row())
        assert f.odd_even_mismatch_score is None
        assert f.secondary_eclipse_score is None
        assert f.stellar_variability_score is None

    def test_missing_snr_gives_none(self) -> None:
        row = _full_row().copy()
        row["koi_model_snr"] = float("nan")
        f = row_to_features(row)
        assert f.snr_score is None
        assert f.log_snr_score is None

    def test_missing_period_gives_none_duration_scores(self) -> None:
        row = _full_row().copy()
        row["koi_period"] = float("nan")
        f = row_to_features(row)
        assert f.duration_plausibility_score is None
        assert f.duration_implausibility_score is None

    def test_high_snr_high_score(self) -> None:
        row = _full_row().copy()
        row["koi_model_snr"] = 100.0
        f_high = row_to_features(row)
        row["koi_model_snr"] = 5.0
        f_low = row_to_features(row)
        assert f_high.snr_score > f_low.snr_score  # type: ignore[operator]

    def test_large_depth_high_score(self) -> None:
        row = _full_row().copy()
        row["koi_depth"] = 50000.0
        f = row_to_features(row)
        assert f.large_depth_score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# build_training_data
# ---------------------------------------------------------------------------


def _make_koi_df(n_conf: int = 5, n_fp: int = 5) -> pd.DataFrame:
    rows = []
    for _ in range(n_conf):
        rows.append({
            "koi_disposition": "CONFIRMED",
            "koi_model_snr": 20.0, "koi_count": 15.0,
            "koi_period": 5.0, "koi_duration": 2.0,
            "koi_depth": 3000.0, "koi_prad": 2.0,
            "koi_dikco_msky": 0.1,
        })
    for _ in range(n_fp):
        rows.append({
            "koi_disposition": "FALSE POSITIVE",
            "koi_model_snr": 8.0, "koi_count": 5.0,
            "koi_period": 3.0, "koi_duration": 1.5,
            "koi_depth": 20000.0, "koi_prad": 15.0,
            "koi_dikco_msky": 2.0,
        })
    return pd.DataFrame(rows)


class TestBuildTrainingData:
    def test_correct_count(self, tmp_path: Path) -> None:
        df = _make_koi_df(5, 5)
        csv = tmp_path / "koi.csv"
        df.to_csv(csv, index=False)
        pkl = tmp_path / "out.pkl"
        features_list, labels = build_training_data(csv, pkl)
        assert len(features_list) == 10
        assert len(labels) == 10

    def test_labels_binary(self, tmp_path: Path) -> None:
        df = _make_koi_df(3, 4)
        csv = tmp_path / "koi.csv"
        df.to_csv(csv, index=False)
        _, labels = build_training_data(csv, tmp_path / "out.pkl")
        assert all(lbl in (0, 1) for lbl in labels)

    def test_confirmed_label_one(self, tmp_path: Path) -> None:
        df = _make_koi_df(3, 0)
        csv = tmp_path / "koi.csv"
        df.to_csv(csv, index=False)
        _, labels = build_training_data(csv, tmp_path / "out.pkl")
        assert all(lbl == 1 for lbl in labels)

    def test_fp_label_zero(self, tmp_path: Path) -> None:
        df = _make_koi_df(0, 4)
        csv = tmp_path / "koi.csv"
        df.to_csv(csv, index=False)
        _, labels = build_training_data(csv, tmp_path / "out.pkl")
        assert all(lbl == 0 for lbl in labels)

    def test_candidate_excluded(self, tmp_path: Path) -> None:
        df = _make_koi_df(2, 2)
        candidate_row = {
            "koi_disposition": "CANDIDATE",
            "koi_model_snr": 10.0, "koi_count": 5.0,
            "koi_period": 4.0, "koi_duration": 1.0,
            "koi_depth": 1000.0, "koi_prad": 1.5,
            "koi_dikco_msky": 0.2,
        }
        df = pd.concat([df, pd.DataFrame([candidate_row])], ignore_index=True)
        csv = tmp_path / "koi.csv"
        df.to_csv(csv, index=False)
        features_list, labels = build_training_data(csv, tmp_path / "out.pkl")
        assert len(labels) == 4  # CANDIDATE excluded

    def test_pickle_written(self, tmp_path: Path) -> None:
        df = _make_koi_df(2, 2)
        csv = tmp_path / "koi.csv"
        df.to_csv(csv, index=False)
        pkl = tmp_path / "out.pkl"
        build_training_data(csv, pkl)
        assert pkl.exists()
        with pkl.open("rb") as fh:
            data = pickle.load(fh)
        assert "features_list" in data
        assert "labels" in data

    def test_features_are_candidate_features(self, tmp_path: Path) -> None:
        df = _make_koi_df(2, 2)
        csv = tmp_path / "koi.csv"
        df.to_csv(csv, index=False)
        features_list, _ = build_training_data(csv, tmp_path / "out.pkl")
        for f in features_list:
            assert isinstance(f, CandidateFeatures)
