"""Tests for Skills/build_tess_training_data.py."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from Skills.build_tess_training_data import (
    _clip01,
    _disposition_to_label,
    _safe,
    _sigmoid,
    build_training_data,
    row_to_features,
)

from exo_toolkit.schemas import CandidateFeatures

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _full_row() -> pd.Series:
    return pd.Series({
        "snr": 20.0,
        "period_days": 5.0,
        "duration_hours": 2.0,
        "depth_mmag": 5.0,
        "planet_radius_earth": 3.0,
        "n_sectors": 4.0,
        "tfopwg_disposition": "CP",
    })


def _make_toi_df(n_cp: int = 5, n_fp: int = 5) -> pd.DataFrame:
    rows = []
    for _ in range(n_cp):
        rows.append({
            "tfopwg_disposition": "CP",
            "snr": 25.0, "period_days": 5.0, "duration_hours": 2.0,
            "depth_mmag": 3.0, "planet_radius_earth": 2.0, "n_sectors": 4.0,
        })
    for _ in range(n_fp):
        rows.append({
            "tfopwg_disposition": "FP",
            "snr": 8.0, "period_days": 3.0, "duration_hours": 1.5,
            "depth_mmag": 20.0, "planet_radius_earth": 18.0, "n_sectors": 2.0,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# _safe
# ---------------------------------------------------------------------------


class TestSafe:
    def test_float_passthrough(self) -> None:
        assert _safe(3.14) == pytest.approx(3.14)

    def test_nan_returns_none(self) -> None:
        assert _safe(float("nan")) is None

    def test_none_returns_none(self) -> None:
        assert _safe(None) is None

    def test_string_number(self) -> None:
        assert _safe("2.5") == pytest.approx(2.5)

    def test_non_numeric_returns_none(self) -> None:
        assert _safe("abc") is None


# ---------------------------------------------------------------------------
# _sigmoid / _clip01
# ---------------------------------------------------------------------------


class TestSigmoid:
    def test_zero_at_mu(self) -> None:
        assert _sigmoid(0.0, mu=0.0, sigma=1.0) == pytest.approx(0.5)

    def test_high_input_near_one(self) -> None:
        assert _sigmoid(100.0) > 0.99


class TestClip01:
    def test_interior_unchanged(self) -> None:
        assert _clip01(0.5) == pytest.approx(0.5)

    def test_clips_above_one(self) -> None:
        assert _clip01(1.5) == pytest.approx(1.0)

    def test_clips_below_zero(self) -> None:
        assert _clip01(-0.3) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _disposition_to_label
# ---------------------------------------------------------------------------


class TestDispositionToLabel:
    def test_cp_is_one(self) -> None:
        assert _disposition_to_label("CP") == 1

    def test_fp_is_zero(self) -> None:
        assert _disposition_to_label("FP") == 0

    def test_eb_is_zero(self) -> None:
        assert _disposition_to_label("EB") == 0

    def test_pc_is_none(self) -> None:
        assert _disposition_to_label("PC") is None

    def test_unknown_is_none(self) -> None:
        assert _disposition_to_label("APC") is None

    def test_lowercase_handled(self) -> None:
        assert _disposition_to_label("cp") == 1


# ---------------------------------------------------------------------------
# row_to_features
# ---------------------------------------------------------------------------


class TestRowToFeatures:
    def test_returns_candidate_features(self) -> None:
        assert isinstance(row_to_features(_full_row()), CandidateFeatures)

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

    def test_companion_radius_score_in_range(self) -> None:
        f = row_to_features(_full_row())
        assert f.companion_radius_too_large_score is not None
        assert 0.0 <= f.companion_radius_too_large_score <= 1.0

    def test_unmapped_fields_are_none(self) -> None:
        f = row_to_features(_full_row())
        assert f.odd_even_mismatch_score is None
        assert f.secondary_eclipse_score is None
        assert f.stellar_variability_score is None

    def test_missing_snr_gives_none(self) -> None:
        row = _full_row().copy()
        row["snr"] = float("nan")
        f = row_to_features(row)
        assert f.snr_score is None
        assert f.log_snr_score is None

    def test_missing_period_gives_none_duration_scores(self) -> None:
        row = _full_row().copy()
        row["period_days"] = float("nan")
        f = row_to_features(row)
        assert f.duration_plausibility_score is None
        assert f.duration_implausibility_score is None

    def test_high_snr_higher_score(self) -> None:
        row_hi = _full_row().copy()
        row_lo = _full_row().copy()
        row_hi["snr"] = 100.0
        row_lo["snr"] = 5.0
        assert row_to_features(row_hi).snr_score > row_to_features(row_lo).snr_score  # type: ignore[operator]

    def test_large_prad_high_companion_score(self) -> None:
        row = _full_row().copy()
        row["planet_radius_earth"] = 50.0
        f = row_to_features(row)
        assert f.companion_radius_too_large_score is not None
        assert f.companion_radius_too_large_score > 0.9


# ---------------------------------------------------------------------------
# build_training_data
# ---------------------------------------------------------------------------


class TestBuildTrainingData:
    def test_correct_count(self, tmp_path: Path) -> None:
        df = _make_toi_df(5, 5)
        csv = tmp_path / "toi.csv"
        df.to_csv(csv, index=False)
        fl, labels = build_training_data(csv, tmp_path / "out.pkl")
        assert len(fl) == 10
        assert len(labels) == 10

    def test_labels_binary(self, tmp_path: Path) -> None:
        df = _make_toi_df(3, 4)
        csv = tmp_path / "toi.csv"
        df.to_csv(csv, index=False)
        _, labels = build_training_data(csv, tmp_path / "out.pkl")
        assert all(lbl in (0, 1) for lbl in labels)

    def test_cp_label_one(self, tmp_path: Path) -> None:
        df = _make_toi_df(3, 0)
        csv = tmp_path / "toi.csv"
        df.to_csv(csv, index=False)
        _, labels = build_training_data(csv, tmp_path / "out.pkl")
        assert all(lbl == 1 for lbl in labels)

    def test_fp_label_zero(self, tmp_path: Path) -> None:
        df = _make_toi_df(0, 4)
        csv = tmp_path / "toi.csv"
        df.to_csv(csv, index=False)
        _, labels = build_training_data(csv, tmp_path / "out.pkl")
        assert all(lbl == 0 for lbl in labels)

    def test_pc_excluded(self, tmp_path: Path) -> None:
        df = _make_toi_df(2, 2)
        pc_row = {
            "tfopwg_disposition": "PC",
            "snr": 10.0, "period_days": 4.0, "duration_hours": 1.5,
            "depth_mmag": 2.0, "planet_radius_earth": 1.5, "n_sectors": 3.0,
        }
        df = pd.concat([df, pd.DataFrame([pc_row])], ignore_index=True)
        csv = tmp_path / "toi.csv"
        df.to_csv(csv, index=False)
        _, labels = build_training_data(csv, tmp_path / "out.pkl")
        assert len(labels) == 4  # PC excluded

    def test_pickle_written(self, tmp_path: Path) -> None:
        df = _make_toi_df(2, 2)
        csv = tmp_path / "toi.csv"
        df.to_csv(csv, index=False)
        pkl = tmp_path / "out.pkl"
        build_training_data(csv, pkl)
        assert pkl.exists()
        with pkl.open("rb") as fh:
            data = pickle.load(fh)
        assert "features_list" in data
        assert "labels" in data

    def test_features_are_candidate_features(self, tmp_path: Path) -> None:
        df = _make_toi_df(2, 2)
        csv = tmp_path / "toi.csv"
        df.to_csv(csv, index=False)
        fl, _ = build_training_data(csv, tmp_path / "out.pkl")
        for f in fl:
            assert isinstance(f, CandidateFeatures)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            build_training_data(tmp_path / "nonexistent.csv", tmp_path / "out.pkl")

    def test_eb_label_zero(self, tmp_path: Path) -> None:
        rows = [{"tfopwg_disposition": "EB", "snr": 10.0, "period_days": 3.0,
                 "duration_hours": 1.0, "depth_mmag": 10.0,
                 "planet_radius_earth": 20.0, "n_sectors": 2.0}]
        df = pd.DataFrame(rows)
        csv = tmp_path / "toi.csv"
        df.to_csv(csv, index=False)
        _, labels = build_training_data(csv, tmp_path / "out.pkl")
        assert labels == [0]
