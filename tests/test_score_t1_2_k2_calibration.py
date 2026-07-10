"""Tests for Skills/score_t1_2_k2_calibration.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Skills"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from score_t1_2_k2_calibration import (  # noqa: E402
    format_scoring_summary,
    row_to_candidate_features,
    score_calibration_manifest,
    write_partial_predictions,
)

_XGB_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "xgboost_koi.json"


def _sample_manifest_rows() -> list[dict]:
    return [
        {
            "epic_id": 201092629,
            "label": 1,
            "period_days": 26.8199,
            "epoch_bjd": 2457584.2063,
            "duration_hours": 4.1,
            "depth_ppm": 900.0,
            "planet_radius_rearth": 2.55,
            "system_planet_count": 1,
        },
        {
            "epic_id": 999999999,
            "label": 0,
            "period_days": 0.51,
            "epoch_bjd": 2457063.5,
            "duration_hours": 1.2,
            "depth_ppm": 5000.0,
            "planet_radius_rearth": 20.0,
            "system_planet_count": 1,
        },
    ]


def _write_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        "".join(json.dumps(r) + "\n" for r in _sample_manifest_rows()), encoding="utf-8"
    )
    return path


def test_row_to_candidate_features_maps_expected_fields() -> None:
    row = _sample_manifest_rows()[0]
    features = row_to_candidate_features(row)
    assert features.snr_score is None  # no K2 SNR equivalent
    assert features.centroid_offset_score is None  # no K2 centroid equivalent
    assert features.transit_count_score is not None  # from system_planet_count
    assert features.large_depth_score is not None  # from depth_ppm
    assert features.companion_radius_too_large_score is not None  # from planet_radius


def test_row_to_candidate_features_handles_missing_values() -> None:
    row = {"epic_id": 1, "label": 0}
    features = row_to_candidate_features(row)
    assert features.snr_score is None
    assert features.large_depth_score is None
    assert features.companion_radius_too_large_score is None


def test_score_calibration_manifest_end_to_end(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path)
    scored = score_calibration_manifest(manifest_path, xgb_model_path=_XGB_MODEL_PATH)

    assert len(scored) == 2
    for row in scored:
        assert 0.0 <= row["xgb_prob"] <= 1.0
        assert 0.0 <= row["bayes_prob"] <= 1.0
        assert row["cnn_prob"] is None
        assert row["label"] in (0, 1)
    assert {r["epic_id"] for r in scored} == {201092629, 999999999}


def test_write_partial_predictions_round_trip(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path)
    scored = score_calibration_manifest(manifest_path, xgb_model_path=_XGB_MODEL_PATH)
    output_path = tmp_path / "predictions.jsonl"
    write_partial_predictions(scored, output_path)

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    parsed = json.loads(lines[0])
    assert set(parsed) == {"epic_id", "label", "xgb_prob", "bayes_prob", "cnn_prob"}


def test_format_scoring_summary_reports_missing_cnn() -> None:
    rows = [
        {"epic_id": 1, "label": 1, "xgb_prob": 0.9, "bayes_prob": 0.8, "cnn_prob": None},
        {"epic_id": 2, "label": 0, "xgb_prob": 0.1, "bayes_prob": 0.2, "cnn_prob": None},
    ]
    text = format_scoring_summary(rows)
    assert "Rows scored**: 2" in text
    assert "Rows still missing cnn_prob**: 2" in text
    assert "Positive (label=1)**: 1" in text
