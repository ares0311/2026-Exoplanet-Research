"""Tests for Skills/calibrate_stacking_weights.py (offline only)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from Skills.calibrate_stacking_weights import (
    StackingCalibResult,
    calibrate_stacking_weights,
    extract_from_pipeline_output,
    format_calibration_result,
    load_predictions_jsonl,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_predictions_jsonl(
    path: Path,
    n: int = 40,
    *,
    informative: bool = True,
) -> None:
    """Write a balanced JSONL predictions file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for i in range(n):
            label = i % 2
            if informative:
                cnn = 0.85 if label == 1 else 0.15
                xgb = 0.80 if label == 1 else 0.20
                bayes = 0.75 if label == 1 else 0.25
            else:
                cnn = xgb = bayes = 0.5
            fh.write(json.dumps({
                "label": label,
                "cnn_prob": cnn,
                "xgb_prob": xgb,
                "bayes_prob": bayes,
            }) + "\n")


def _write_pipeline_output(path: Path, tic_ids: list[int]) -> None:
    records = []
    for tic_id in tic_ids:
        records.append({
            "tic_id": tic_id,
            "posterior": {"planet_candidate": 0.7},
            "xgb_planet_probability": 0.65,
            "cnn_planet_probability": 0.72,
        })
    path.write_text(json.dumps(records))


def _write_labels_csv(path: Path, tic_ids: list[int]) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["tic_id", "label"])
        writer.writeheader()
        for i, tic_id in enumerate(tic_ids):
            writer.writerow({"tic_id": tic_id, "label": i % 2})


# ---------------------------------------------------------------------------
# load_predictions_jsonl
# ---------------------------------------------------------------------------


class TestLoadPredictionsJsonl:
    def test_loads_valid_file(self, tmp_path: Path) -> None:
        p = tmp_path / "pred.jsonl"
        _write_predictions_jsonl(p, n=10)
        labels, xgb, cnn, bayes = load_predictions_jsonl(p)
        assert len(labels) == 10
        assert len(xgb) == len(cnn) == len(bayes) == 10

    def test_missing_key_raises_value_error(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.jsonl"
        p.write_text(json.dumps({"label": 1, "cnn_prob": 0.8}) + "\n")
        with pytest.raises(ValueError, match="xgb_prob"):
            load_predictions_jsonl(p)

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        p = tmp_path / "pred.jsonl"
        p.write_text(
            json.dumps({"label": 1, "cnn_prob": 0.8, "xgb_prob": 0.7, "bayes_prob": 0.6})
            + "\n\n"
        )
        labels, _, _, _ = load_predictions_jsonl(p)
        assert len(labels) == 1


# ---------------------------------------------------------------------------
# extract_from_pipeline_output
# ---------------------------------------------------------------------------


class TestExtractFromPipelineOutput:
    def test_extracts_matched_records(self, tmp_path: Path) -> None:
        tic_ids = list(range(100, 110))
        pipeline = tmp_path / "out.json"
        labels_csv = tmp_path / "labels.csv"
        _write_pipeline_output(pipeline, tic_ids)
        _write_labels_csv(labels_csv, tic_ids)
        out = extract_from_pipeline_output(pipeline, labels_csv)
        assert out.exists()
        lines = [ln for ln in out.read_text().splitlines() if ln.strip()]
        assert len(lines) == 10

    def test_unmatched_records_skipped(self, tmp_path: Path) -> None:
        pipeline = tmp_path / "out.json"
        labels_csv = tmp_path / "labels.csv"
        _write_pipeline_output(pipeline, [100, 101])
        _write_labels_csv(labels_csv, [999])  # no overlap
        with pytest.raises(ValueError, match="No records matched"):
            extract_from_pipeline_output(pipeline, labels_csv)

    def test_output_has_required_keys(self, tmp_path: Path) -> None:
        tic_ids = [200, 201, 202]
        pipeline = tmp_path / "out.json"
        labels_csv = tmp_path / "labels.csv"
        _write_pipeline_output(pipeline, tic_ids)
        _write_labels_csv(labels_csv, tic_ids)
        out = extract_from_pipeline_output(pipeline, labels_csv)
        rec = json.loads(out.read_text().splitlines()[0])
        for key in ("tic_id", "label", "cnn_prob", "xgb_prob", "bayes_prob"):
            assert key in rec


# ---------------------------------------------------------------------------
# calibrate_stacking_weights
# ---------------------------------------------------------------------------


class TestCalibrateStackingWeights:
    def test_missing_file_returns_missing_flag(self, tmp_path: Path) -> None:
        result = calibrate_stacking_weights(
            tmp_path / "missing.jsonl",
            output_path=tmp_path / "weights.json",
        )
        assert result.flag == "MISSING_FILE"

    def test_insufficient_data_returns_flag(self, tmp_path: Path) -> None:
        p = tmp_path / "pred.jsonl"
        _write_predictions_jsonl(p, n=5)
        result = calibrate_stacking_weights(p, output_path=tmp_path / "w.json")
        assert result.flag == "INSUFFICIENT"

    def test_ok_with_informative_predictions(self, tmp_path: Path) -> None:
        p = tmp_path / "pred.jsonl"
        _write_predictions_jsonl(p, n=60, informative=True)
        result = calibrate_stacking_weights(p, output_path=tmp_path / "w.json")
        assert result.flag == "OK"

    def test_weights_sum_to_one(self, tmp_path: Path) -> None:
        p = tmp_path / "pred.jsonl"
        _write_predictions_jsonl(p, n=60)
        result = calibrate_stacking_weights(p, output_path=tmp_path / "w.json")
        if result.flag == "OK":
            assert sum(result.best_weights) == pytest.approx(1.0, abs=0.01)

    def test_output_json_written_on_ok(self, tmp_path: Path) -> None:
        p = tmp_path / "pred.jsonl"
        out = tmp_path / "weights.json"
        _write_predictions_jsonl(p, n=60)
        result = calibrate_stacking_weights(p, output_path=out)
        if result.flag == "OK":
            assert out.exists()
            w = json.loads(out.read_text())
            assert "w_xgb" in w
            assert "w_cnn" in w
            assert "w_bayes" in w

    def test_n_samples_reported(self, tmp_path: Path) -> None:
        p = tmp_path / "pred.jsonl"
        _write_predictions_jsonl(p, n=40)
        result = calibrate_stacking_weights(p, output_path=tmp_path / "w.json")
        assert result.n_samples == 40

    def test_auc_above_half_for_informative(self, tmp_path: Path) -> None:
        p = tmp_path / "pred.jsonl"
        _write_predictions_jsonl(p, n=60, informative=True)
        result = calibrate_stacking_weights(p, output_path=tmp_path / "w.json")
        if result.flag == "OK":
            assert result.best_auc > 0.5

    def test_calibrated_at_iso_format(self, tmp_path: Path) -> None:
        p = tmp_path / "pred.jsonl"
        _write_predictions_jsonl(p, n=40)
        result = calibrate_stacking_weights(p, output_path=tmp_path / "w.json")
        assert "T" in result.calibrated_at


# ---------------------------------------------------------------------------
# format_calibration_result
# ---------------------------------------------------------------------------


class TestFormatCalibrationResult:
    def test_format_ok_contains_weights(self, tmp_path: Path) -> None:
        p = tmp_path / "pred.jsonl"
        _write_predictions_jsonl(p, n=60)
        result = calibrate_stacking_weights(p, output_path=tmp_path / "w.json")
        text = format_calibration_result(result)
        assert "XGBoost" in text
        assert "CNN" in text
        assert "Bayesian" in text

    def test_format_ok_contains_commit_recipe(self, tmp_path: Path) -> None:
        p = tmp_path / "pred.jsonl"
        _write_predictions_jsonl(p, n=60)
        result = calibrate_stacking_weights(p, output_path=tmp_path / "w.json")
        text = format_calibration_result(result)
        if result.flag == "OK":
            assert "git commit" in text

    def test_format_missing_file(self) -> None:
        result = StackingCalibResult(
            best_weights=(0.35, 0.35, 0.30),
            best_auc=0.0,
            n_samples=0,
            n_positive=0,
            n_negative=0,
            grid_step=0.05,
            flag="MISSING_FILE",
            calibrated_at="2026-06-11T10:00:00+00:00",
            output_path="models/stacking_weights.json",
        )
        text = format_calibration_result(result)
        assert "MISSING_FILE" in text
