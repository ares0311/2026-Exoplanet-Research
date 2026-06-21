"""Tests for Skills/evaluate_cnn_checkpoint.py (offline only — no PyTorch)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from Skills.evaluate_cnn_checkpoint import (
    CnnEvalMetrics,
    CnnEvalResult,
    _apply_platt,
    _apply_temperature,
    _auc_roc,
    _best_f1_threshold,
    _brier,
    _compute_metrics,
    _ece,
    _fit_platt,
    _fit_temperature,
    evaluate_cnn_checkpoint,
    format_eval_result,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_split(path: Path, examples: list[dict], split: str = "test") -> None:
    path.write_text(json.dumps({"split": split, "examples": examples}))


def _good_example(label: int, signal_strength: float = 1.0) -> dict:
    """Phase-folded snippet: transit-like dip at center for label=1."""
    flux = [1.0] * 201
    if label == 1:
        mid = 100
        for i in range(mid - 5, mid + 6):
            flux[i] = 1.0 - 0.01 * signal_strength
    return {"flux": flux, "label": label}


def _make_splits(tmp_path: Path, n: int = 30) -> Path:
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    examples = [_good_example(i % 2) for i in range(n)]
    _write_split(split_dir / "val.json", examples, "val")
    _write_split(split_dir / "test.json", examples, "test")
    return split_dir


def _dummy_model_fn_pass(fluxes: list[list[float]]) -> list[float]:
    """Returns high prob for transit-like snippets (label=1 examples have dip)."""
    probs = []
    for flux in fluxes:
        mid = flux[100]
        # Dipped flux (label=1) → high probability
        probs.append(max(0.01, min(0.99, 1.0 - mid * 0.9)))
    return probs


def _dummy_model_fn_random(fluxes: list[list[float]]) -> list[float]:
    """Returns 0.5 for all inputs (poor model)."""
    return [0.5] * len(fluxes)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


class TestAucRoc:
    def test_perfect_auc(self) -> None:
        y_true = [1, 1, 0, 0]
        y_score = [0.9, 0.8, 0.2, 0.1]
        assert _auc_roc(y_true, y_score) == pytest.approx(1.0, abs=0.01)

    def test_worst_auc_near_zero(self) -> None:
        # Inverted classifier: positives get low scores
        y_true = [1, 1, 0, 0]
        y_score = [0.1, 0.2, 0.8, 0.9]
        assert _auc_roc(y_true, y_score) == pytest.approx(0.0, abs=0.01)

    def test_all_same_class_returns_half(self) -> None:
        y_true = [1, 1, 1]
        y_score = [0.9, 0.8, 0.7]
        assert _auc_roc(y_true, y_score) == 0.5

    def test_empty_returns_half(self) -> None:
        assert _auc_roc([], []) == 0.5

    def test_tied_scores_return_half(self) -> None:
        y_true = [1, 0, 1, 0]
        y_score = [0.5, 0.5, 0.5, 0.5]
        assert _auc_roc(y_true, y_score) == pytest.approx(0.5, abs=1e-9)

    def test_ties_are_order_independent(self) -> None:
        y_true_a = [1, 1, 0, 0]
        y_true_b = list(reversed(y_true_a))
        y_score = [0.5, 0.5, 0.5, 0.5]
        assert _auc_roc(y_true_a, y_score) == pytest.approx(
            _auc_roc(y_true_b, y_score), abs=1e-9
        )


class TestBestF1Threshold:
    def test_finds_good_threshold(self) -> None:
        y_true = [1, 1, 0, 0]
        y_score = [0.9, 0.8, 0.2, 0.1]
        t, f1 = _best_f1_threshold(y_true, y_score)
        assert f1 == pytest.approx(1.0, abs=0.01)
        assert 0.0 <= t <= 1.0

    def test_returns_tuple(self) -> None:
        y_true = [1, 0]
        y_score = [0.6, 0.4]
        result = _best_f1_threshold(y_true, y_score)
        assert len(result) == 2


class TestBrier:
    def test_perfect_brier_zero(self) -> None:
        y_true = [1, 0]
        y_score = [1.0, 0.0]
        assert _brier(y_true, y_score) == pytest.approx(0.0, abs=1e-7)

    def test_worst_brier_one(self) -> None:
        y_true = [1, 0]
        y_score = [0.0, 1.0]
        assert _brier(y_true, y_score) == pytest.approx(1.0, abs=1e-7)


class TestEce:
    def test_ece_miscalibrated_nonzero(self) -> None:
        # All positives, predictions all 0.1 → large calibration error
        y_true = [1, 1, 1, 1]
        y_score = [0.1, 0.1, 0.1, 0.1]
        ece = _ece(y_true, y_score)
        assert ece > 0.5

    def test_empty_returns_zero(self) -> None:
        assert _ece([], []) == 0.0


class TestFitPlatt:
    def test_platt_converges(self) -> None:
        y_true = [1, 1, 0, 0]
        y_prob = [0.9, 0.8, 0.2, 0.1]
        a, b = _fit_platt(y_true, y_prob)
        assert isinstance(a, float)
        assert isinstance(b, float)

    def test_apply_platt_in_range(self) -> None:
        result = _apply_platt(0.5, 1.0, 0.0)
        assert 0.0 < result < 1.0


class TestFitTemperature:
    def test_temperature_converges(self) -> None:
        y_true = [1, 1, 0, 0]
        y_prob = [0.9, 0.8, 0.2, 0.1]
        t = _fit_temperature(y_true, y_prob)
        assert isinstance(t, float)
        assert t > 0.0

    def test_discriminating_model_sharpens(self) -> None:
        # Any model that correctly ranks pos > neg has T < 1: sharpening reduces NLL
        y_true = [1, 1, 0, 0]
        y_prob = [0.9, 0.85, 0.15, 0.1]
        t = _fit_temperature(y_true, y_prob)
        # T is clamped at 0.1 minimum; for a discriminating model T <= 1.0
        assert 0.1 <= t <= 1.0

    def test_apply_temperature_identity_at_one(self) -> None:
        # T=1 should be nearly identity (only clamping differs)
        result = _apply_temperature(0.7, 1.0)
        assert abs(result - 0.7) < 1e-4

    def test_apply_temperature_in_range(self) -> None:
        for raw in [0.1, 0.5, 0.9]:
            result = _apply_temperature(raw, 1.5)
            assert 0.0 < result < 1.0

    def test_high_temperature_softens(self) -> None:
        # T > 1 should pull probabilities toward 0.5
        raw = 0.9
        result = _apply_temperature(raw, 3.0)
        assert result < raw  # moves toward 0.5

    def test_low_temperature_sharpens(self) -> None:
        # T < 1 should push probabilities away from 0.5
        raw = 0.7
        result = _apply_temperature(raw, 0.5)
        assert result > raw  # moves away from 0.5


class TestComputeMetrics:
    def test_returns_cnn_eval_metrics(self) -> None:
        y_true = [1, 1, 0, 0]
        y_score = [0.9, 0.8, 0.2, 0.1]
        m = _compute_metrics(y_true, y_score)
        assert isinstance(m, CnnEvalMetrics)
        assert m.n == 4
        assert 0.0 <= m.auc <= 1.0
        assert 0.0 <= m.f1 <= 1.0
        assert 0.0 <= m.brier <= 1.0
        assert 0.0 <= m.ece <= 1.0


# ---------------------------------------------------------------------------
# evaluate_cnn_checkpoint
# ---------------------------------------------------------------------------


class TestEvaluateCnnCheckpoint:
    def test_missing_val_split_returns_missing_flag(self, tmp_path: Path) -> None:
        split_dir = tmp_path / "splits"
        split_dir.mkdir()
        _write_split(split_dir / "test.json", [_good_example(0)])
        # no val.json
        result = evaluate_cnn_checkpoint(
            split_dir,
            tmp_path / "fake.pt",
            model_fn=_dummy_model_fn_pass,
        )
        assert result.flag == "MISSING_SPLIT"
        assert not result.passed

    def test_missing_test_split_returns_missing_flag(self, tmp_path: Path) -> None:
        split_dir = tmp_path / "splits"
        split_dir.mkdir()
        _write_split(split_dir / "val.json", [_good_example(0)])
        result = evaluate_cnn_checkpoint(
            split_dir,
            tmp_path / "fake.pt",
            model_fn=_dummy_model_fn_pass,
        )
        assert result.flag == "MISSING_SPLIT"

    def test_model_fn_error_returns_load_error(self, tmp_path: Path) -> None:
        split_dir = _make_splits(tmp_path)

        def _bad_fn(fluxes: list[list[float]]) -> list[float]:
            raise RuntimeError("simulated error")

        result = evaluate_cnn_checkpoint(
            split_dir, tmp_path / "fake.pt", model_fn=_bad_fn
        )
        assert result.flag == "LOAD_ERROR"

    def test_missing_flux_fails_closed(self, tmp_path: Path) -> None:
        split_dir = _make_splits(tmp_path, n=40)
        data = json.loads((split_dir / "val.json").read_text())
        del data["examples"][0]["flux"]
        (split_dir / "val.json").write_text(json.dumps(data))

        result = evaluate_cnn_checkpoint(
            split_dir,
            tmp_path / "fake.pt",
            model_fn=_dummy_model_fn_pass,
        )
        assert result.flag == "INVALID_SPLIT"
        assert not result.passed

    def test_non_binary_label_fails_closed(self, tmp_path: Path) -> None:
        split_dir = _make_splits(tmp_path, n=40)
        data = json.loads((split_dir / "val.json").read_text())
        data["examples"][0]["label"] = 2
        (split_dir / "val.json").write_text(json.dumps(data))

        result = evaluate_cnn_checkpoint(
            split_dir,
            tmp_path / "fake.pt",
            model_fn=_dummy_model_fn_pass,
        )
        assert result.flag == "INVALID_SPLIT"
        assert not result.passed

    def test_string_label_fails_closed(self, tmp_path: Path) -> None:
        split_dir = _make_splits(tmp_path, n=40)
        data = json.loads((split_dir / "val.json").read_text())
        data["examples"][0]["label"] = "1"
        (split_dir / "val.json").write_text(json.dumps(data))

        result = evaluate_cnn_checkpoint(
            split_dir,
            tmp_path / "fake.pt",
            model_fn=_dummy_model_fn_pass,
        )
        assert result.flag == "INVALID_SPLIT"
        assert not result.passed

    def test_non_finite_flux_fails_closed(self, tmp_path: Path) -> None:
        split_dir = _make_splits(tmp_path, n=40)
        data = json.loads((split_dir / "test.json").read_text())
        data["examples"][0]["flux"][0] = math.nan
        (split_dir / "test.json").write_text(json.dumps(data))

        result = evaluate_cnn_checkpoint(
            split_dir,
            tmp_path / "fake.pt",
            model_fn=_dummy_model_fn_pass,
        )
        assert result.flag == "INVALID_SPLIT"
        assert not result.passed

    def test_non_numeric_flux_fails_closed(self, tmp_path: Path) -> None:
        split_dir = _make_splits(tmp_path, n=40)
        data = json.loads((split_dir / "test.json").read_text())
        data["examples"][0]["flux"][0] = "1.0"
        (split_dir / "test.json").write_text(json.dumps(data))

        result = evaluate_cnn_checkpoint(
            split_dir,
            tmp_path / "fake.pt",
            model_fn=_dummy_model_fn_pass,
        )
        assert result.flag == "INVALID_SPLIT"
        assert not result.passed

    def test_invalid_json_fails_closed(self, tmp_path: Path) -> None:
        split_dir = _make_splits(tmp_path, n=40)
        (split_dir / "val.json").write_text("{")

        result = evaluate_cnn_checkpoint(
            split_dir,
            tmp_path / "fake.pt",
            model_fn=_dummy_model_fn_pass,
        )
        assert result.flag == "INVALID_SPLIT"
        assert not result.passed

    def test_wrong_prediction_count_fails_closed(self, tmp_path: Path) -> None:
        split_dir = _make_splits(tmp_path, n=40)
        result = evaluate_cnn_checkpoint(
            split_dir,
            tmp_path / "fake.pt",
            model_fn=lambda fluxes: [0.5] * (len(fluxes) - 1),
        )
        assert result.flag == "INVALID_PREDICTIONS"
        assert not result.passed

    def test_non_finite_prediction_fails_closed(self, tmp_path: Path) -> None:
        split_dir = _make_splits(tmp_path, n=40)

        def _nan_model(fluxes: list[list[float]]) -> list[float]:
            return [math.nan if i == 0 else 0.5 for i, _ in enumerate(fluxes)]

        result = evaluate_cnn_checkpoint(
            split_dir,
            tmp_path / "fake.pt",
            model_fn=_nan_model,
        )
        assert result.flag == "INVALID_PREDICTIONS"
        assert not result.passed

    def test_out_of_range_prediction_fails_closed(self, tmp_path: Path) -> None:
        split_dir = _make_splits(tmp_path, n=40)
        result = evaluate_cnn_checkpoint(
            split_dir,
            tmp_path / "fake.pt",
            model_fn=lambda fluxes: [1.2] * len(fluxes),
        )
        assert result.flag == "INVALID_PREDICTIONS"
        assert not result.passed

    def test_string_prediction_fails_closed(self, tmp_path: Path) -> None:
        split_dir = _make_splits(tmp_path, n=40)
        result = evaluate_cnn_checkpoint(
            split_dir,
            tmp_path / "fake.pt",
            model_fn=lambda fluxes: ["0.5"] * len(fluxes),  # type: ignore[list-item]
        )
        assert result.flag == "INVALID_PREDICTIONS"
        assert not result.passed

    def test_pass_with_good_model(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import Skills.evaluate_cnn_checkpoint as mod

        split_dir = _make_splits(tmp_path, n=40)
        monkeypatch.setattr(mod, "_apply_temperature", lambda raw, t: raw)
        result = evaluate_cnn_checkpoint(
            split_dir,
            tmp_path / "fake.pt",
            gate_auc=0.50,
            gate_f1=0.50,
            model_fn=_dummy_model_fn_pass,
        )
        assert result.flag == "PASS"
        assert result.passed

    def test_fail_with_random_model(self, tmp_path: Path) -> None:
        split_dir = _make_splits(tmp_path, n=40)
        result = evaluate_cnn_checkpoint(
            split_dir,
            tmp_path / "fake.pt",
            gate_auc=0.90,
            gate_f1=0.90,
            model_fn=_dummy_model_fn_random,
        )
        assert result.flag == "FAIL"
        assert not result.passed

    def test_result_has_all_metrics(self, tmp_path: Path) -> None:
        split_dir = _make_splits(tmp_path)
        result = evaluate_cnn_checkpoint(
            split_dir,
            tmp_path / "fake.pt",
            gate_auc=0.5,
            gate_f1=0.5,
            model_fn=_dummy_model_fn_pass,
        )
        assert result.val_metrics_raw is not None
        assert result.test_metrics_raw is not None
        assert result.test_metrics_cal is not None

    def test_temp_param_set(self, tmp_path: Path) -> None:
        split_dir = _make_splits(tmp_path)
        result = evaluate_cnn_checkpoint(
            split_dir,
            tmp_path / "fake.pt",
            gate_auc=0.5,
            gate_f1=0.5,
            model_fn=_dummy_model_fn_pass,
        )
        assert isinstance(result.temp, float)
        assert result.temp > 0.0

    def test_calibration_json_written_on_pass(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import Skills.evaluate_cnn_checkpoint as mod

        split_dir = _make_splits(tmp_path, n=40)
        cal_path = tmp_path / "calibration.json"
        monkeypatch.setattr(mod, "_apply_temperature", lambda raw, t: raw)
        evaluate_cnn_checkpoint(
            split_dir,
            tmp_path / "fake.pt",
            gate_auc=0.5,
            gate_f1=0.5,
            output_calibration=cal_path,
            model_fn=_dummy_model_fn_pass,
        )
        assert cal_path.exists()
        cal = json.loads(cal_path.read_text())
        assert "temperature" in cal
        assert cal.get("method") == "temperature"
        assert cal.get("flag") == "OK"

    def test_calibration_worsening_blocks_promotion(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import Skills.evaluate_cnn_checkpoint as mod

        split_dir = _make_splits(tmp_path, n=40)
        cal_path = tmp_path / "calibration.json"
        monkeypatch.setattr(mod, "_apply_temperature", lambda raw, t: 0.5)

        result = evaluate_cnn_checkpoint(
            split_dir,
            tmp_path / "fake.pt",
            gate_auc=0.5,
            gate_f1=0.5,
            output_calibration=cal_path,
            model_fn=lambda fluxes: [
                0.9 if flux[100] < 1.0 else 0.1 for flux in fluxes
            ],
        )

        assert result.flag == "FAIL"
        assert not result.passed
        assert result.test_metrics_raw is not None
        assert result.test_metrics_cal is not None
        assert result.test_metrics_cal.brier > result.test_metrics_raw.brier
        assert not cal_path.exists()

    def test_calibration_json_not_written_on_fail(self, tmp_path: Path) -> None:
        split_dir = _make_splits(tmp_path, n=40)
        cal_path = tmp_path / "calibration.json"
        evaluate_cnn_checkpoint(
            split_dir,
            tmp_path / "fake.pt",
            gate_auc=0.99,
            gate_f1=0.99,
            output_calibration=cal_path,
            model_fn=_dummy_model_fn_random,
        )
        assert not cal_path.exists()

    def test_gates_reflected_in_result(self, tmp_path: Path) -> None:
        split_dir = _make_splits(tmp_path)
        result = evaluate_cnn_checkpoint(
            split_dir,
            tmp_path / "fake.pt",
            gate_auc=0.77,
            gate_f1=0.66,
            model_fn=_dummy_model_fn_pass,
        )
        assert result.gate_auc == pytest.approx(0.77)
        assert result.gate_f1 == pytest.approx(0.66)

    def test_evaluated_at_iso_format(self, tmp_path: Path) -> None:
        split_dir = _make_splits(tmp_path)
        result = evaluate_cnn_checkpoint(
            split_dir,
            tmp_path / "fake.pt",
            model_fn=_dummy_model_fn_pass,
        )
        assert "T" in result.evaluated_at  # ISO 8601


# ---------------------------------------------------------------------------
# Ensemble support
# ---------------------------------------------------------------------------


class TestEnsembleSupport:
    def test_model_fn_takes_priority_over_checkpoint_paths(self, tmp_path: Path) -> None:
        split_dir = _make_splits(tmp_path, n=30)
        result = evaluate_cnn_checkpoint(
            split_dir,
            tmp_path / "fake.pt",
            gate_auc=0.5,
            gate_f1=0.5,
            model_fn=_dummy_model_fn_pass,
            checkpoint_paths=[tmp_path / "a.pt", tmp_path / "b.pt"],
        )
        assert result.flag in {"PASS", "FAIL"}
        assert result.val_metrics_raw is not None

    def test_single_checkpoint_path_uses_single_mode(self, tmp_path: Path) -> None:
        split_dir = _make_splits(tmp_path, n=30)
        result = evaluate_cnn_checkpoint(
            split_dir,
            tmp_path / "fake.pt",
            gate_auc=0.5,
            gate_f1=0.5,
            model_fn=_dummy_model_fn_pass,
            checkpoint_paths=[tmp_path / "only.pt"],
        )
        assert result.val_metrics_raw is not None

    def test_ensemble_infer_averages_predictions(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import Skills.evaluate_cnn_checkpoint as mod

        call_count = {"n": 0}

        def _fake_infer(
            fluxes: list[list[float]], ckpt: Path, cfg: Path
        ) -> list[float]:
            call_count["n"] += 1
            return [0.4 if i % 2 == 0 else 0.6 for i in range(len(fluxes))]

        monkeypatch.setattr(mod, "_torch_infer", _fake_infer)

        fluxes = [[1.0] * 201 for _ in range(10)]
        paths = [tmp_path / "a.pt", tmp_path / "b.pt", tmp_path / "c.pt"]
        result = mod._ensemble_infer(fluxes, paths, tmp_path / "config.json")

        assert call_count["n"] == 3
        assert len(result) == 10
        # Average of identical predictions is the same prediction
        assert abs(result[0] - 0.4) < 1e-6

    def test_ensemble_evaluate_with_mocked_infer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import Skills.evaluate_cnn_checkpoint as mod

        def _fake_infer(
            fluxes: list[list[float]], ckpt: Path, cfg: Path
        ) -> list[float]:
            return _dummy_model_fn_pass(fluxes)

        monkeypatch.setattr(mod, "_torch_infer", _fake_infer)

        split_dir = _make_splits(tmp_path, n=30)
        (tmp_path / "a.pt").touch()
        (tmp_path / "b.pt").touch()
        result = evaluate_cnn_checkpoint(
            split_dir,
            tmp_path / "a.pt",
            gate_auc=0.5,
            gate_f1=0.5,
            checkpoint_paths=[tmp_path / "a.pt", tmp_path / "b.pt"],
        )
        assert result.val_metrics_raw is not None
        assert result.flag in {"PASS", "FAIL"}


# ---------------------------------------------------------------------------
# format_eval_result
# ---------------------------------------------------------------------------


class TestFormatEvalResult:
    def test_format_contains_flag(self, tmp_path: Path) -> None:
        split_dir = _make_splits(tmp_path)
        result = evaluate_cnn_checkpoint(
            split_dir,
            tmp_path / "fake.pt",
            gate_auc=0.5,
            gate_f1=0.5,
            model_fn=_dummy_model_fn_pass,
        )
        text = format_eval_result(result)
        assert "PASS" in text or "FAIL" in text

    def test_format_contains_auc(self, tmp_path: Path) -> None:
        split_dir = _make_splits(tmp_path)
        result = evaluate_cnn_checkpoint(
            split_dir,
            tmp_path / "fake.pt",
            gate_auc=0.5,
            gate_f1=0.5,
            model_fn=_dummy_model_fn_pass,
        )
        text = format_eval_result(result)
        assert "AUC" in text

    def test_format_missing_split(self) -> None:
        result = CnnEvalResult(
            val_metrics_raw=None,
            test_metrics_raw=None,
            test_metrics_cal=None,
            temp=1.0,
            gate_auc=0.85,
            gate_f1=0.80,
            passed=False,
            flag="MISSING_SPLIT",
            evaluated_at="2026-01-01T00:00:00+00:00",
        )
        text = format_eval_result(result)
        assert "MISSING_SPLIT" in text
