"""Tests for Skills.cnn_calibrator (13 tests)."""
from __future__ import annotations

import tempfile
from pathlib import Path

from Skills.cnn_calibrator import (
    CnnCalibrationResult,
    apply_cnn_calibration,
    fit_cnn_calibration,
    format_cnn_calibration,
    load_cnn_calibration,
    save_cnn_calibration,
)

# ---------------------------------------------------------------------------
# fit_cnn_calibration
# ---------------------------------------------------------------------------


class TestFitCnnCalibration:
    def test_returns_calibration_result(self) -> None:
        y_true = [1, 1, 0, 0, 1, 0, 1, 0]
        y_prob = [0.9, 0.8, 0.2, 0.1, 0.85, 0.15, 0.75, 0.25]
        result = fit_cnn_calibration(y_true, y_prob)
        assert isinstance(result, CnnCalibrationResult)

    def test_ok_flag_on_valid_data(self) -> None:
        y_true = [1, 1, 0, 0, 1, 0, 1, 0]
        y_prob = [0.9, 0.8, 0.2, 0.1, 0.85, 0.15, 0.75, 0.25]
        result = fit_cnn_calibration(y_true, y_prob)
        assert result.flag == "OK"

    def test_method_is_platt(self) -> None:
        y_true = [1, 0, 1, 0, 1, 0, 1, 0]
        y_prob = [0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.85, 0.15]
        result = fit_cnn_calibration(y_true, y_prob)
        assert result.method == "platt"

    def test_insufficient_data_flag(self) -> None:
        result = fit_cnn_calibration([1, 0], [0.9, 0.1])
        assert result.flag == "INSUFFICIENT"

    def test_empty_input_insufficient(self) -> None:
        result = fit_cnn_calibration([], [])
        assert result.flag in {"INSUFFICIENT", "INVALID"}

    def test_all_one_class_insufficient(self) -> None:
        result = fit_cnn_calibration([1, 1, 1, 1, 1], [0.9, 0.8, 0.7, 0.85, 0.75])
        assert result.flag == "INSUFFICIENT"

    def test_brier_score_uncal_computed(self) -> None:
        y_true = [1, 1, 0, 0, 1, 0, 1, 0]
        y_prob = [0.9, 0.8, 0.2, 0.1, 0.85, 0.15, 0.75, 0.25]
        result = fit_cnn_calibration(y_true, y_prob)
        if result.flag == "OK":
            assert result.brier_score_uncal is not None
            assert 0.0 <= result.brier_score_uncal <= 1.0

    def test_mismatched_lengths_invalid(self) -> None:
        result = fit_cnn_calibration([1, 0, 1], [0.9, 0.1])
        assert result.flag == "INVALID"

    def test_platt_parameters_are_floats(self) -> None:
        y_true = [1, 1, 0, 0, 1, 0, 1, 0]
        y_prob = [0.9, 0.8, 0.2, 0.1, 0.85, 0.15, 0.75, 0.25]
        result = fit_cnn_calibration(y_true, y_prob)
        assert isinstance(result.platt_a, float)
        assert isinstance(result.platt_b, float)


# ---------------------------------------------------------------------------
# apply_cnn_calibration
# ---------------------------------------------------------------------------


class TestApplyCnnCalibration:
    def _identity_result(self) -> CnnCalibrationResult:
        """Platt A=1, B=0 leaves sigmoid(p) ≈ p for mid-range p."""
        return CnnCalibrationResult(
            method="platt", platt_a=1.0, platt_b=0.0, n_samples=10,
            brier_score_uncal=None, brier_score_cal=None,
            fitted_at="2026-01-01T00:00:00+00:00", flag="OK",
        )

    def test_output_in_unit_interval(self) -> None:
        result = self._identity_result()
        cal = apply_cnn_calibration(0.5, result)
        assert 0.0 < cal < 1.0

    def test_clipped_not_zero_or_one(self) -> None:
        result = self._identity_result()
        assert apply_cnn_calibration(0.0, result) > 0.0
        assert apply_cnn_calibration(1.0, result) < 1.0

    def test_monotone_with_positive_a(self) -> None:
        result = self._identity_result()
        assert apply_cnn_calibration(0.3, result) < apply_cnn_calibration(0.7, result)


# ---------------------------------------------------------------------------
# save_cnn_calibration / load_cnn_calibration
# ---------------------------------------------------------------------------


class TestSaveLoadCnnCalibration:
    def test_round_trip(self) -> None:
        y_true = [1, 1, 0, 0, 1, 0, 1, 0]
        y_prob = [0.9, 0.8, 0.2, 0.1, 0.85, 0.15, 0.75, 0.25]
        result = fit_cnn_calibration(y_true, y_prob)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cal.json"
            save_cnn_calibration(result, path)
            loaded = load_cnn_calibration(path)
        assert loaded.platt_a == result.platt_a
        assert loaded.platt_b == result.platt_b
        assert loaded.flag == result.flag


# ---------------------------------------------------------------------------
# format_cnn_calibration
# ---------------------------------------------------------------------------


class TestFormatCnnCalibration:
    def test_returns_string(self) -> None:
        y_true = [1, 1, 0, 0, 1, 0, 1, 0]
        y_prob = [0.9, 0.8, 0.2, 0.1, 0.85, 0.15, 0.75, 0.25]
        result = fit_cnn_calibration(y_true, y_prob)
        out = format_cnn_calibration(result)
        assert isinstance(out, str)

    def test_contains_platt(self) -> None:
        y_true = [1, 1, 0, 0, 1, 0, 1, 0]
        y_prob = [0.9, 0.8, 0.2, 0.1, 0.85, 0.15, 0.75, 0.25]
        result = fit_cnn_calibration(y_true, y_prob)
        out = format_cnn_calibration(result)
        assert "platt" in out.lower() or "Platt" in out
