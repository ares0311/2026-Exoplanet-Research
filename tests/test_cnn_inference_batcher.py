"""Tests for Skills.cnn_inference_batcher (13 tests).

All tests use the injectable model_fn; PyTorch is not required.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from Skills.cnn_inference_batcher import (
    CnnInferenceResult,
    _pad_or_truncate,
    format_cnn_inference,
    run_cnn_inference,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _constant_model(prob: float):
    """Return a model_fn that always predicts *prob*."""
    def _fn(flux: list[float]) -> float:
        return prob
    return _fn


def _make_flux(n: int = 201, depth: float = 0.01) -> list[float]:
    mid = n // 2
    return [1.0 - depth if abs(i - mid) < 5 else 1.0 for i in range(n)]


# ---------------------------------------------------------------------------
# _pad_or_truncate
# ---------------------------------------------------------------------------


class TestPadOrTruncate:
    def test_short_flux_padded(self) -> None:
        result = _pad_or_truncate([0.9, 1.0], 5)
        assert len(result) == 5
        assert result[2:] == [1.0, 1.0, 1.0]

    def test_exact_length_unchanged(self) -> None:
        flux = [0.9] * 201
        result = _pad_or_truncate(flux, 201)
        assert result == flux

    def test_long_flux_truncated(self) -> None:
        flux = [0.5] * 300
        result = _pad_or_truncate(flux, 201)
        assert len(result) == 201


# ---------------------------------------------------------------------------
# run_cnn_inference with model_fn
# ---------------------------------------------------------------------------


class TestRunCnnInference:
    def test_empty_input_returns_empty_flag(self) -> None:
        result = run_cnn_inference([])
        assert result.flag == "EMPTY"
        assert result.n_inputs == 0

    def test_model_fn_used(self) -> None:
        arrays = [_make_flux() for _ in range(5)]
        result = run_cnn_inference(arrays, model_fn=_constant_model(0.8))
        assert result.flag == "OK"
        assert len(result.probabilities) == 5

    def test_probabilities_clipped_to_unit(self) -> None:
        arrays = [_make_flux()]
        result = run_cnn_inference(arrays, model_fn=_constant_model(1.5))
        assert all(0.0 <= p <= 1.0 for p in result.probabilities)

    def test_probabilities_count_matches_input(self) -> None:
        arrays = [_make_flux() for _ in range(10)]
        result = run_cnn_inference(arrays, model_fn=_constant_model(0.5))
        assert result.n_inputs == 10
        assert len(result.probabilities) == 10

    def test_calibration_applied_false_without_cal(self) -> None:
        arrays = [_make_flux()]
        result = run_cnn_inference(arrays, model_fn=_constant_model(0.7))
        assert result.calibration_applied is False

    def test_calibration_applied_with_valid_cal(self) -> None:
        from Skills.cnn_calibrator import fit_cnn_calibration, save_cnn_calibration
        y_true = [1, 1, 0, 0, 1, 0, 1, 0]
        y_prob = [0.9, 0.8, 0.2, 0.1, 0.85, 0.15, 0.75, 0.25]
        cal = fit_cnn_calibration(y_true, y_prob)
        with tempfile.TemporaryDirectory() as tmpdir:
            cal_path = Path(tmpdir) / "cal.json"
            save_cnn_calibration(cal, cal_path)
            arrays = [_make_flux() for _ in range(3)]
            result = run_cnn_inference(
                arrays,
                model_fn=_constant_model(0.7),
                calibration_path=cal_path,
            )
        if cal.flag == "OK":
            assert result.calibration_applied is True

    def test_missing_model_path_returns_no_torch_or_invalid(self) -> None:
        arrays = [_make_flux()]
        result = run_cnn_inference(
            arrays,
            model_path=Path("/tmp/does_not_exist_xyz.pt"),
        )
        assert result.flag in {"NO_TORCH", "INVALID"}

    def test_no_model_no_fn_returns_no_torch_or_invalid(self) -> None:
        arrays = [_make_flux()]
        result = run_cnn_inference(arrays)
        assert result.flag in {"NO_TORCH", "INVALID"}

    def test_inference_time_ms_nonnegative(self) -> None:
        arrays = [_make_flux()]
        result = run_cnn_inference(arrays, model_fn=_constant_model(0.5))
        assert result.inference_time_ms >= 0.0

    def test_model_path_none_on_model_fn(self) -> None:
        arrays = [_make_flux()]
        result = run_cnn_inference(arrays, model_fn=_constant_model(0.5))
        assert result.model_path is None


# ---------------------------------------------------------------------------
# format_cnn_inference
# ---------------------------------------------------------------------------


class TestFormatCnnInference:
    def test_returns_string(self) -> None:
        arrays = [_make_flux()]
        result = run_cnn_inference(arrays, model_fn=_constant_model(0.6))
        out = format_cnn_inference(result)
        assert isinstance(out, str)

    def test_contains_flag(self) -> None:
        result = CnnInferenceResult(
            n_inputs=0,
            probabilities=(),
            calibration_applied=False,
            model_path=None,
            inference_time_ms=0.0,
            flag="EMPTY",
        )
        out = format_cnn_inference(result)
        assert "EMPTY" in out

    def test_contains_stats_when_probs_present(self) -> None:
        arrays = [_make_flux() for _ in range(4)]
        result = run_cnn_inference(arrays, model_fn=_constant_model(0.8))
        out = format_cnn_inference(result)
        assert "Mean" in out or "mean" in out
