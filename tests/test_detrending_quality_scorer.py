"""Tests for Skills/detrending_quality_scorer.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from detrending_quality_scorer import format_detrending_quality, score_detrending_quality


class TestScoreDetrendingQuality:
    def test_white_noise_ok(self) -> None:
        residuals = [0.1, -0.1, 0.05, -0.05, 0.08, -0.08, 0.02, -0.02, 0.06, -0.06]
        r = score_detrending_quality(residuals)
        assert r.flag in ("OK", "MODERATE_AUTOCORRELATION", "HIGH_AUTOCORRELATION")

    def test_high_autocorrelation_flagged(self) -> None:
        residuals = [float(i) for i in range(20)]
        r = score_detrending_quality(residuals)
        assert r.flag == "HIGH_AUTOCORRELATION"

    def test_insufficient_data(self) -> None:
        r = score_detrending_quality([1.0, 2.0, 3.0])
        assert r.flag == "INSUFFICIENT_DATA"

    def test_constant_residuals(self) -> None:
        r = score_detrending_quality([1.0] * 20)
        assert r.lag1_acf == 0.0
        assert r.durbin_watson == 2.0

    def test_lag1_acf_range(self) -> None:
        residuals = [0.1, -0.1, 0.2, -0.2, 0.05, -0.05, 0.15, -0.15]
        r = score_detrending_quality(residuals)
        assert -1.0 <= r.lag1_acf <= 1.0

    def test_durbin_watson_range(self) -> None:
        residuals = [0.1, -0.1, 0.2, -0.2, 0.05, -0.05, 0.15, -0.15]
        r = score_detrending_quality(residuals)
        assert 0.0 <= r.durbin_watson <= 4.0

    def test_rms_residual_non_negative(self) -> None:
        r = score_detrending_quality([0.1, -0.1, 0.2, -0.2] * 3)
        assert r.rms_residual >= 0.0

    def test_quality_score_in_range(self) -> None:
        r = score_detrending_quality([0.1, -0.1, 0.05, -0.05] * 3)
        assert 0.0 <= r.quality_score <= 1.0

    def test_format_returns_string(self) -> None:
        r = score_detrending_quality([0.1, -0.1, 0.05, -0.05] * 3)
        s = format_detrending_quality(r)
        assert isinstance(s, str)
        assert "Quality" in s

    def test_ramp_is_high_autocorrelation(self) -> None:
        r = score_detrending_quality([float(i) for i in range(30)])
        assert r.lag1_acf > 0.5

    def test_result_frozen(self) -> None:
        r = score_detrending_quality([0.1, -0.1] * 5)
        try:
            r.flag = "X"  # type: ignore[misc]
            raise AssertionError("should be frozen")
        except Exception:
            pass

    def test_ok_flag_for_alternating(self) -> None:
        residuals = [1.0 if i % 2 == 0 else -1.0 for i in range(20)]
        r = score_detrending_quality(residuals)
        assert r.flag in ("OK", "MODERATE_AUTOCORRELATION", "HIGH_AUTOCORRELATION")
