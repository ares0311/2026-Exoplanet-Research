"""Tests for Skills/transit_detrending_quality_estimator.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from transit_detrending_quality_estimator import (
    compute_detrending_quality,
    format_detrending_quality_result,
)


def _flat_oot(n: int = 50, noise: float = 0.0001) -> list[float]:
    import random
    random.seed(42)
    return [1.0 + random.gauss(0, noise) for _ in range(n)]


class TestComputeDetrendingQuality:
    def test_ok_flag(self) -> None:
        r = compute_detrending_quality(_flat_oot(), [0.99] * 5, 10000.0)
        assert r.flag == "OK"

    def test_quality_score_between_zero_and_one(self) -> None:
        r = compute_detrending_quality(_flat_oot(), [0.99] * 5, 10000.0)
        assert 0.0 <= r.detrending_quality_score <= 1.0

    def test_grade_valid_letter(self) -> None:
        r = compute_detrending_quality(_flat_oot(), [0.99] * 5, 10000.0)
        assert r.quality_grade in ("A", "B", "C", "D", "F")

    def test_low_noise_high_snr_good_grade(self) -> None:
        oot = _flat_oot(100, noise=0.00001)
        r = compute_detrending_quality(oot, [0.99] * 5, 10000.0)
        assert r.snr_ratio > 5.0

    def test_high_noise_low_snr(self) -> None:
        oot = _flat_oot(100, noise=0.01)
        r = compute_detrending_quality(oot, [0.99] * 5, 10000.0)
        assert r.snr_ratio < 5.0

    def test_oot_rms_positive(self) -> None:
        r = compute_detrending_quality(_flat_oot(), [0.99] * 5, 10000.0)
        assert r.oot_rms_ppm > 0.0

    def test_depth_preserved(self) -> None:
        r = compute_detrending_quality(_flat_oot(), [0.99] * 5, 10000.0)
        assert r.transit_depth_ppm == 10000.0

    def test_custom_residuals_used(self) -> None:
        oot = _flat_oot(50)
        residuals = _flat_oot(50, noise=0.0001)
        r = compute_detrending_quality(oot, [0.99] * 5, 5000.0,
                                        detrended_residuals=residuals)
        assert r.flag == "OK"

    def test_insufficient_oot(self) -> None:
        r = compute_detrending_quality([1.0, 1.0, 0.999], [0.99], 1000.0)
        assert r.flag == "INSUFFICIENT_OOT_DATA"

    def test_invalid_depth(self) -> None:
        r = compute_detrending_quality(_flat_oot(), [0.99] * 5, 0.0)
        assert r.flag == "INVALID_DEPTH"

    def test_result_frozen(self) -> None:
        r = compute_detrending_quality(_flat_oot(), [0.99] * 5, 10000.0)
        try:
            r.quality_grade = "Z"  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except (AttributeError, TypeError):
            pass

    def test_format_returns_string(self) -> None:
        r = compute_detrending_quality(_flat_oot(), [0.99] * 5, 10000.0)
        s = format_detrending_quality_result(r)
        assert isinstance(s, str)
        assert r.flag in s
