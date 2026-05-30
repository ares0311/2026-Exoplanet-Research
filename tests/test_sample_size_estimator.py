"""Tests for Skills/sample_size_estimator.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from sample_size_estimator import estimate_sample_size, format_sample_size_result


class TestEstimateSampleSize:
    def test_already_achieved(self) -> None:
        r = estimate_sample_size([(500, 0.85), (1000, 0.90)], target_auc=0.88)
        assert r.flag == "ALREADY_ACHIEVED"
        assert r.achievable is True

    def test_estimate_future(self) -> None:
        r = estimate_sample_size([(500, 0.80), (1000, 0.85)], target_auc=0.95)
        assert r.flag == "OK"
        assert r.estimated_n > 1000

    def test_insufficient_data(self) -> None:
        r = estimate_sample_size([], target_auc=0.90)
        assert r.flag == "INSUFFICIENT_DATA"

    def test_invalid_target_auc_zero(self) -> None:
        r = estimate_sample_size([(500, 0.80)], target_auc=0.0)
        assert r.flag == "INVALID_TARGET_AUC"

    def test_invalid_target_auc_gt1(self) -> None:
        r = estimate_sample_size([(500, 0.80)], target_auc=1.1)
        assert r.flag == "INVALID_TARGET_AUC"

    def test_target_exceeds_max_auc(self) -> None:
        r = estimate_sample_size([(500, 0.80)], target_auc=0.99, max_auc=0.98)
        assert r.flag == "TARGET_EXCEEDS_MAX_AUC"

    def test_estimated_n_positive(self) -> None:
        r = estimate_sample_size([(500, 0.80), (1000, 0.85)], target_auc=0.95)
        assert r.estimated_n > 0

    def test_current_best_auc_stored(self) -> None:
        r = estimate_sample_size([(500, 0.80), (1000, 0.87)], target_auc=0.95)
        assert abs(r.current_best_auc - 0.87) < 1e-9

    def test_larger_n_gives_higher_estimate(self) -> None:
        r1 = estimate_sample_size([(500, 0.80)], target_auc=0.95)
        r2 = estimate_sample_size([(500, 0.75)], target_auc=0.95)
        assert r2.estimated_n >= r1.estimated_n

    def test_achievable_true_for_ok(self) -> None:
        r = estimate_sample_size([(500, 0.80), (1000, 0.85)], target_auc=0.95)
        assert r.achievable is True

    def test_single_point(self) -> None:
        r = estimate_sample_size([(1000, 0.82)], target_auc=0.95)
        assert r.flag in ("OK", "ALREADY_ACHIEVED", "EXTRAPOLATION_FAILED")

    def test_format_returns_string(self) -> None:
        r = estimate_sample_size([(500, 0.80), (1000, 0.85)], target_auc=0.95)
        s = format_sample_size_result(r)
        assert isinstance(s, str)
        assert "Sample" in s
