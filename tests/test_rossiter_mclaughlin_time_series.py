"""Tests for Skills/rossiter_mclaughlin_time_series.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from rossiter_mclaughlin_time_series import (
    compute_rm_time_series,
    format_rm_time_series_result,
)


class TestComputeRmTimeSeries:
    def test_ok_flag(self) -> None:
        r = compute_rm_time_series(5000.0, 0.01)
        assert r.flag == "OK"

    def test_n_phases_correct(self) -> None:
        r = compute_rm_time_series(5000.0, 0.01, n_points=30)
        assert len(r.phases) == 30

    def test_n_rv_points_correct(self) -> None:
        r = compute_rm_time_series(5000.0, 0.01, n_points=30)
        assert len(r.rv_anomaly_ms) == 30

    def test_aligned_orbit_antisymmetric(self) -> None:
        r = compute_rm_time_series(5000.0, 0.01, lambda_deg=0.0, n_points=100)
        assert abs(r.rv_anomaly_ms[0] + r.rv_anomaly_ms[-1]) < 1.0

    def test_max_anomaly_finite(self) -> None:
        r = compute_rm_time_series(5000.0, 0.01)
        assert r.max_anomaly_ms < float("inf")

    def test_min_anomaly_finite(self) -> None:
        r = compute_rm_time_series(5000.0, 0.01)
        assert r.min_anomaly_ms > -float("inf")

    def test_larger_vsini_larger_anomaly(self) -> None:
        r1 = compute_rm_time_series(1000.0, 0.01)
        r2 = compute_rm_time_series(10000.0, 0.01)
        assert abs(r2.max_anomaly_ms) > abs(r1.max_anomaly_ms)

    def test_lambda_deg_preserved(self) -> None:
        r = compute_rm_time_series(5000.0, 0.01, lambda_deg=30.0)
        assert r.lambda_deg == 30.0

    def test_invalid_vsini(self) -> None:
        r = compute_rm_time_series(0.0, 0.01)
        assert r.flag == "INVALID_VSINI"

    def test_invalid_depth_zero(self) -> None:
        r = compute_rm_time_series(5000.0, 0.0)
        assert r.flag == "INVALID_DEPTH"

    def test_invalid_depth_ge_one(self) -> None:
        r = compute_rm_time_series(5000.0, 1.0)
        assert r.flag == "INVALID_DEPTH"

    def test_invalid_impact(self) -> None:
        r = compute_rm_time_series(5000.0, 0.01, impact_parameter=1.0)
        assert r.flag == "INVALID_IMPACT"

    def test_format_returns_string(self) -> None:
        r = compute_rm_time_series(5000.0, 0.01)
        s = format_rm_time_series_result(r)
        assert isinstance(s, str)
        assert r.flag in s
