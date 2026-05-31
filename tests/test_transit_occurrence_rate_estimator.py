"""Tests for Skills/transit_occurrence_rate_estimator.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from transit_occurrence_rate_estimator import (
    estimate_occurrence_rate,
    format_occurrence_rate_result,
)


class TestOccurrenceRateEstimator:
    def test_basic_rate(self) -> None:
        r = estimate_occurrence_rate(10, 1000, 0.8, 0.1)
        assert r.flag == "OK"
        assert r.occurrence_rate > 0.0

    def test_earth_like_rate(self) -> None:
        # ~1 in 10 stars has an Earth analog: N=100, N★=1000, ε=0.5, p=0.005
        r = estimate_occurrence_rate(100, 1000, 0.5, 0.005)
        # eta = 100 / (1000 * 0.5 * 0.005) = 100/2.5 = 40
        assert abs(r.occurrence_rate - 40.0) < 1.0

    def test_invalid_detections(self) -> None:
        r = estimate_occurrence_rate(-1, 1000, 0.8, 0.1)
        assert r.flag == "INVALID_DETECTIONS"

    def test_invalid_n_stars(self) -> None:
        r = estimate_occurrence_rate(10, 0, 0.8, 0.1)
        assert r.flag == "INVALID_N_STARS"

    def test_invalid_efficiency_zero(self) -> None:
        r = estimate_occurrence_rate(10, 1000, 0.0, 0.1)
        assert r.flag == "INVALID_EFFICIENCY"

    def test_invalid_efficiency_over_one(self) -> None:
        r = estimate_occurrence_rate(10, 1000, 1.5, 0.1)
        assert r.flag == "INVALID_EFFICIENCY"

    def test_invalid_transit_prob(self) -> None:
        r = estimate_occurrence_rate(10, 1000, 0.8, 0.0)
        assert r.flag == "INVALID_TRANSIT_PROB"

    def test_upper_bound_gt_rate(self) -> None:
        r = estimate_occurrence_rate(5, 1000, 0.8, 0.1)
        assert r.rate_upper_1sigma > r.occurrence_rate

    def test_lower_bound_lt_rate(self) -> None:
        r = estimate_occurrence_rate(5, 1000, 0.8, 0.1)
        assert r.rate_lower_1sigma <= r.occurrence_rate

    def test_zero_detections(self) -> None:
        r = estimate_occurrence_rate(0, 1000, 0.8, 0.1)
        assert r.flag == "OK"
        assert r.occurrence_rate == 0.0
        assert r.rate_upper_1sigma > 0.0

    def test_rate_float(self) -> None:
        r = estimate_occurrence_rate(10, 1000, 0.8, 0.1)
        assert isinstance(r.occurrence_rate, float)

    def test_format_returns_string(self) -> None:
        r = estimate_occurrence_rate(10, 1000, 0.8, 0.1)
        s = format_occurrence_rate_result(r)
        assert isinstance(s, str)
        assert "Occurrence" in s
