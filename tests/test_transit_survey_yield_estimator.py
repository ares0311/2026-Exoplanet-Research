"""Tests for Skills/transit_survey_yield_estimator.py"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from transit_survey_yield_estimator import (
    estimate_survey_yield,
    format_survey_yield_result,
)


class TestTransitSurveyYieldEstimator:
    def test_basic_ok(self) -> None:
        r = estimate_survey_yield(1000, 365.0)
        assert r.flag == "OK"
        assert r.expected_detections >= 0

    def test_more_stars_more_detections(self) -> None:
        r_few = estimate_survey_yield(100, 365.0)
        r_many = estimate_survey_yield(10000, 365.0)
        assert r_many.expected_detections >= r_few.expected_detections

    def test_geometric_efficiency_between_0_and_1(self) -> None:
        r = estimate_survey_yield(1000, 365.0)
        assert 0.0 <= r.geometric_efficiency <= 1.0

    def test_window_efficiency_between_0_and_1(self) -> None:
        r = estimate_survey_yield(1000, 365.0)
        assert 0.0 <= r.window_efficiency <= 1.0

    def test_combined_efficiency_equals_product(self) -> None:
        r = estimate_survey_yield(1000, 365.0)
        expected = r.geometric_efficiency * r.window_efficiency
        assert abs(r.combined_efficiency - expected) < 1e-9

    def test_detections_per_period_bin_length(self) -> None:
        r = estimate_survey_yield(1000, 365.0)
        assert len(r.detections_per_period_bin) == 6

    def test_short_period_bins_non_negative(self) -> None:
        r = estimate_survey_yield(1000, 365.0)
        assert all(y >= 0 for y in r.detections_per_period_bin)

    def test_invalid_n_stars(self) -> None:
        r = estimate_survey_yield(0, 365.0)
        assert r.flag == "INVALID_N_STARS"
        assert math.isnan(r.expected_detections)

    def test_invalid_baseline(self) -> None:
        r = estimate_survey_yield(1000, 0.0)
        assert r.flag == "INVALID_BASELINE"

    def test_invalid_eta(self) -> None:
        r = estimate_survey_yield(1000, 365.0, eta_earth=0.0)
        assert r.flag == "INVALID_ETA"

    def test_result_is_frozen(self) -> None:
        r = estimate_survey_yield(1000, 365.0)
        try:
            r.expected_detections = 0.0  # type: ignore[misc]
            raise AssertionError("frozen dataclass should not allow setattr")
        except Exception:
            pass

    def test_format_ok(self) -> None:
        r = estimate_survey_yield(1000, 365.0)
        s = format_survey_yield_result(r)
        assert "detection" in s.lower() or "yield" in s.lower()

    def test_format_error(self) -> None:
        r = estimate_survey_yield(0, 365.0)
        s = format_survey_yield_result(r)
        assert "INVALID_N_STARS" in s
