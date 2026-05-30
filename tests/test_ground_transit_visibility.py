"""Tests for Skills/ground_transit_visibility.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from ground_transit_visibility import (
    check_ground_transit_visibility,
    format_ground_visibility_result,
)


class TestCheckGroundTransitVisibility:
    def test_fully_visible(self) -> None:
        # transit 2460001.1 to 2460001.2, night 2460001.0 to 2460001.5
        r = check_ground_transit_visibility(2460001.1, 2460001.2, 2460001.0, 2460001.5)
        assert r.fully_visible is True
        assert r.flag == "FULLY_VISIBLE"

    def test_partial(self) -> None:
        # transit starts before sunset
        r = check_ground_transit_visibility(2460001.0, 2460001.2, 2460001.05, 2460001.5)
        assert r.fully_visible is False
        assert r.flag == "PARTIAL"

    def test_no_overlap(self) -> None:
        r = check_ground_transit_visibility(2460001.6, 2460001.7, 2460001.0, 2460001.5)
        assert r.flag == "NO_OVERLAP"

    def test_invalid_transit_window(self) -> None:
        r = check_ground_transit_visibility(2460001.5, 2460001.2, 2460001.0, 2460001.8)
        assert r.flag == "INVALID_TRANSIT_WINDOW"

    def test_invalid_night_window(self) -> None:
        r = check_ground_transit_visibility(2460001.1, 2460001.2, 2460001.5, 2460001.1)
        assert r.flag == "INVALID_NIGHT_WINDOW"

    def test_invalid_altitude(self) -> None:
        r = check_ground_transit_visibility(2460001.1, 2460001.2, 2460001.0, 2460001.5,
                                             min_altitude_deg=-5.0)
        assert r.flag == "INVALID_ALTITUDE"

    def test_transit_duration_correct(self) -> None:
        r = check_ground_transit_visibility(2460001.0, 2460001.25, 2460000.9, 2460001.5)
        assert abs(r.transit_duration_hours - 6.0) < 1e-6

    def test_overlap_hours_correct(self) -> None:
        r = check_ground_transit_visibility(2460001.1, 2460001.2, 2460001.0, 2460001.5)
        assert abs(r.overlap_hours - 2.4) < 1e-5

    def test_format_returns_string(self) -> None:
        r = check_ground_transit_visibility(2460001.1, 2460001.2, 2460001.0, 2460001.5)
        s = format_ground_visibility_result(r)
        assert isinstance(s, str)
        assert "Visibility" in s

    def test_overlap_start_ge_sunset(self) -> None:
        r = check_ground_transit_visibility(2460001.1, 2460001.3, 2460001.0, 2460001.5)
        assert r.overlap_start_jd >= 2460001.0

    def test_fully_visible_overlap_equals_duration(self) -> None:
        r = check_ground_transit_visibility(2460001.1, 2460001.2, 2460001.0, 2460001.5)
        assert abs(r.overlap_hours - r.transit_duration_hours) < 1e-9

    def test_overlap_cannot_exceed_transit(self) -> None:
        r = check_ground_transit_visibility(2460001.1, 2460001.2, 2460001.0, 2460001.5)
        assert r.overlap_hours <= r.transit_duration_hours + 1e-9
