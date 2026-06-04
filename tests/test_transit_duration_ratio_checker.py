"""Tests for Skills/transit_duration_ratio_checker.py."""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from transit_duration_ratio_checker import (
    check_duration_ratio,
    format_duration_ratio_result,
)


class TestCheckDurationRatio:
    def test_normal_transit(self) -> None:
        r = check_duration_ratio(3.0, 2.0, 1000.0)
        assert r.flag == "OK"
        assert 0.0 < r.ratio < 10.0

    def test_ratio_calculation(self) -> None:
        r = check_duration_ratio(4.0, 2.0)
        assert abs(r.ratio - 2.0) < 1e-4

    def test_t23_exceeds_t14(self) -> None:
        r = check_duration_ratio(2.0, 3.0)
        assert r.flag == "T23_EXCEEDS_T14"
        assert math.isnan(r.ratio)

    def test_grazing_transit_detection(self) -> None:
        r = check_duration_ratio(10.0, 1.0)
        assert r.flag == "GRAZING_TRANSIT"

    def test_zero_t14(self) -> None:
        r = check_duration_ratio(0.0, 0.0)
        assert "INVALID" in r.flag

    def test_negative_t14(self) -> None:
        r = check_duration_ratio(-1.0, 0.5)
        assert "INVALID" in r.flag

    def test_impact_parameter_estimate_range(self) -> None:
        r = check_duration_ratio(3.0, 2.5, 10000.0)
        if r.flag == "OK":
            assert 0.0 <= r.impact_parameter_est <= 1.0

    def test_deep_transit(self) -> None:
        r = check_duration_ratio(3.0, 2.8, 50000.0)
        assert r.flag in ("OK", "HIGH_INGRESS_FRACTION")

    def test_ingress_fraction_formula(self) -> None:
        r = check_duration_ratio(4.0, 2.0)
        assert abs(r.ingress_fraction - 0.5) < 1e-4

    def test_ingress_fraction_range(self) -> None:
        r = check_duration_ratio(4.0, 2.0)
        assert 0.0 <= r.ingress_fraction <= 1.0

    def test_equal_durations_ratio_one(self) -> None:
        r = check_duration_ratio(3.0, 3.0)
        assert abs(r.ratio - 1.0) < 1e-4

    def test_format_output(self) -> None:
        r = check_duration_ratio(3.0, 2.0, 5000.0)
        s = format_duration_ratio_result(r)
        assert "T14" in s or "ratio" in s.lower()
        assert "|" in s
