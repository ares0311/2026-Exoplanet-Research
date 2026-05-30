"""Tests for Skills/impact_parameter_from_duration.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from impact_parameter_from_duration import compute_impact_parameter, format_impact_parameter_result


class TestComputeImpactParameter:
    def test_basic_ok(self) -> None:
        r = compute_impact_parameter(2.5, 2.0, 1.0, 5.0, 10000.0)
        assert r.flag == "OK"

    def test_impact_parameter_non_negative(self) -> None:
        r = compute_impact_parameter(2.5, 2.0, 1.0, 5.0, 10000.0)
        assert r.impact_parameter >= 0.0

    def test_rp_over_rstar_from_depth(self) -> None:
        import math
        r = compute_impact_parameter(2.5, 2.0, 1.0, 5.0, 10000.0)
        assert abs(r.rp_over_rstar - math.sqrt(10000.0 / 1e6)) < 1e-6

    def test_invalid_t14(self) -> None:
        r = compute_impact_parameter(0.0, 1.5, 1.0, 5.0, 10000.0)
        assert r.flag == "INVALID_T14"

    def test_invalid_t23_ge_t14(self) -> None:
        r = compute_impact_parameter(2.0, 2.0, 1.0, 5.0, 10000.0)
        assert r.flag == "INVALID_T23"

    def test_invalid_t23_negative(self) -> None:
        r = compute_impact_parameter(2.0, -1.0, 1.0, 5.0, 10000.0)
        assert r.flag == "INVALID_T23"

    def test_invalid_stellar_radius(self) -> None:
        r = compute_impact_parameter(2.5, 2.0, 0.0, 5.0, 10000.0)
        assert r.flag == "INVALID_STELLAR_RADIUS"

    def test_invalid_period(self) -> None:
        r = compute_impact_parameter(2.5, 2.0, 1.0, 0.0, 10000.0)
        assert r.flag == "INVALID_PERIOD"

    def test_invalid_depth(self) -> None:
        r = compute_impact_parameter(2.5, 2.0, 1.0, 5.0, 0.0)
        assert r.flag == "INVALID_DEPTH"

    def test_inclination_near_90_for_zero_b(self) -> None:
        # When t23 close to t14, b should be near 0 and inclination near 90
        r = compute_impact_parameter(3.0, 2.9, 1.0, 5.0, 10000.0)
        assert r.inclination_deg > 80.0

    def test_a_over_rstar_positive(self) -> None:
        r = compute_impact_parameter(2.5, 2.0, 1.0, 5.0, 10000.0)
        assert r.a_over_rstar >= 0.0

    def test_format_returns_string(self) -> None:
        r = compute_impact_parameter(2.5, 2.0, 1.0, 5.0, 10000.0)
        s = format_impact_parameter_result(r)
        assert isinstance(s, str)
        assert "Impact" in s
