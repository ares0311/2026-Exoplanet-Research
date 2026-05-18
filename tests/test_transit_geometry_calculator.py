"""Tests for Skills.transit_geometry_calculator."""
from __future__ import annotations

import pytest
from Skills.transit_geometry_calculator import (
    TransitGeometryResult,
    compute_transit_geometry,
    format_geometry_result,
)


class TestComputeTransitGeometry:
    def test_returns_result(self) -> None:
        r = compute_transit_geometry(10.0, 1000.0, 3.0, 1.0, 1.0)
        assert isinstance(r, TransitGeometryResult)

    def test_ok_flag_valid_inputs(self) -> None:
        r = compute_transit_geometry(10.0, 1000.0, 3.0, 1.0, 1.0)
        assert r.flag in {"OK", "UNPHYSICAL"}

    def test_invalid_zero_period(self) -> None:
        r = compute_transit_geometry(0.0, 1000.0, 3.0, 1.0, 1.0)
        assert r.flag == "INVALID"

    def test_invalid_negative_depth(self) -> None:
        r = compute_transit_geometry(10.0, -100.0, 3.0, 1.0, 1.0)
        assert r.flag == "INVALID"

    def test_invalid_zero_duration(self) -> None:
        r = compute_transit_geometry(10.0, 1000.0, 0.0, 1.0, 1.0)
        assert r.flag == "INVALID"

    def test_invalid_zero_stellar_radius(self) -> None:
        r = compute_transit_geometry(10.0, 1000.0, 3.0, 0.0, 1.0)
        assert r.flag == "INVALID"

    def test_rp_over_rs_positive(self) -> None:
        r = compute_transit_geometry(10.0, 1000.0, 3.0, 1.0, 1.0)
        if r.rp_over_rs is not None:
            assert r.rp_over_rs > 0

    def test_a_over_rs_positive(self) -> None:
        r = compute_transit_geometry(10.0, 1000.0, 3.0, 1.0, 1.0)
        if r.a_over_rs is not None:
            assert r.a_over_rs > 1.0

    def test_a_au_positive(self) -> None:
        r = compute_transit_geometry(10.0, 1000.0, 3.0, 1.0, 1.0)
        if r.a_au is not None:
            assert r.a_au > 0

    def test_inclination_in_range(self) -> None:
        r = compute_transit_geometry(10.0, 1000.0, 3.0, 1.0, 1.0)
        if r.inclination_deg is not None:
            assert 0 <= r.inclination_deg <= 90

    def test_rp_over_rs_from_depth(self) -> None:
        import math
        r = compute_transit_geometry(10.0, 10000.0, 3.0, 1.0, 1.0)
        if r.rp_over_rs is not None:
            assert r.rp_over_rs == pytest.approx(math.sqrt(10000.0 / 1e6), rel=1e-3)

    def test_equatorial_transit_90_deg(self) -> None:
        r = compute_transit_geometry(10.0, 1000.0, 3.0, 1.0, 1.0, impact_parameter=0.0)
        if r.inclination_deg is not None:
            assert r.inclination_deg == pytest.approx(90.0, abs=0.1)


class TestFormatGeometryResult:
    def test_returns_string(self) -> None:
        r = compute_transit_geometry(10.0, 1000.0, 3.0, 1.0, 1.0)
        assert isinstance(format_geometry_result(r), str)

    def test_invalid_formatted(self) -> None:
        r = compute_transit_geometry(0.0, 1000.0, 3.0, 1.0, 1.0)
        assert "INVALID" in format_geometry_result(r)

    def test_contains_period(self) -> None:
        r = compute_transit_geometry(10.0, 1000.0, 3.0, 1.0, 1.0)
        assert "10" in format_geometry_result(r)
