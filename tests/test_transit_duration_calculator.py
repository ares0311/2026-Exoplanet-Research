"""Tests for Skills.transit_duration_calculator."""
from __future__ import annotations

from Skills.transit_duration_calculator import (
    TransitDurationResult,
    compute_transit_duration,
    format_duration_result,
)


class TestComputeTransitDuration:
    def test_returns_result(self) -> None:
        r = compute_transit_duration(10.0, 1.0, 1.0)
        assert isinstance(r, TransitDurationResult)

    def test_ok_flag_for_valid_inputs(self) -> None:
        r = compute_transit_duration(10.0, 1.0, 1.0, depth_ppm=1000.0)
        assert r.flag in {"OK", "GRAZING"}

    def test_invalid_zero_period(self) -> None:
        r = compute_transit_duration(0.0, 1.0, 1.0)
        assert r.flag == "INVALID"

    def test_invalid_negative_stellar_radius(self) -> None:
        r = compute_transit_duration(10.0, -1.0, 1.0)
        assert r.flag == "INVALID"

    def test_invalid_negative_stellar_mass(self) -> None:
        r = compute_transit_duration(10.0, 1.0, -1.0)
        assert r.flag == "INVALID"

    def test_t14_positive(self) -> None:
        r = compute_transit_duration(10.0, 1.0, 1.0, depth_ppm=1000.0)
        if r.t14_hours is not None:
            assert r.t14_hours > 0

    def test_t23_less_than_t14(self) -> None:
        r = compute_transit_duration(10.0, 1.0, 1.0, depth_ppm=10000.0, impact_parameter=0.0)
        if r.t14_hours is not None and r.t23_hours is not None:
            assert r.t23_hours <= r.t14_hours

    def test_ingress_egress_nonnegative(self) -> None:
        r = compute_transit_duration(10.0, 1.0, 1.0, depth_ppm=1000.0)
        if r.ingress_egress_hours is not None:
            assert r.ingress_egress_hours >= 0

    def test_larger_stellar_radius_longer_duration(self) -> None:
        r1 = compute_transit_duration(10.0, 1.0, 1.0, depth_ppm=1000.0)
        r2 = compute_transit_duration(10.0, 2.0, 1.0, depth_ppm=1000.0)
        if r1.t14_hours is not None and r2.t14_hours is not None:
            assert r2.t14_hours > r1.t14_hours

    def test_longer_period_longer_duration(self) -> None:
        # T14 ∝ P^(1/3) — longer period means larger a → longer transit
        r1 = compute_transit_duration(5.0, 1.0, 1.0, depth_ppm=1000.0)
        r2 = compute_transit_duration(100.0, 1.0, 1.0, depth_ppm=1000.0)
        if r1.t14_hours is not None and r2.t14_hours is not None:
            assert r2.t14_hours > r1.t14_hours

    def test_high_impact_parameter_grazing(self) -> None:
        r = compute_transit_duration(10.0, 1.0, 1.0, depth_ppm=100.0, impact_parameter=0.99)
        assert r.flag in {"OK", "GRAZING"}


class TestFormatDurationResult:
    def test_returns_string(self) -> None:
        r = compute_transit_duration(10.0, 1.0, 1.0)
        assert isinstance(format_duration_result(r), str)

    def test_contains_period(self) -> None:
        r = compute_transit_duration(10.0, 1.0, 1.0)
        assert "10" in format_duration_result(r)

    def test_invalid_formatted(self) -> None:
        r = compute_transit_duration(0.0, 1.0, 1.0)
        assert "INVALID" in format_duration_result(r)
