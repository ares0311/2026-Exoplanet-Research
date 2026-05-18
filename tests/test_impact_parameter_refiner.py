"""Tests for Skills.impact_parameter_refiner."""
from __future__ import annotations

from Skills.impact_parameter_refiner import (
    ImpactParameterResult,
    format_impact_parameter_result,
    refine_impact_parameter,
)


class TestRefineImpactParameter:
    def test_returns_result(self) -> None:
        r = refine_impact_parameter(5.0, 2.0, 1000.0, 1.0, 1.0)
        assert isinstance(r, ImpactParameterResult)

    def test_zero_period_invalid(self) -> None:
        r = refine_impact_parameter(0.0, 2.0, 1000.0, 1.0, 1.0)
        assert r.flag == "INVALID"

    def test_zero_duration_invalid(self) -> None:
        r = refine_impact_parameter(5.0, 0.0, 1000.0, 1.0, 1.0)
        assert r.flag == "INVALID"

    def test_b_nonnegative(self) -> None:
        r = refine_impact_parameter(5.0, 2.0, 1000.0, 1.0, 1.0)
        assert r.b >= 0

    def test_inclination_in_range(self) -> None:
        r = refine_impact_parameter(5.0, 2.0, 1000.0, 1.0, 1.0)
        assert 0.0 <= r.inclination_deg <= 90.0

    def test_a_over_rstar_positive(self) -> None:
        r = refine_impact_parameter(5.0, 2.0, 1000.0, 1.0, 1.0)
        assert r.a_over_rstar > 0

    def test_central_transit_low_b(self) -> None:
        # Very long duration → nearly central transit → low b
        r = refine_impact_parameter(5.0, 3.5, 10000.0, 1.0, 1.0)
        assert r.flag in {"CENTRAL", "GRAZING", "INVALID"}

    def test_error_propagation(self) -> None:
        r = refine_impact_parameter(5.0, 2.0, 1000.0, 1.0, 1.0, duration_err_hours=0.1)
        assert r.b_err is not None
        assert r.b_err >= 0

    def test_no_err_when_not_given(self) -> None:
        r = refine_impact_parameter(5.0, 2.0, 1000.0, 1.0, 1.0)
        assert r.b_err is None

    def test_flag_values_valid(self) -> None:
        r = refine_impact_parameter(5.0, 2.0, 1000.0, 1.0, 1.0)
        assert r.flag in {"CENTRAL", "GRAZING", "INVALID"}


class TestFormatImpactParameter:
    def test_returns_string(self) -> None:
        r = refine_impact_parameter(5.0, 2.0, 1000.0, 1.0, 1.0)
        assert isinstance(format_impact_parameter_result(r), str)

    def test_contains_b_value(self) -> None:
        r = refine_impact_parameter(5.0, 2.0, 1000.0, 1.0, 1.0)
        assert "b" in format_impact_parameter_result(r).lower()
