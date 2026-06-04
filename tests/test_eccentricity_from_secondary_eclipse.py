"""Tests for Skills/eccentricity_from_secondary_eclipse.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from eccentricity_from_secondary_eclipse import (
    compute_eccentricity_from_eclipse,
    format_eccentricity_result,
)


class TestComputeEccentricityFromEclipse:
    def test_ok_flag_circular(self) -> None:
        r = compute_eccentricity_from_eclipse(0.5, 3.0, 3.0)
        assert r.flag == "OK"

    def test_circular_e_cos_omega_zero(self) -> None:
        r = compute_eccentricity_from_eclipse(0.5, 3.0, 3.0)
        assert abs(r.e_cos_omega) < 1e-10

    def test_circular_e_sin_omega_zero(self) -> None:
        r = compute_eccentricity_from_eclipse(0.5, 3.0, 3.0)
        assert abs(r.e_sin_omega) < 1e-10

    def test_offset_phase_gives_nonzero_e_cos_omega(self) -> None:
        r = compute_eccentricity_from_eclipse(0.55, 3.0, 3.0)
        assert abs(r.e_cos_omega) > 0

    def test_duration_ratio_asymmetry_gives_nonzero_e_sin_omega(self) -> None:
        r = compute_eccentricity_from_eclipse(0.5, 3.0, 3.3)
        assert abs(r.e_sin_omega) > 0

    def test_e_lower_bound_non_negative(self) -> None:
        r = compute_eccentricity_from_eclipse(0.52, 3.0, 2.8)
        assert r.eccentricity_lower_bound >= 0.0

    def test_e_lower_bound_less_than_one(self) -> None:
        r = compute_eccentricity_from_eclipse(0.52, 3.0, 2.8)
        assert r.eccentricity_lower_bound <= 1.0

    def test_omega_in_valid_range(self) -> None:
        r = compute_eccentricity_from_eclipse(0.52, 3.0, 2.8)
        assert 0.0 <= r.omega_deg < 360.0

    def test_invalid_phase_zero(self) -> None:
        r = compute_eccentricity_from_eclipse(0.0, 3.0, 3.0)
        assert r.flag == "INVALID_PHASE"

    def test_invalid_phase_one(self) -> None:
        r = compute_eccentricity_from_eclipse(1.0, 3.0, 3.0)
        assert r.flag == "INVALID_PHASE"

    def test_invalid_transit_duration(self) -> None:
        r = compute_eccentricity_from_eclipse(0.5, 0.0, 3.0)
        assert r.flag == "INVALID_TRANSIT_DURATION"

    def test_invalid_eclipse_duration(self) -> None:
        r = compute_eccentricity_from_eclipse(0.5, 3.0, 0.0)
        assert r.flag == "INVALID_ECLIPSE_DURATION"

    def test_format_returns_string(self) -> None:
        r = compute_eccentricity_from_eclipse(0.5, 3.0, 3.0)
        s = format_eccentricity_result(r)
        assert isinstance(s, str)
        assert r.flag in s
