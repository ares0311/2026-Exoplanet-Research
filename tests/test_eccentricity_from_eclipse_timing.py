"""Tests for Skills/eccentricity_from_eclipse_timing.py"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from eccentricity_from_eclipse_timing import estimate_eccentricity, format_eccentricity_result


class TestEccentricityFromEclipseTiming:
    def test_circular_orbit(self) -> None:
        r = estimate_eccentricity(0.5)
        assert r.flag == "CONSISTENT_CIRCULAR"
        assert abs(r.e_cos_omega) < 1e-6

    def test_late_secondary_positive_ecos(self) -> None:
        r = estimate_eccentricity(0.6)
        assert r.flag == "ECCENTRIC_ORBIT"
        assert r.e_cos_omega > 0.0

    def test_early_secondary_negative_ecos(self) -> None:
        r = estimate_eccentricity(0.4)
        assert r.flag == "ECCENTRIC_ORBIT"
        assert r.e_cos_omega < 0.0

    def test_eccentricity_lower_bound(self) -> None:
        r = estimate_eccentricity(0.6)
        assert r.eccentricity_lower > 0.0
        assert abs(r.eccentricity_lower - abs(r.e_cos_omega)) < 1e-9

    def test_invalid_phase_zero(self) -> None:
        r = estimate_eccentricity(0.0)
        assert r.flag == "INVALID_SECONDARY_PHASE"
        assert math.isnan(r.e_cos_omega)

    def test_invalid_phase_one(self) -> None:
        r = estimate_eccentricity(1.0)
        assert r.flag == "INVALID_SECONDARY_PHASE"

    def test_invalid_phase_negative(self) -> None:
        r = estimate_eccentricity(-0.1)
        assert r.flag == "INVALID_SECONDARY_PHASE"

    def test_invalid_phase_nan(self) -> None:
        r = estimate_eccentricity(float("nan"))
        assert r.flag == "INVALID_SECONDARY_PHASE"

    def test_large_offset_eccentric(self) -> None:
        r = estimate_eccentricity(0.7, phase_err=0.001)
        assert r.flag == "ECCENTRIC_ORBIT"

    def test_phase_err_zero_large_sigma(self) -> None:
        r = estimate_eccentricity(0.55, phase_err=0.001)
        assert r.flag == "ECCENTRIC_ORBIT"

    def test_small_offset_consistent_circular(self) -> None:
        r = estimate_eccentricity(0.51, phase_err=0.02)
        assert r.flag == "CONSISTENT_CIRCULAR"

    def test_format_returns_string(self) -> None:
        r = estimate_eccentricity(0.5)
        s = format_eccentricity_result(r)
        assert isinstance(s, str)
        assert "cos" in s
