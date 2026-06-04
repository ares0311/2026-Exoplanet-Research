"""Tests for Skills/spin_orbit_alignment_estimator.py"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from spin_orbit_alignment_estimator import compute_spin_orbit_alignment, format_spin_orbit_result


class TestComputeSpinOrbitAlignment:
    def test_ok_flag(self) -> None:
        r = compute_spin_orbit_alignment(5000.0, 0.01)
        assert r.flag == "OK"

    def test_max_rm_positive(self) -> None:
        r = compute_spin_orbit_alignment(5000.0, 0.01)
        assert r.max_rm_amplitude_ms > 0

    def test_feasible_with_high_vsini(self) -> None:
        r = compute_spin_orbit_alignment(10000.0, 0.05, rv_precision_ms=1.0)
        assert r.measurement_feasible

    def test_not_feasible_small_planet_slow_rotator(self) -> None:
        r = compute_spin_orbit_alignment(100.0, 0.001, rv_precision_ms=1.0)
        assert not r.measurement_feasible

    def test_impact_reduces_rm(self) -> None:
        r_central = compute_spin_orbit_alignment(5000.0, 0.01, impact_parameter=0.0)
        r_grazing = compute_spin_orbit_alignment(5000.0, 0.01, impact_parameter=0.8)
        assert r_grazing.max_rm_amplitude_ms < r_central.max_rm_amplitude_ms

    def test_min_lambda_decreases_with_larger_rm(self) -> None:
        r_large = compute_spin_orbit_alignment(10000.0, 0.05, rv_precision_ms=1.0)
        r_small = compute_spin_orbit_alignment(1000.0, 0.01, rv_precision_ms=1.0)
        assert r_large.min_detectable_lambda_deg < r_small.min_detectable_lambda_deg

    def test_min_lambda_scales_with_precision(self) -> None:
        r1 = compute_spin_orbit_alignment(5000.0, 0.01, rv_precision_ms=1.0)
        r2 = compute_spin_orbit_alignment(5000.0, 0.01, rv_precision_ms=10.0)
        assert r2.min_detectable_lambda_deg > r1.min_detectable_lambda_deg

    def test_invalid_vsini(self) -> None:
        r = compute_spin_orbit_alignment(-100.0, 0.01)
        assert r.flag == "INVALID_VSINI"

    def test_invalid_depth_zero(self) -> None:
        r = compute_spin_orbit_alignment(5000.0, 0.0)
        assert r.flag == "INVALID_DEPTH"

    def test_invalid_impact(self) -> None:
        r = compute_spin_orbit_alignment(5000.0, 0.01, impact_parameter=-0.1)
        assert r.flag == "INVALID_IMPACT"

    def test_result_frozen(self) -> None:
        r = compute_spin_orbit_alignment(5000.0, 0.01)
        try:
            r.max_rm_amplitude_ms = 0.0  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except (AttributeError, TypeError):
            pass

    def test_format_returns_string(self) -> None:
        r = compute_spin_orbit_alignment(5000.0, 0.01)
        s = format_spin_orbit_result(r)
        assert isinstance(s, str)
        assert r.flag in s
