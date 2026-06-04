"""Tests for Skills/phase_curve_amplitude_estimator.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from phase_curve_amplitude_estimator import (
    compute_phase_curve_amplitude,
    format_phase_curve_amplitude_result,
)


class TestComputePhaseCurveAmplitude:
    def test_ok_flag_valid_inputs(self) -> None:
        r = compute_phase_curve_amplitude(12.0, 1.0, 0.05, 1500.0)
        assert r.flag == "OK"

    def test_reflected_amplitude_positive(self) -> None:
        r = compute_phase_curve_amplitude(12.0, 1.0, 0.05, 1500.0, geometric_albedo=0.3)
        assert r.reflected_amplitude_ppm > 0

    def test_thermal_amplitude_positive_hot_planet(self) -> None:
        r = compute_phase_curve_amplitude(12.0, 1.0, 0.05, 2000.0)
        assert r.thermal_amplitude_ppm > 0

    def test_reflected_scales_with_albedo(self) -> None:
        r1 = compute_phase_curve_amplitude(12.0, 1.0, 0.05, 1500.0, geometric_albedo=0.1)
        r2 = compute_phase_curve_amplitude(12.0, 1.0, 0.05, 1500.0, geometric_albedo=0.2)
        assert abs(r2.reflected_amplitude_ppm / r1.reflected_amplitude_ppm - 2.0) < 0.01

    def test_secondary_eclipse_depth_positive(self) -> None:
        r = compute_phase_curve_amplitude(12.0, 1.0, 0.05, 1500.0)
        assert r.secondary_eclipse_depth_ppm > 0

    def test_total_amplitude_positive(self) -> None:
        r = compute_phase_curve_amplitude(12.0, 1.0, 0.05, 1500.0)
        assert r.total_amplitude_ppm > 0

    def test_closer_orbit_higher_reflected(self) -> None:
        r_close = compute_phase_curve_amplitude(12.0, 1.0, 0.02, 1500.0)
        r_far = compute_phase_curve_amplitude(12.0, 1.0, 0.1, 1500.0)
        assert r_close.reflected_amplitude_ppm > r_far.reflected_amplitude_ppm

    def test_invalid_radius_zero(self) -> None:
        r = compute_phase_curve_amplitude(0.0, 1.0, 0.05, 1500.0)
        assert r.flag == "INVALID_RADIUS"

    def test_invalid_distance_zero(self) -> None:
        r = compute_phase_curve_amplitude(12.0, 1.0, 0.0, 1500.0)
        assert r.flag == "INVALID_DISTANCE"

    def test_invalid_teq_zero(self) -> None:
        r = compute_phase_curve_amplitude(12.0, 1.0, 0.05, 0.0)
        assert r.flag == "INVALID_TEQ"

    def test_result_frozen(self) -> None:
        r = compute_phase_curve_amplitude(12.0, 1.0, 0.05, 1500.0)
        try:
            r.reflected_amplitude_ppm = 0.0  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except (AttributeError, TypeError):
            pass

    def test_format_returns_string(self) -> None:
        r = compute_phase_curve_amplitude(12.0, 1.0, 0.05, 1500.0)
        s = format_phase_curve_amplitude_result(r)
        assert isinstance(s, str)
        assert r.flag in s
