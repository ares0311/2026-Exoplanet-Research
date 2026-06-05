"""Tests for Skills/phase_curve_predictor.py."""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from phase_curve_predictor import (
    PhaseCurveResult,
    format_phase_curve_result,
    predict_phase_curve,
)


class TestPredictPhaseCurve:
    def test_returns_result_type(self):
        r = predict_phase_curve(period_days=3.0, stellar_teff_k=5778.0,
                                 stellar_radius_rsun=1.0, planet_radius_rjup=1.0)
        assert isinstance(r, PhaseCurveResult)

    def test_flag_ok(self):
        r = predict_phase_curve(period_days=3.0, stellar_teff_k=5778.0,
                                 stellar_radius_rsun=1.0, planet_radius_rjup=1.0)
        assert r.flag == "OK"

    def test_amplitudes_positive(self):
        r = predict_phase_curve(period_days=3.0, stellar_teff_k=5778.0,
                                 stellar_radius_rsun=1.0, planet_radius_rjup=1.0)
        assert r.reflected_amplitude_ppm >= 0.0
        assert r.thermal_amplitude_ppm >= 0.0
        assert r.total_amplitude_ppm >= 0.0

    def test_total_amplitude_sum(self):
        r = predict_phase_curve(period_days=3.0, stellar_teff_k=5778.0,
                                 stellar_radius_rsun=1.0, planet_radius_rjup=1.0)
        assert r.total_amplitude_ppm == pytest.approx(
            r.reflected_amplitude_ppm + r.thermal_amplitude_ppm, rel=0.01
        )

    def test_secondary_eclipse_depth_positive(self):
        r = predict_phase_curve(period_days=3.0, stellar_teff_k=5778.0,
                                 stellar_radius_rsun=1.0, planet_radius_rjup=1.0)
        assert r.secondary_eclipse_depth_ppm >= 0.0

    def test_closer_orbit_larger_amplitude(self):
        r1 = predict_phase_curve(period_days=1.5, stellar_teff_k=5778.0,
                                  stellar_radius_rsun=1.0, planet_radius_rjup=1.0)
        r2 = predict_phase_curve(period_days=10.0, stellar_teff_k=5778.0,
                                  stellar_radius_rsun=1.0, planet_radius_rjup=1.0)
        assert r1.total_amplitude_ppm > r2.total_amplitude_ppm

    def test_higher_albedo_larger_reflected(self):
        r1 = predict_phase_curve(period_days=3.0, stellar_teff_k=5778.0,
                                  stellar_radius_rsun=1.0, planet_radius_rjup=1.0,
                                  geometric_albedo=0.5)
        r2 = predict_phase_curve(period_days=3.0, stellar_teff_k=5778.0,
                                  stellar_radius_rsun=1.0, planet_radius_rjup=1.0,
                                  geometric_albedo=0.05)
        assert r1.reflected_amplitude_ppm > r2.reflected_amplitude_ppm

    def test_imperfect_redistribution_gives_thermal(self):
        # With f < 0.5 there is a day/night temp difference → nonzero thermal amplitude
        r = predict_phase_curve(period_days=3.0, stellar_teff_k=5778.0,
                                 stellar_radius_rsun=1.0, planet_radius_rjup=1.0,
                                 heat_redistribution=0.1)
        assert r.thermal_amplitude_ppm > 0.0

    def test_invalid_period(self):
        r = predict_phase_curve(period_days=0.0, stellar_teff_k=5778.0,
                                 stellar_radius_rsun=1.0, planet_radius_rjup=1.0)
        assert r.flag != "OK"
        assert math.isnan(r.reflected_amplitude_ppm)

    def test_invalid_teff(self):
        r = predict_phase_curve(period_days=3.0, stellar_teff_k=0.0,
                                 stellar_radius_rsun=1.0, planet_radius_rjup=1.0)
        assert r.flag != "OK"

    def test_frozen_dataclass(self):
        r = predict_phase_curve(period_days=3.0, stellar_teff_k=5778.0,
                                 stellar_radius_rsun=1.0, planet_radius_rjup=1.0)
        try:
            r.reflected_amplitude_ppm = 0.0  # type: ignore[misc]
            raise AssertionError("frozen dataclass should not allow setattr")
        except (AttributeError, TypeError):
            pass

    def test_phase_offset_finite(self):
        r = predict_phase_curve(period_days=3.0, stellar_teff_k=5778.0,
                                 stellar_radius_rsun=1.0, planet_radius_rjup=1.0)
        assert math.isfinite(r.phase_offset_deg)


class TestFormatPhaseCurveResult:
    def test_ok_returns_table(self):
        r = predict_phase_curve(period_days=3.0, stellar_teff_k=5778.0,
                                 stellar_radius_rsun=1.0, planet_radius_rjup=1.0)
        out = format_phase_curve_result(r)
        assert "Reflected" in out
        assert "|" in out

    def test_invalid_returns_flag(self):
        r = predict_phase_curve(period_days=0.0, stellar_teff_k=5778.0,
                                 stellar_radius_rsun=1.0, planet_radius_rjup=1.0)
        out = format_phase_curve_result(r)
        assert "flag=" in out
