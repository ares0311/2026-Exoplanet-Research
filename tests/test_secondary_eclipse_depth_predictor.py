"""Tests for Skills/secondary_eclipse_depth_predictor.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from secondary_eclipse_depth_predictor import (
    format_secondary_eclipse_result,
    predict_secondary_eclipse_depth,
)


class TestPredictSecondaryEclipseDepth:
    def test_ok_flag(self) -> None:
        r = predict_secondary_eclipse_depth(1.0, 1.0, 5778.0, 1500.0)
        assert r.flag == "OK"

    def test_thermal_depth_positive(self) -> None:
        r = predict_secondary_eclipse_depth(1.0, 1.0, 5778.0, 2500.0, wavelength_um=4.5)
        assert r.thermal_depth_ppm > 0.0

    def test_reflected_depth_positive(self) -> None:
        r = predict_secondary_eclipse_depth(1.0, 1.0, 5778.0, 1500.0)
        assert r.reflected_depth_ppm > 0.0

    def test_total_depth_positive(self) -> None:
        r = predict_secondary_eclipse_depth(1.0, 1.0, 5778.0, 1500.0)
        assert r.total_depth_ppm > 0.0

    def test_thermal_fraction_between_zero_and_one(self) -> None:
        r = predict_secondary_eclipse_depth(1.0, 1.0, 5778.0, 2500.0, wavelength_um=4.5)
        assert 0.0 <= r.thermal_fraction <= 1.0

    def test_hotter_planet_larger_infrared_depth(self) -> None:
        r1 = predict_secondary_eclipse_depth(1.0, 1.0, 5778.0, 1000.0, wavelength_um=10.0)
        r2 = predict_secondary_eclipse_depth(1.0, 1.0, 5778.0, 2500.0, wavelength_um=10.0)
        assert r2.thermal_depth_ppm > r1.thermal_depth_ppm

    def test_larger_albedo_more_reflected(self) -> None:
        r1 = predict_secondary_eclipse_depth(1.0, 1.0, 5778.0, 1500.0, geometric_albedo=0.1)
        r2 = predict_secondary_eclipse_depth(1.0, 1.0, 5778.0, 1500.0, geometric_albedo=0.5)
        assert r2.reflected_depth_ppm > r1.reflected_depth_ppm

    def test_day_side_temp_above_teq(self) -> None:
        r = predict_secondary_eclipse_depth(1.0, 1.0, 5778.0, 1500.0, day_night_ratio=1.3)
        assert r.day_side_temp_k > 1500.0

    def test_invalid_planet_radius(self) -> None:
        r = predict_secondary_eclipse_depth(0.0, 1.0, 5778.0, 1500.0)
        assert r.flag == "INVALID_PLANET_RADIUS"

    def test_invalid_stellar_radius(self) -> None:
        r = predict_secondary_eclipse_depth(1.0, 0.0, 5778.0, 1500.0)
        assert r.flag == "INVALID_STELLAR_RADIUS"

    def test_invalid_stellar_teff(self) -> None:
        r = predict_secondary_eclipse_depth(1.0, 1.0, 0.0, 1500.0)
        assert r.flag == "INVALID_STELLAR_TEFF"

    def test_invalid_teq(self) -> None:
        r = predict_secondary_eclipse_depth(1.0, 1.0, 5778.0, 0.0)
        assert r.flag == "INVALID_TEQ"

    def test_format_returns_string(self) -> None:
        r = predict_secondary_eclipse_depth(1.0, 1.0, 5778.0, 1500.0)
        s = format_secondary_eclipse_result(r)
        assert isinstance(s, str)
        assert r.flag in s
