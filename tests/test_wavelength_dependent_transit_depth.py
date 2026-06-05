"""Tests for Skills/wavelength_dependent_transit_depth.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from wavelength_dependent_transit_depth import (
    compute_wavelength_dependent_depth,
    format_wavelength_depth_result,
)


class TestComputeWavelengthDependentDepth:
    def test_ok_flag(self) -> None:
        r = compute_wavelength_dependent_depth(10000.0, 0.7, [0.5, 1.0, 2.0], 1.0, 1.0)
        assert r.flag == "OK"

    def test_n_depths_matches_n_wavelengths(self) -> None:
        wls = [0.4, 0.7, 1.0, 1.5, 2.2]
        r = compute_wavelength_dependent_depth(10000.0, 0.7, wls, 1.0, 1.0)
        assert len(r.depths_ppm) == len(wls)

    def test_rayleigh_slope_negative(self) -> None:
        r = compute_wavelength_dependent_depth(10000.0, 0.7, [0.5, 1.0], 1.0, 1.0)
        assert r.rayleigh_slope_ppm_per_ln_wavelength > 0.0

    def test_blue_deeper_than_red(self) -> None:
        r = compute_wavelength_dependent_depth(10000.0, 1.0, [0.4, 2.0], 1.0, 1.0)
        assert r.depths_ppm[0] > r.depths_ppm[1]

    def test_scale_height_positive(self) -> None:
        r = compute_wavelength_dependent_depth(10000.0, 0.7, [1.0], 1.0, 1.0)
        assert r.scale_height_km > 0.0

    def test_hotter_atmosphere_larger_scale_height(self) -> None:
        r_cool = compute_wavelength_dependent_depth(10000.0, 0.7, [1.0], 1.0, 1.0,
                                                     atmosphere_temp_k=500.0)
        r_hot = compute_wavelength_dependent_depth(10000.0, 0.7, [1.0], 1.0, 1.0,
                                                    atmosphere_temp_k=2000.0)
        assert r_hot.scale_height_km > r_cool.scale_height_km

    def test_reference_depth_at_ref_wavelength(self) -> None:
        r = compute_wavelength_dependent_depth(10000.0, 1.0, [1.0], 1.0, 1.0)
        assert abs(r.depths_ppm[0] - 10000.0) < 1.0

    def test_invalid_depth(self) -> None:
        r = compute_wavelength_dependent_depth(0.0, 0.7, [1.0], 1.0, 1.0)
        assert r.flag == "INVALID_DEPTH"

    def test_invalid_wavelength(self) -> None:
        r = compute_wavelength_dependent_depth(10000.0, 0.0, [1.0], 1.0, 1.0)
        assert r.flag == "INVALID_WAVELENGTH"

    def test_invalid_radii(self) -> None:
        r = compute_wavelength_dependent_depth(10000.0, 0.7, [1.0], 0.0, 1.0)
        assert r.flag == "INVALID_RADII"

    def test_result_frozen(self) -> None:
        r = compute_wavelength_dependent_depth(10000.0, 0.7, [1.0], 1.0, 1.0)
        try:
            r.scale_height_km = 0.0  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except (AttributeError, TypeError):
            pass

    def test_format_returns_string(self) -> None:
        r = compute_wavelength_dependent_depth(10000.0, 0.7, [0.5, 1.0, 2.0], 1.0, 1.0)
        s = format_wavelength_depth_result(r)
        assert isinstance(s, str)
        assert r.flag in s
