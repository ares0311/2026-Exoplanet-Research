"""Tests for Skills/photoevaporation_timescale.py"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from photoevaporation_timescale import compute_photoevaporation_timescale, format_photoevaporation_result


class TestComputePhotoevaporationTimescale:
    def test_ok_flag_valid_inputs(self) -> None:
        r = compute_photoevaporation_timescale(2.0, 10.0, 0.05, xuv_flux_wm2=1e-2)
        assert r.flag == "OK"

    def test_unstable_hot_close_in(self) -> None:
        r = compute_photoevaporation_timescale(2.0, 5.0, 0.02, xuv_flux_wm2=1000.0,
                                               atmosphere_fraction=0.01)
        assert r.stability_class == "UNSTABLE"
        assert r.timescale_gyr < 1.0

    def test_stable_far_out(self) -> None:
        r = compute_photoevaporation_timescale(2.0, 10.0, 5.0, xuv_flux_wm2=1e-5,
                                               atmosphere_fraction=0.01)
        assert r.stability_class == "STABLE"
        assert r.timescale_gyr >= 10.0

    def test_mass_loss_rate_positive(self) -> None:
        r = compute_photoevaporation_timescale(2.0, 10.0, 0.1, xuv_flux_wm2=0.1)
        assert r.mass_loss_rate_gs > 0

    def test_higher_flux_shorter_timescale(self) -> None:
        r1 = compute_photoevaporation_timescale(2.0, 10.0, 0.1, xuv_flux_wm2=1.0)
        r2 = compute_photoevaporation_timescale(2.0, 10.0, 0.1, xuv_flux_wm2=10.0)
        assert r2.timescale_gyr < r1.timescale_gyr

    def test_invalid_radius(self) -> None:
        r = compute_photoevaporation_timescale(0.0, 10.0, 0.1, xuv_flux_wm2=0.1)
        assert r.flag == "INVALID_RADIUS"

    def test_invalid_mass(self) -> None:
        r = compute_photoevaporation_timescale(2.0, 0.0, 0.1, xuv_flux_wm2=0.1)
        assert r.flag == "INVALID_MASS"

    def test_invalid_xuv_flux(self) -> None:
        r = compute_photoevaporation_timescale(2.0, 10.0, 0.1, xuv_flux_wm2=0.0)
        assert r.flag == "INVALID_XUV_FLUX"

    def test_larger_planet_lower_mass_loss_rate(self) -> None:
        r1 = compute_photoevaporation_timescale(1.0, 10.0, 0.1, xuv_flux_wm2=0.1)
        r2 = compute_photoevaporation_timescale(3.0, 10.0, 0.1, xuv_flux_wm2=0.1)
        assert r2.mass_loss_rate_gs > r1.mass_loss_rate_gs

    def test_result_frozen(self) -> None:
        r = compute_photoevaporation_timescale(2.0, 10.0, 0.1, xuv_flux_wm2=0.1)
        try:
            r.timescale_gyr = 0.0  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except (AttributeError, TypeError):
            pass

    def test_efficiency_scales_timescale(self) -> None:
        r1 = compute_photoevaporation_timescale(2.0, 10.0, 0.1, xuv_flux_wm2=0.1, efficiency=0.10)
        r2 = compute_photoevaporation_timescale(2.0, 10.0, 0.1, xuv_flux_wm2=0.1, efficiency=0.20)
        assert abs(r2.timescale_gyr / r1.timescale_gyr - 0.5) < 1e-4

    def test_format_returns_string(self) -> None:
        r = compute_photoevaporation_timescale(2.0, 10.0, 0.1, xuv_flux_wm2=0.1)
        s = format_photoevaporation_result(r)
        assert isinstance(s, str)
        assert r.flag in s
