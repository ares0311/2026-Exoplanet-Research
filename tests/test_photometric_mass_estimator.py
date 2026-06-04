"""Tests for Skills/photometric_mass_estimator.py."""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from photometric_mass_estimator import (
    estimate_mass_from_photometry,
    format_photo_mass_result,
)


class TestEstimateMassFromPhotometry:
    def test_solar_bv(self) -> None:
        r = estimate_mass_from_photometry("B-V", 0.65)
        assert r.flag == "OK"
        assert 0.5 < r.mass_msun < 2.0

    def test_solar_bprp(self) -> None:
        r = estimate_mass_from_photometry("Bp-Rp", 0.82)
        assert r.flag == "OK"

    def test_mass_increases_for_bluer_fgk(self) -> None:
        r_blue = estimate_mass_from_photometry("B-V", 0.35)
        r_red = estimate_mass_from_photometry("B-V", 1.20)
        assert r_blue.mass_msun > r_red.mass_msun

    def test_unknown_index(self) -> None:
        r = estimate_mass_from_photometry("V-I", 1.0)
        assert r.flag == "UNKNOWN_COLOR_INDEX"
        assert math.isnan(r.mass_msun)

    def test_invalid_color_value(self) -> None:
        r = estimate_mass_from_photometry("B-V", float("nan"))
        assert r.flag == "INVALID_COLOR_VALUE"

    def test_giant_out_of_calibration(self) -> None:
        r = estimate_mass_from_photometry("B-V", 0.65, luminosity_class="III")
        assert r.flag == "OUT_OF_CALIBRATION"

    def test_out_of_range_bv(self) -> None:
        r = estimate_mass_from_photometry("B-V", 2.5)
        assert r.flag == "OUT_OF_RANGE"
        assert r.mass_err_msun > 0

    def test_out_of_range_larger_error(self) -> None:
        r_in = estimate_mass_from_photometry("B-V", 0.65)
        r_out = estimate_mass_from_photometry("B-V", 2.5)
        assert r_out.mass_err_msun / r_out.mass_msun > r_in.mass_err_msun / r_in.mass_msun

    def test_mass_positive(self) -> None:
        r = estimate_mass_from_photometry("B-V", 0.65)
        assert r.mass_msun > 0

    def test_mass_err_positive(self) -> None:
        r = estimate_mass_from_photometry("Bp-Rp", 1.0)
        assert r.mass_err_msun > 0

    def test_luminosity_class_v_default(self) -> None:
        r = estimate_mass_from_photometry("B-V", 0.65)
        assert r.luminosity_class == "V"

    def test_format_output(self) -> None:
        r = estimate_mass_from_photometry("B-V", 0.65)
        s = format_photo_mass_result(r)
        assert "|" in s
        assert "Mass" in s
