"""Tests for Skills/photometric_parallax_estimator.py."""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from photometric_parallax_estimator import (
    estimate_parallax_distance,
    format_parallax_result,
)


class TestEstimateParallaxDistance:
    def test_solar_bv(self) -> None:
        r = estimate_parallax_distance(5.0, "B-V", 0.65)
        assert r.flag == "OK"
        assert r.distance_pc > 0

    def test_distance_increases_with_apparent_mag(self) -> None:
        r1 = estimate_parallax_distance(8.0, "B-V", 0.65)
        r2 = estimate_parallax_distance(12.0, "B-V", 0.65)
        assert r2.distance_pc > r1.distance_pc

    def test_bprp_colour(self) -> None:
        r = estimate_parallax_distance(10.0, "Bp-Rp", 0.82)
        assert r.flag == "OK"
        assert r.distance_pc > 0

    def test_unknown_colour(self) -> None:
        r = estimate_parallax_distance(10.0, "V-I", 1.0)
        assert r.flag == "UNKNOWN_COLOR_INDEX"
        assert math.isnan(r.distance_pc)

    def test_invalid_apparent_mag(self) -> None:
        r = estimate_parallax_distance(float("nan"), "B-V", 0.65)
        assert r.flag == "INVALID_APPARENT_MAG"

    def test_invalid_color_value(self) -> None:
        r = estimate_parallax_distance(10.0, "B-V", float("inf"))
        assert r.flag == "INVALID_COLOR_VALUE"

    def test_extinction_reduces_distance(self) -> None:
        r0 = estimate_parallax_distance(10.0, "B-V", 0.65, 0.0)
        r1 = estimate_parallax_distance(10.0, "B-V", 0.65, 1.0)
        assert r1.distance_pc < r0.distance_pc

    def test_out_of_range(self) -> None:
        r = estimate_parallax_distance(10.0, "B-V", 2.5)
        assert r.flag == "OUT_OF_RANGE"
        assert r.distance_err_pc > 0

    def test_out_of_range_larger_fractional_error(self) -> None:
        r_in = estimate_parallax_distance(10.0, "B-V", 0.65)
        r_out = estimate_parallax_distance(10.0, "B-V", 2.5)
        frac_in = r_in.distance_err_pc / r_in.distance_pc
        frac_out = r_out.distance_err_pc / r_out.distance_pc
        assert frac_out > frac_in

    def test_abs_mag_returned(self) -> None:
        r = estimate_parallax_distance(10.0, "B-V", 0.65)
        assert math.isfinite(r.abs_mag_est)

    def test_distance_err_positive(self) -> None:
        r = estimate_parallax_distance(10.0, "B-V", 0.65)
        assert r.distance_err_pc > 0

    def test_format_output(self) -> None:
        r = estimate_parallax_distance(10.0, "B-V", 0.65)
        s = format_parallax_result(r)
        assert "|" in s
        assert "Distance" in s
