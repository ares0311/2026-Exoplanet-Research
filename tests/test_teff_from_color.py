"""Tests for Skills/teff_from_color.py"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from teff_from_color import estimate_teff_from_color, format_teff_result


class TestTeffFromColor:
    def test_solar_bv(self) -> None:
        # Solar B-V ≈ 0.65 → Teff ~5778 K
        r = estimate_teff_from_color("B-V", 0.65)
        assert r.flag == "OK"
        assert 5000 < r.teff_k < 7000

    def test_solar_bprp(self) -> None:
        # Solar Bp-Rp ≈ 0.82 → Teff ~5778 K
        r = estimate_teff_from_color("Bp-Rp", 0.82)
        assert r.flag == "OK"
        assert 5000 < r.teff_k < 7000

    def test_red_star_bv(self) -> None:
        # B-V ~ 1.4 → M-dwarf temperature
        r = estimate_teff_from_color("B-V", 1.4)
        assert r.flag == "OK"
        assert r.teff_k < 5000

    def test_blue_star_bv(self) -> None:
        # B-V ~ 0.3 → hotter (A star range)
        r = estimate_teff_from_color("B-V", 0.3)
        assert r.flag == "OK"
        assert r.teff_k > 6000

    def test_out_of_range_bv(self) -> None:
        r = estimate_teff_from_color("B-V", 2.5)
        assert r.flag == "OUT_OF_RANGE"
        assert r.teff_err_k == 300.0

    def test_out_of_range_bprp(self) -> None:
        r = estimate_teff_from_color("Bp-Rp", 4.0)
        assert r.flag == "OUT_OF_RANGE"

    def test_unknown_color_index(self) -> None:
        r = estimate_teff_from_color("U-B", 0.5)
        assert r.flag == "UNKNOWN_COLOR_INDEX"
        assert math.isnan(r.teff_k)

    def test_invalid_color_value_nan(self) -> None:
        r = estimate_teff_from_color("B-V", float("nan"))
        assert r.flag == "INVALID_COLOR_VALUE"

    def test_uncertainty_in_range(self) -> None:
        r = estimate_teff_from_color("B-V", 0.65)
        assert r.teff_err_k == 150.0

    def test_redder_bprp_cooler(self) -> None:
        r1 = estimate_teff_from_color("Bp-Rp", 0.8)
        r2 = estimate_teff_from_color("Bp-Rp", 1.5)
        assert r2.teff_k < r1.teff_k

    def test_redder_bv_cooler(self) -> None:
        r1 = estimate_teff_from_color("B-V", 0.4)
        r2 = estimate_teff_from_color("B-V", 1.0)
        assert r2.teff_k < r1.teff_k

    def test_format_returns_string(self) -> None:
        r = estimate_teff_from_color("B-V", 0.65)
        s = format_teff_result(r)
        assert isinstance(s, str)
        assert "Teff" in s
