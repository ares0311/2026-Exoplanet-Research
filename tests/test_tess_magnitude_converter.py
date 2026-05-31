"""Tests for Skills/tess_magnitude_converter.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from tess_magnitude_converter import convert_tess_magnitude, format_magnitude_conversion


class TestTessMagnitudeConverter:
    def test_tess_to_gaia_g(self) -> None:
        r = convert_tess_magnitude(10.0, "TESS_T", "Gaia_G")
        assert r.flag == "OK"
        assert isinstance(r.output_magnitude, float)

    def test_same_band_no_change(self) -> None:
        r = convert_tess_magnitude(10.0, "TESS_T", "TESS_T")
        assert r.output_magnitude == 10.0
        assert r.uncertainty_mag == 0.0

    def test_v_to_tess(self) -> None:
        r = convert_tess_magnitude(10.0, "V", "TESS_T")
        assert r.flag == "OK"

    def test_tess_to_j(self) -> None:
        r = convert_tess_magnitude(10.0, "TESS_T", "J")
        assert r.flag == "OK"
        assert r.output_magnitude < 10.0  # J is redder, brighter (lower mag)

    def test_unknown_input_band(self) -> None:
        r = convert_tess_magnitude(10.0, "UNKNOWN", "TESS_T")
        assert r.flag == "UNKNOWN_INPUT_BAND"

    def test_unknown_target_band(self) -> None:
        r = convert_tess_magnitude(10.0, "TESS_T", "UNKNOWN")
        assert r.flag == "UNKNOWN_TARGET_BAND"

    def test_invalid_magnitude(self) -> None:
        import math
        r = convert_tess_magnitude(math.nan, "TESS_T", "V")
        assert r.flag == "INVALID_MAGNITUDE"

    def test_uncertainty_positive(self) -> None:
        r = convert_tess_magnitude(10.0, "TESS_T", "J")
        assert r.uncertainty_mag > 0.0

    def test_gaia_bp_redder_than_rp(self) -> None:
        r_bp = convert_tess_magnitude(10.0, "TESS_T", "Gaia_Bp")
        r_rp = convert_tess_magnitude(10.0, "TESS_T", "Gaia_Rp")
        assert r_bp.output_magnitude > r_rp.output_magnitude

    def test_output_magnitude_float(self) -> None:
        r = convert_tess_magnitude(12.5, "TESS_T", "Gaia_G")
        assert isinstance(r.output_magnitude, float)

    def test_roundtrip_approximate(self) -> None:
        r1 = convert_tess_magnitude(10.0, "TESS_T", "V")
        r2 = convert_tess_magnitude(r1.output_magnitude, "V", "TESS_T")
        assert abs(r2.output_magnitude - 10.0) < 0.01

    def test_format_returns_string(self) -> None:
        r = convert_tess_magnitude(10.0, "TESS_T", "V")
        s = format_magnitude_conversion(r)
        assert isinstance(s, str)
        assert "TESS_T" in s
