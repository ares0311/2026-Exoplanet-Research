"""Tests for Skills/limb_darkening_depth_corrector.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from limb_darkening_depth_corrector import correct_depth_for_ld, format_ld_depth_result


class TestCorrectDepthForLd:
    def test_zero_ld_no_correction(self) -> None:
        r = correct_depth_for_ld(1000.0, 0.0, 0.0)
        assert r.flag == "OK"
        assert abs(r.corrected_depth_ppm - 1000.0) < 1e-6

    def test_corrected_larger_than_observed(self) -> None:
        r = correct_depth_for_ld(1000.0, 0.3, 0.2)
        assert r.corrected_depth_ppm >= r.observed_depth_ppm

    def test_invalid_depth(self) -> None:
        r = correct_depth_for_ld(0.0, 0.3, 0.2)
        assert r.flag == "INVALID_DEPTH"

    def test_invalid_ld_coeffs_sum(self) -> None:
        r = correct_depth_for_ld(1000.0, 0.8, 0.8)
        assert r.flag == "INVALID_LD_COEFFS"

    def test_invalid_impact_parameter(self) -> None:
        r = correct_depth_for_ld(1000.0, 0.3, 0.2, impact_parameter=1.0)
        assert r.flag == "INVALID_IMPACT_PARAMETER"

    def test_flag_ok(self) -> None:
        r = correct_depth_for_ld(1000.0, 0.3, 0.1)
        assert r.flag == "OK"

    def test_ld_factor_in_range(self) -> None:
        r = correct_depth_for_ld(1000.0, 0.3, 0.1)
        assert 0.0 < r.ld_correction_factor <= 1.0

    def test_u1_u2_stored(self) -> None:
        r = correct_depth_for_ld(1000.0, 0.35, 0.15)
        assert r.u1 == 0.35
        assert r.u2 == 0.15

    def test_observed_depth_stored(self) -> None:
        r = correct_depth_for_ld(1234.0, 0.3, 0.1)
        assert r.observed_depth_ppm == 1234.0

    def test_nonzero_impact_parameter(self) -> None:
        r0 = correct_depth_for_ld(1000.0, 0.3, 0.1, impact_parameter=0.0)
        r1 = correct_depth_for_ld(1000.0, 0.3, 0.1, impact_parameter=0.5)
        assert r0.corrected_depth_ppm != r1.corrected_depth_ppm

    def test_format_returns_string(self) -> None:
        r = correct_depth_for_ld(1000.0, 0.3, 0.1)
        s = format_ld_depth_result(r)
        assert isinstance(s, str)
        assert "Depth" in s

    def test_result_frozen(self) -> None:
        r = correct_depth_for_ld(1000.0, 0.3, 0.1)
        try:
            r.flag = "X"  # type: ignore[misc]
            raise AssertionError("should be frozen")
        except Exception:
            pass
