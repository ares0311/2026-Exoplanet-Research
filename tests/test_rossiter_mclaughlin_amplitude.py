"""Tests for Skills/rossiter_mclaughlin_amplitude.py"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from rossiter_mclaughlin_amplitude import compute_rm_amplitude, format_rm_amplitude_result


class TestComputeRmAmplitude:
    def test_central_transit(self) -> None:
        r = compute_rm_amplitude(v_sin_i_ms=5000.0, depth=0.01, impact_parameter=0.0)
        assert r.flag == "OK"
        assert abs(r.rm_amplitude_ms - 50.0) < 1e-3

    def test_scales_with_vsini(self) -> None:
        r1 = compute_rm_amplitude(10000.0, 0.01)
        r2 = compute_rm_amplitude(5000.0, 0.01)
        assert abs(r1.rm_amplitude_ms / r2.rm_amplitude_ms - 2.0) < 1e-6

    def test_scales_with_depth(self) -> None:
        r1 = compute_rm_amplitude(5000.0, 0.02)
        r2 = compute_rm_amplitude(5000.0, 0.01)
        assert abs(r1.rm_amplitude_ms / r2.rm_amplitude_ms - 2.0) < 1e-6

    def test_impact_reduces_amplitude(self) -> None:
        r_central = compute_rm_amplitude(5000.0, 0.01, impact_parameter=0.0)
        r_grazing = compute_rm_amplitude(5000.0, 0.01, impact_parameter=0.8)
        assert r_grazing.rm_amplitude_ms < r_central.rm_amplitude_ms

    def test_impact_zero_formula(self) -> None:
        r = compute_rm_amplitude(1000.0, 0.01, impact_parameter=0.0)
        expected = 1000.0 * 0.01 * math.sqrt(1 - 0.0)
        assert abs(r.rm_amplitude_ms - expected) < 1e-3

    def test_invalid_vsini_negative(self) -> None:
        r = compute_rm_amplitude(-100.0, 0.01)
        assert r.flag == "INVALID_VSINI"

    def test_invalid_depth_zero(self) -> None:
        r = compute_rm_amplitude(5000.0, 0.0)
        assert r.flag == "INVALID_DEPTH"

    def test_invalid_depth_one(self) -> None:
        r = compute_rm_amplitude(5000.0, 1.0)
        assert r.flag == "INVALID_DEPTH"

    def test_invalid_impact_negative(self) -> None:
        r = compute_rm_amplitude(5000.0, 0.01, impact_parameter=-0.1)
        assert r.flag == "INVALID_IMPACT"

    def test_invalid_impact_one(self) -> None:
        r = compute_rm_amplitude(5000.0, 0.01, impact_parameter=1.0)
        assert r.flag == "INVALID_IMPACT"

    def test_result_frozen(self) -> None:
        r = compute_rm_amplitude(5000.0, 0.01)
        try:
            r.rm_amplitude_ms = 999.0  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except (AttributeError, TypeError):
            pass

    def test_format_returns_string_with_flag(self) -> None:
        r = compute_rm_amplitude(5000.0, 0.01)
        s = format_rm_amplitude_result(r)
        assert isinstance(s, str)
        assert r.flag in s
