"""Tests for Skills/obliquity_rm_calculator.py."""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from obliquity_rm_calculator import (
    ObliquityRMResult,
    compute_obliquity_from_rm,
    format_obliquity_rm_result,
)


class TestComputeObliquityFromRM:
    def test_returns_result_type(self):
        r = compute_obliquity_from_rm(vsini_kms=5.0, depth_ppm=10000.0)
        assert isinstance(r, ObliquityRMResult)

    def test_flag_ok(self):
        r = compute_obliquity_from_rm(vsini_kms=5.0, depth_ppm=10000.0)
        assert r.flag == "OK"

    def test_rm_amplitude_positive(self):
        r = compute_obliquity_from_rm(vsini_kms=5.0, depth_ppm=10000.0)
        assert r.rm_amplitude_ms > 0.0

    def test_lambda_deg_finite(self):
        r = compute_obliquity_from_rm(vsini_kms=5.0, depth_ppm=10000.0)
        assert math.isfinite(r.lambda_deg)

    def test_obliquity_class_string(self):
        r = compute_obliquity_from_rm(vsini_kms=5.0, depth_ppm=10000.0)
        assert r.obliquity_class in ("ALIGNED", "MISALIGNED", "RETROGRADE")

    def test_larger_vsini_larger_amplitude(self):
        r1 = compute_obliquity_from_rm(vsini_kms=10.0, depth_ppm=10000.0)
        r2 = compute_obliquity_from_rm(vsini_kms=2.0, depth_ppm=10000.0)
        assert r1.rm_amplitude_ms > r2.rm_amplitude_ms

    def test_deeper_transit_larger_amplitude(self):
        r1 = compute_obliquity_from_rm(vsini_kms=5.0, depth_ppm=20000.0)
        r2 = compute_obliquity_from_rm(vsini_kms=5.0, depth_ppm=5000.0)
        assert r1.rm_amplitude_ms > r2.rm_amplitude_ms

    def test_invalid_vsini(self):
        r = compute_obliquity_from_rm(vsini_kms=0.0, depth_ppm=10000.0)
        assert r.flag != "OK"
        assert math.isnan(r.rm_amplitude_ms)

    def test_invalid_depth(self):
        r = compute_obliquity_from_rm(vsini_kms=5.0, depth_ppm=0.0)
        assert r.flag != "OK"

    def test_impact_parameter_effect(self):
        r0 = compute_obliquity_from_rm(vsini_kms=5.0, depth_ppm=10000.0, impact_parameter=0.0)
        r5 = compute_obliquity_from_rm(vsini_kms=5.0, depth_ppm=10000.0, impact_parameter=0.5)
        # Different b should give different amplitudes
        assert r0.rm_amplitude_ms != pytest.approx(r5.rm_amplitude_ms)

    def test_frozen_dataclass(self):
        r = compute_obliquity_from_rm(vsini_kms=5.0, depth_ppm=10000.0)
        try:
            r.rm_amplitude_ms = 0.0  # type: ignore[misc]
            raise AssertionError("frozen dataclass should not allow setattr")
        except (AttributeError, TypeError):
            pass

    def test_observed_rm_amplitude_sets_lambda(self):
        r = compute_obliquity_from_rm(vsini_kms=5.0, depth_ppm=10000.0,
                                       observed_rm_amplitude_ms=50.0)
        assert math.isfinite(r.lambda_deg)
        assert r.flag == "OK"


class TestFormatObliquityRMResult:
    def test_ok_returns_table(self):
        r = compute_obliquity_from_rm(vsini_kms=5.0, depth_ppm=10000.0)
        out = format_obliquity_rm_result(r)
        assert "RM amplitude" in out
        assert "|" in out

    def test_invalid_returns_flag(self):
        r = compute_obliquity_from_rm(vsini_kms=0.0, depth_ppm=10000.0)
        out = format_obliquity_rm_result(r)
        assert "flag=" in out
