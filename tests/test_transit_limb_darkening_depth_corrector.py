"""Tests for Skills/transit_limb_darkening_depth_corrector.py."""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from transit_limb_darkening_depth_corrector import (
    LDDepthCorrectionResult,
    correct_depth_for_limb_darkening,
    format_ld_depth_correction_result,
)


class TestCorrectDepthForLimbDarkening:
    def test_returns_result_type(self):
        r = correct_depth_for_limb_darkening(observed_depth_ppm=10000.0, u1=0.4, u2=0.26)
        assert isinstance(r, LDDepthCorrectionResult)

    def test_flag_ok(self):
        r = correct_depth_for_limb_darkening(observed_depth_ppm=10000.0, u1=0.4, u2=0.26)
        assert r.flag == "OK"

    def test_corrected_rp_rs_positive(self):
        r = correct_depth_for_limb_darkening(observed_depth_ppm=10000.0, u1=0.4, u2=0.26)
        assert r.corrected_rp_rs > 0.0

    def test_geometric_depth_different_from_observed(self):
        # LD correction changes the depth
        r = correct_depth_for_limb_darkening(observed_depth_ppm=10000.0, u1=0.4, u2=0.26)
        assert r.geometric_depth_ppm != pytest.approx(10000.0)

    def test_no_ld_depth_unchanged(self):
        # u1=u2=0 → I_centre=I_mean=1 → LD factor=1 → geometric depth = observed
        r = correct_depth_for_limb_darkening(observed_depth_ppm=10000.0, u1=0.0, u2=0.0)
        assert r.geometric_depth_ppm == pytest.approx(10000.0, rel=0.01)
        assert r.ld_correction_factor == pytest.approx(1.0, rel=0.01)

    def test_ld_correction_factor_finite(self):
        r = correct_depth_for_limb_darkening(observed_depth_ppm=10000.0, u1=0.4, u2=0.26)
        assert math.isfinite(r.ld_correction_factor)
        assert r.ld_correction_factor > 0.0

    def test_impact_parameter_stored(self):
        r = correct_depth_for_limb_darkening(observed_depth_ppm=10000.0, u1=0.4, u2=0.26,
                                               impact_parameter=0.3)
        assert r.impact_parameter == pytest.approx(0.3)

    def test_invalid_depth(self):
        r = correct_depth_for_limb_darkening(observed_depth_ppm=0.0, u1=0.4, u2=0.26)
        assert r.flag != "OK"
        assert math.isnan(r.corrected_rp_rs)

    def test_invalid_u1(self):
        r = correct_depth_for_limb_darkening(observed_depth_ppm=10000.0, u1=2.0, u2=0.26)
        assert r.flag != "OK"

    def test_invalid_u2(self):
        r = correct_depth_for_limb_darkening(observed_depth_ppm=10000.0, u1=0.4, u2=-2.0)
        assert r.flag != "OK"

    def test_invalid_impact_parameter(self):
        r = correct_depth_for_limb_darkening(observed_depth_ppm=10000.0, u1=0.4, u2=0.26,
                                               impact_parameter=2.0)
        assert r.flag != "OK"

    def test_rp_rs_consistent_with_geometric_depth(self):
        r = correct_depth_for_limb_darkening(observed_depth_ppm=10000.0, u1=0.0, u2=0.0)
        # rp_rs = sqrt(geom_depth_ppm * 1e-6)
        expected_rp_rs = math.sqrt(r.geometric_depth_ppm * 1e-6)
        assert r.corrected_rp_rs == pytest.approx(expected_rp_rs, rel=0.001)

    def test_frozen_dataclass(self):
        r = correct_depth_for_limb_darkening(observed_depth_ppm=10000.0, u1=0.4, u2=0.26)
        try:
            r.corrected_rp_rs = 0.0  # type: ignore[misc]
            raise AssertionError("frozen dataclass should not allow setattr")
        except (AttributeError, TypeError):
            pass

    def test_observed_depth_stored(self):
        r = correct_depth_for_limb_darkening(observed_depth_ppm=12345.0, u1=0.4, u2=0.26)
        assert r.observed_depth_ppm == pytest.approx(12345.0)


class TestFormatLDDepthCorrectionResult:
    def test_ok_returns_table(self):
        r = correct_depth_for_limb_darkening(observed_depth_ppm=10000.0, u1=0.4, u2=0.26)
        out = format_ld_depth_correction_result(r)
        assert "Observed depth" in out
        assert "|" in out

    def test_invalid_returns_flag(self):
        r = correct_depth_for_limb_darkening(observed_depth_ppm=0.0, u1=0.4, u2=0.26)
        out = format_ld_depth_correction_result(r)
        assert "flag=" in out
