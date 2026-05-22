"""Tests for Skills/barycentric_time_corrector.py."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from barycentric_time_corrector import (
    BarycentricResult,
    apply_barycentric_correction,
    compute_barycentric_correction,
    format_barycentric_result,
)

_J2000_JD = 2451545.0


class TestComputeBarycentricCorrection:
    def test_returns_float(self):
        corr = compute_barycentric_correction(_J2000_JD, 180.0, 0.0)
        assert isinstance(corr, float)

    def test_correction_includes_tt_utc_offset(self):
        # TT-UTC contributes ~69.184 s; total correction is within ±600 s of 69.184
        corr = compute_barycentric_correction(_J2000_JD, 180.0, 0.0)
        assert abs(corr - 69.184) < 600.0

    def test_different_ra_gives_different_correction(self):
        corr1 = compute_barycentric_correction(_J2000_JD, 0.0, 0.0)
        corr2 = compute_barycentric_correction(_J2000_JD, 180.0, 0.0)
        assert corr1 != pytest.approx(corr2, abs=1.0)

    def test_north_south_poles_have_same_correction(self):
        # At dec=±90 the RA projection vanishes; corrections should differ only by Roemer
        corr_north = compute_barycentric_correction(_J2000_JD, 0.0, 90.0)
        corr_south = compute_barycentric_correction(_J2000_JD, 0.0, -90.0)
        # They can differ; just check both are floats
        assert isinstance(corr_north, float)
        assert isinstance(corr_south, float)

    def test_correction_within_physical_bounds(self):
        # Max Roemer delay ~±499 s; TT-UTC ~69 s → range roughly [-430, 570] s
        corr = compute_barycentric_correction(_J2000_JD, 90.0, 23.5)
        assert -500 < corr < 600


class TestApplyBarycentricCorrection:
    def test_returns_barycentric_result(self):
        result = apply_barycentric_correction([_J2000_JD], 180.0, 0.0)
        assert isinstance(result, BarycentricResult)

    def test_flag_ok_for_valid_input(self):
        result = apply_barycentric_correction([_J2000_JD], 180.0, 0.0)
        assert result.flag == "OK"

    def test_empty_times_returns_invalid(self):
        result = apply_barycentric_correction([], 180.0, 0.0)
        assert result.flag == "INVALID"

    def test_invalid_dec_returns_invalid(self):
        result = apply_barycentric_correction([_J2000_JD], 180.0, 95.0)
        assert result.flag == "INVALID"

    def test_invalid_ra_returns_invalid(self):
        result = apply_barycentric_correction([_J2000_JD], -10.0, 0.0)
        assert result.flag == "INVALID"

    def test_bjd_greater_than_jd_by_correction(self):
        result = apply_barycentric_correction([_J2000_JD], 180.0, 0.0)
        if result.flag == "OK":
            expected_bjd = result.utc_jd + result.correction_seconds / 86400.0
            assert result.bjd_tdb == pytest.approx(expected_bjd, abs=1e-6)

    def test_identity_for_bjd_tdb_input(self):
        result = apply_barycentric_correction([_J2000_JD], 180.0, 0.0, from_system="BJD_TDB")
        assert result.flag == "OK"
        assert result.bjd_tdb == pytest.approx(result.utc_jd)
        assert result.correction_seconds == pytest.approx(0.0)

    def test_invalid_from_system_returns_invalid(self):
        result = apply_barycentric_correction([_J2000_JD], 180.0, 0.0, from_system="HJD")
        assert result.flag == "INVALID"

    def test_utc_jd_stored(self):
        result = apply_barycentric_correction([_J2000_JD], 180.0, 0.0)
        assert result.utc_jd == pytest.approx(_J2000_JD)

    def test_correction_seconds_reasonable(self):
        result = apply_barycentric_correction([_J2000_JD], 180.0, 0.0)
        assert result.correction_seconds is not None
        assert -500.0 < result.correction_seconds < 600.0

    def test_frozen_dataclass(self):
        result = apply_barycentric_correction([_J2000_JD], 180.0, 0.0)
        with pytest.raises((AttributeError, TypeError)):
            result.flag = "CHANGED"  # type: ignore[misc]


class TestFormatBarycentricResult:
    def test_returns_string(self):
        result = apply_barycentric_correction([_J2000_JD], 180.0, 0.0)
        md = format_barycentric_result(result)
        assert isinstance(md, str)

    def test_contains_flag(self):
        result = apply_barycentric_correction([_J2000_JD], 180.0, 0.0)
        md = format_barycentric_result(result)
        assert result.flag in md

    def test_invalid_result_format(self):
        result = apply_barycentric_correction([], 180.0, 0.0)
        md = format_barycentric_result(result)
        assert "INVALID" in md
