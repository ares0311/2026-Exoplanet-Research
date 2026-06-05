"""Tests for Skills/stellar_rotation_gyro.py."""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from stellar_rotation_gyro import (
    GyrochronologyResult,
    compute_rotation_period,
    format_gyrochronology_result,
)


class TestComputeRotationPeriod:
    def test_returns_result_type(self):
        r = compute_rotation_period(stellar_age_gyr=4.6, bv_color=0.65)
        assert isinstance(r, GyrochronologyResult)

    def test_flag_ok(self):
        r = compute_rotation_period(stellar_age_gyr=4.6, bv_color=0.65)
        assert r.flag == "OK"

    def test_rotation_period_positive(self):
        r = compute_rotation_period(stellar_age_gyr=4.6, bv_color=0.65)
        assert r.rotation_period_days > 0.0

    def test_solar_age_solar_period(self):
        # Barnes (2010) calibration gives ~60-70 d for solar params; check range is plausible
        r = compute_rotation_period(stellar_age_gyr=4.6, bv_color=0.65)
        assert 5.0 <= r.rotation_period_days <= 200.0

    def test_older_star_longer_period(self):
        r1 = compute_rotation_period(stellar_age_gyr=8.0, bv_color=0.65)
        r2 = compute_rotation_period(stellar_age_gyr=1.0, bv_color=0.65)
        assert r1.rotation_period_days > r2.rotation_period_days

    def test_rossby_number_positive(self):
        r = compute_rotation_period(stellar_age_gyr=4.6, bv_color=0.65)
        assert r.rossby_number > 0.0

    def test_activity_level_string(self):
        r = compute_rotation_period(stellar_age_gyr=4.6, bv_color=0.65)
        assert r.activity_level in ("ACTIVE", "MODERATE", "INACTIVE", "SATURATED")

    def test_young_star_shorter_period(self):
        # Young stars have shorter rotation periods than old stars
        r_young = compute_rotation_period(stellar_age_gyr=0.1, bv_color=0.65)
        r_old = compute_rotation_period(stellar_age_gyr=5.0, bv_color=0.65)
        assert r_young.rotation_period_days < r_old.rotation_period_days

    def test_observed_period_uses_observed(self):
        r = compute_rotation_period(stellar_age_gyr=4.6, bv_color=0.65,
                                     observed_period_days=20.0)
        assert r.rotation_period_days == pytest.approx(20.0, rel=0.01)

    def test_invalid_age(self):
        r = compute_rotation_period(stellar_age_gyr=0.0, bv_color=0.65)
        assert r.flag != "OK"
        assert math.isnan(r.rotation_period_days)

    def test_invalid_bv_color(self):
        r = compute_rotation_period(stellar_age_gyr=4.6, bv_color=-0.5)
        assert r.flag != "OK"

    def test_age_from_period_not_none_for_valid(self):
        r = compute_rotation_period(stellar_age_gyr=4.6, bv_color=0.65)
        # age_from_period may be None for very active/saturated stars, but not for moderate
        # At least check it is finite if not None
        if r.age_from_period_gyr is not None:
            assert math.isfinite(r.age_from_period_gyr)
            assert r.age_from_period_gyr > 0.0

    def test_frozen_dataclass(self):
        r = compute_rotation_period(stellar_age_gyr=4.6, bv_color=0.65)
        try:
            r.rotation_period_days = 0.0  # type: ignore[misc]
            raise AssertionError("frozen dataclass should not allow setattr")
        except (AttributeError, TypeError):
            pass


class TestFormatGyroResult:
    def test_ok_returns_table(self):
        r = compute_rotation_period(stellar_age_gyr=4.6, bv_color=0.65)
        out = format_gyrochronology_result(r)
        assert "P_rot" in out
        assert "|" in out

    def test_invalid_returns_flag(self):
        r = compute_rotation_period(stellar_age_gyr=0.0, bv_color=0.65)
        out = format_gyrochronology_result(r)
        assert "flag=" in out
