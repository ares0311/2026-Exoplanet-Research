"""Tests for Skills/photo_distance_estimator.py"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from photo_distance_estimator import (
    PhotoDistanceResult,
    estimate_photo_distance,
    format_photo_distance,
)


class TestPhotoDistanceResult:
    def test_dataclass_fields(self):
        r = PhotoDistanceResult(
            distance_pc=100.0, distance_uncertainty_pc=2.0,
            m_apparent=10.0, M_absolute=5.0, flag="OK"
        )
        assert r.distance_pc == 100.0
        assert r.flag == "OK"

    def test_frozen(self):
        r = PhotoDistanceResult(
            distance_pc=100.0, distance_uncertainty_pc=2.0,
            m_apparent=10.0, M_absolute=5.0, flag="OK"
        )
        try:
            r.distance_pc = 0
            raise AssertionError()
        except Exception:
            pass


class TestEstimatePhotoDistance:
    def test_distance_modulus_10pc(self):
        # m - M = 0 → d = 10 pc
        r = estimate_photo_distance(5.0, 5.0)
        assert abs(r.distance_pc - 10.0) < 0.01

    def test_distance_modulus_100pc(self):
        # m - M = 5 → d = 100 pc
        r = estimate_photo_distance(10.0, 5.0)
        assert abs(r.distance_pc - 100.0) < 0.1

    def test_distance_modulus_1000pc(self):
        # m - M = 10 → d = 1000 pc
        r = estimate_photo_distance(15.0, 5.0)
        assert abs(r.distance_pc - 1000.0) < 1.0

    def test_extinction_reduces_distance(self):
        r_no_ext = estimate_photo_distance(10.0, 5.0, A_v=0.0)
        r_ext = estimate_photo_distance(10.0, 5.0, A_v=1.0)
        assert r_ext.distance_pc < r_no_ext.distance_pc

    def test_uncertainty_positive(self):
        r = estimate_photo_distance(10.0, 5.0, delta_m=0.1)
        assert r.distance_uncertainty_pc > 0

    def test_larger_delta_m_larger_uncertainty(self):
        r1 = estimate_photo_distance(10.0, 5.0, delta_m=0.05)
        r2 = estimate_photo_distance(10.0, 5.0, delta_m=0.20)
        assert r2.distance_uncertainty_pc > r1.distance_uncertainty_pc

    def test_ok_flag_small_uncertainty(self):
        r = estimate_photo_distance(10.0, 5.0, delta_m=0.05)
        assert r.flag == "OK"

    def test_uncertain_flag_large_delta_m(self):
        # Very large delta_m → uncertainty > 30%
        r = estimate_photo_distance(10.0, 5.0, delta_m=2.0)
        assert r.flag == "UNCERTAIN"

    def test_m_apparent_stored(self):
        r = estimate_photo_distance(12.0, 7.0)
        assert r.m_apparent == 12.0

    def test_M_absolute_stored(self):
        r = estimate_photo_distance(12.0, 7.0)
        assert r.M_absolute == 7.0

    def test_uncertainty_formula(self):
        # sigma_d = d * ln(10)/5 * delta_m
        r = estimate_photo_distance(10.0, 5.0, delta_m=0.1)
        expected_sigma = r.distance_pc * math.log(10) / 5 * 0.1
        assert abs(r.distance_uncertainty_pc - expected_sigma) < 0.01


class TestFormatPhotoDistance:
    def test_returns_string(self):
        r = estimate_photo_distance(10.0, 5.0)
        s = format_photo_distance(r)
        assert isinstance(s, str)

    def test_contains_pc(self):
        r = estimate_photo_distance(10.0, 5.0)
        s = format_photo_distance(r)
        assert "pc" in s
