"""Tests for Skills/stellar_age_gyrochronology.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from stellar_age_gyrochronology import estimate_stellar_age, format_gyro_result


def test_solar_age():
    # Sun: P≈25d, B-V≈0.65 → Barnes (2007) gives ~7 Gyr
    r = estimate_stellar_age(25.0, 0.65)
    assert r.flag == "OK"
    assert r.age_gyr is not None
    assert 3.0 < r.age_gyr < 12.0


def test_interface_sequence():
    r = estimate_stellar_age(25.0, 0.65)
    assert r.sequence == "I"


def test_convective_sequence():
    # Short rotation period → convective sequence
    r = estimate_stellar_age(0.1, 0.65)
    assert r.flag == "OK"
    assert r.sequence == "C"


def test_age_increases_with_period():
    r1 = estimate_stellar_age(10.0, 0.65)
    r2 = estimate_stellar_age(30.0, 0.65)
    assert r2.age_gyr > r1.age_gyr


def test_age_myr_gyr_consistent():
    r = estimate_stellar_age(25.0, 0.65)
    assert r.age_myr is not None and r.age_gyr is not None
    assert abs(r.age_myr / 1000.0 - r.age_gyr) < 0.001


def test_invalid_zero_period():
    r = estimate_stellar_age(0.0, 0.65)
    assert r.flag == "INVALID"
    assert r.age_gyr is None


def test_invalid_negative_period():
    r = estimate_stellar_age(-1.0, 0.65)
    assert r.flag == "INVALID"


def test_invalid_bv_at_boundary():
    r = estimate_stellar_age(10.0, 0.495)
    assert r.flag == "INVALID"


def test_invalid_bv_below_min():
    r = estimate_stellar_age(10.0, 0.3)
    assert r.flag == "INVALID"


def test_invalid_nan_period():
    r = estimate_stellar_age(float("nan"), 0.65)
    assert r.flag == "INVALID"


def test_invalid_nan_bv():
    r = estimate_stellar_age(10.0, float("nan"))
    assert r.flag == "INVALID"


def test_result_fields():
    r = estimate_stellar_age(20.0, 0.7)
    assert r.p_rot_days == 20.0
    assert r.b_minus_v == 0.7


def test_format_ok():
    r = estimate_stellar_age(25.0, 0.65)
    text = format_gyro_result(r)
    assert "Gyrochronology" in text
    assert "OK" in text


def test_format_invalid():
    r = estimate_stellar_age(0.0, 0.65)
    text = format_gyro_result(r)
    assert "INVALID" in text
