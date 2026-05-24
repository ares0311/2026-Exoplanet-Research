"""Tests for Skills/contact_time_calculator.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from contact_time_calculator import compute_contact_times, format_contact_times


def test_basic_contact_times():
    r = compute_contact_times(2459000.0, 2.0, t23_hours=1.0)
    assert r.flag == "OK"
    assert r.t1_bjd is not None
    assert r.t4_bjd is not None
    assert r.t1_bjd < r.t0_bjd < r.t4_bjd


def test_symmetric_about_t0():
    r = compute_contact_times(2459000.0, 4.0, t23_hours=2.0)
    assert abs((r.t0_bjd - r.t1_bjd) - (r.t4_bjd - r.t0_bjd)) < 1e-9


def test_total_duration():
    r = compute_contact_times(2459000.0, 3.0, t23_hours=1.5)
    assert abs(r.total_duration_hours - 3.0) < 1e-9


def test_with_t23():
    r = compute_contact_times(2459000.0, 4.0, t23_hours=2.0)
    assert r.flag == "OK"
    assert r.t2_bjd is not None
    assert r.t3_bjd is not None
    assert r.t1_bjd < r.t2_bjd < r.t3_bjd < r.t4_bjd


def test_flat_bottom_duration():
    r = compute_contact_times(2459000.0, 4.0, t23_hours=2.0)
    assert r.flat_bottom_hours is not None
    assert abs(r.flat_bottom_hours - 2.0) < 1e-9


def test_ingress_duration():
    r = compute_contact_times(2459000.0, 4.0, t23_hours=2.0)
    assert r.ingress_duration_hours is not None
    expected_ingress = (4.0 - 2.0) / 2.0
    assert abs(r.ingress_duration_hours - expected_ingress) < 1e-9


def test_grazing_flag():
    r = compute_contact_times(2459000.0, 2.0)
    assert r.flag == "GRAZING"
    assert r.t2_bjd is None
    assert r.t3_bjd is None


def test_invalid_zero_duration():
    r = compute_contact_times(2459000.0, 0.0)
    assert r.flag == "INVALID"


def test_invalid_negative_duration():
    r = compute_contact_times(2459000.0, -1.0)
    assert r.flag == "INVALID"


def test_invalid_t23_larger_than_t14():
    r = compute_contact_times(2459000.0, 2.0, t23_hours=3.0)
    assert r.flag == "INVALID"


def test_t0_preserved():
    r = compute_contact_times(2459123.456, 2.0)
    assert r.t0_bjd == 2459123.456


def test_format_ok():
    r = compute_contact_times(2459000.0, 2.0, t23_hours=1.0)
    text = format_contact_times(r)
    assert "Contact Time" in text
    assert "OK" in text


def test_format_grazing():
    r = compute_contact_times(2459000.0, 2.0)
    text = format_contact_times(r)
    assert "GRAZING" in text


def test_format_invalid():
    r = compute_contact_times(2459000.0, 0.0)
    text = format_contact_times(r)
    assert "INVALID" in text
