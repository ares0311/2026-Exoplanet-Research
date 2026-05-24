"""Tests for Skills/atmospheric_scale_height_calculator.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from atmospheric_scale_height_calculator import compute_scale_height, format_scale_height_result


def test_basic_ok():
    r = compute_scale_height(1000.0, 18.0, 980.0)
    assert r.flag == "OK"
    assert r.scale_height_km is not None


def test_scale_height_positive():
    r = compute_scale_height(1000.0, 2.3, 980.0)
    assert r.scale_height_km > 0


def test_scale_height_increases_with_temperature():
    r1 = compute_scale_height(500.0, 2.3, 980.0)
    r2 = compute_scale_height(1000.0, 2.3, 980.0)
    assert r2.scale_height_km > r1.scale_height_km


def test_scale_height_decreases_with_gravity():
    r1 = compute_scale_height(1000.0, 2.3, 500.0)
    r2 = compute_scale_height(1000.0, 2.3, 2000.0)
    assert r2.scale_height_km < r1.scale_height_km


def test_amplitude_positive():
    r = compute_scale_height(1000.0, 18.0, 980.0, rp_rearth=2.0, rs_rsun=1.0)
    assert r.amplitude_ppm is not None
    assert r.amplitude_ppm > 0


def test_amplitude_increases_with_n_scale_heights():
    r1 = compute_scale_height(1000.0, 2.3, 980.0, n_scale_heights=3)
    r2 = compute_scale_height(1000.0, 2.3, 980.0, n_scale_heights=6)
    assert r2.amplitude_ppm > r1.amplitude_ppm


def test_amplitude_increases_with_planet_radius():
    r1 = compute_scale_height(1000.0, 2.3, 980.0, rp_rearth=1.0)
    r2 = compute_scale_height(1000.0, 2.3, 980.0, rp_rearth=3.0)
    assert r2.amplitude_ppm > r1.amplitude_ppm


def test_invalid_zero_temperature():
    r = compute_scale_height(0.0, 2.3, 980.0)
    assert r.flag == "INVALID"
    assert r.scale_height_km is None


def test_invalid_zero_gravity():
    r = compute_scale_height(1000.0, 2.3, 0.0)
    assert r.flag == "INVALID"


def test_invalid_zero_mol_weight():
    r = compute_scale_height(1000.0, 0.0, 980.0)
    assert r.flag == "INVALID"


def test_invalid_nan():
    r = compute_scale_height(float("nan"), 2.3, 980.0)
    assert r.flag == "INVALID"


def test_invalid_n_scale_heights_zero():
    r = compute_scale_height(1000.0, 2.3, 980.0, n_scale_heights=0)
    assert r.flag == "INVALID"


def test_result_fields():
    r = compute_scale_height(800.0, 18.0, 1000.0)
    assert r.t_eq_k == 800.0
    assert r.mean_mol_weight_amu == 18.0
    assert r.gravity_cgs == 1000.0


def test_format_ok():
    r = compute_scale_height(1000.0, 2.3, 980.0)
    text = format_scale_height_result(r)
    assert "Scale Height" in text
    assert "OK" in text
