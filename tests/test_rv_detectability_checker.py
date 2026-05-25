"""Tests for Skills/rv_detectability_checker.py"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from rv_detectability_checker import check_rv_detectability, format_rv_detectability


def test_jupiter_detectable():
    # Jupiter-mass (~318 M_earth), 1yr period, Sun-like star, 1 m/s precision
    r = check_rv_detectability(318.0, 1.0, 365.0, rv_precision_ms=1.0, n_obs=20)
    assert r.flag == "OK"
    assert r.is_detectable


def test_earth_not_detectable_1ms():
    # Earth (1 M_earth), 1yr, 1 m/s — not detectable
    r = check_rv_detectability(1.0, 1.0, 365.0, rv_precision_ms=1.0, n_obs=10)
    assert r.flag == "OK"
    assert not r.is_detectable


def test_k_amplitude_positive():
    r = check_rv_detectability(10.0, 1.0, 10.0)
    assert r.k_ms is not None
    assert r.k_ms > 0


def test_k_scales_with_mass():
    r1 = check_rv_detectability(1.0, 1.0, 10.0)
    r2 = check_rv_detectability(10.0, 1.0, 10.0)
    assert r2.k_ms > r1.k_ms


def test_snr_scales_with_n_obs():
    r1 = check_rv_detectability(10.0, 1.0, 10.0, n_obs=4)
    r2 = check_rv_detectability(10.0, 1.0, 10.0, n_obs=16)
    assert r2.snr_rv > r1.snr_rv


def test_n_obs_required():
    r = check_rv_detectability(10.0, 1.0, 10.0)
    assert r.n_obs_required is not None
    assert r.n_obs_required >= 1


def test_invalid_zero_mass():
    r = check_rv_detectability(0.0, 1.0, 10.0)
    assert r.flag == "INVALID"
    assert r.k_ms is None


def test_invalid_zero_period():
    r = check_rv_detectability(1.0, 1.0, 0.0)
    assert r.flag == "INVALID"


def test_invalid_zero_stellar_mass():
    r = check_rv_detectability(1.0, 0.0, 10.0)
    assert r.flag == "INVALID"


def test_invalid_negative_precision():
    r = check_rv_detectability(1.0, 1.0, 10.0, rv_precision_ms=-1.0)
    assert r.flag == "INVALID"


def test_invalid_zero_n_obs():
    r = check_rv_detectability(1.0, 1.0, 10.0, n_obs=0)
    assert r.flag == "INVALID"


def test_format_ok():
    r = check_rv_detectability(318.0, 1.0, 365.0)
    text = format_rv_detectability(r)
    assert "RV Detectability" in text
    assert "OK" in text


def test_format_detectable():
    r = check_rv_detectability(318.0, 1.0, 365.0, n_obs=50)
    text = format_rv_detectability(r)
    assert "DETECTABLE" in text


def test_snr_formula():
    # SNR = K * sqrt(N) / sigma
    r = check_rv_detectability(10.0, 1.0, 10.0, rv_precision_ms=1.0, n_obs=9)
    assert r.snr_rv is not None
    expected = r.k_ms * math.sqrt(9) / 1.0
    assert abs(r.snr_rv - expected) < 0.01
