"""Tests for Skills/stellar_activity_index.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from stellar_activity_index import compute_activity_index, format_activity_index


def _flat_flux(n=100, val=1.0):
    return [val] * n


def _noisy_flux(n=100, amplitude=0.001):
    import math
    return [1.0 + amplitude * math.sin(2 * math.pi * i / 10) for i in range(n)]


def test_quiet_star():
    # Very flat flux → quiet
    flux = _flat_flux()
    r = compute_activity_index(flux)
    assert r.flag == "OK"
    assert r.activity_level == "quiet"


def test_activity_index_range():
    flux = _noisy_flux(amplitude=0.005)
    r = compute_activity_index(flux)
    assert 0.0 <= r.activity_index <= 1.0


def test_active_star():
    # High amplitude variation
    flux = [1.0 + 0.05 * (i % 2) for i in range(100)]
    r = compute_activity_index(flux)
    assert r.flag == "OK"
    assert r.activity_level in ("active", "very_active")


def test_rms_positive():
    flux = _noisy_flux()
    r = compute_activity_index(flux)
    assert r.rms_ppm > 0


def test_mad_positive():
    flux = _noisy_flux()
    r = compute_activity_index(flux)
    assert r.mad_ppm > 0


def test_peak_to_peak_positive():
    flux = _noisy_flux()
    r = compute_activity_index(flux)
    assert r.peak_to_peak_ppm > 0


def test_insufficient_data():
    r = compute_activity_index([1.0, 1.0])
    assert r.flag == "INSUFFICIENT"


def test_empty_flux():
    r = compute_activity_index([])
    assert r.flag == "INSUFFICIENT"


def test_invalid_input():
    r = compute_activity_index("not a list")
    assert r.flag == "INVALID"


def test_outlier_count():
    # Insert obvious outliers
    flux = [1.0] * 95 + [1.5] * 5
    r = compute_activity_index(flux, sigma_threshold=3.0)
    assert r.n_sigma_outliers >= 1


def test_format_returns_string():
    flux = _noisy_flux()
    r = compute_activity_index(flux)
    assert isinstance(format_activity_index(r), str)


def test_format_contains_key_words():
    flux = _noisy_flux()
    r = compute_activity_index(flux)
    text = format_activity_index(r)
    assert "Activity" in text
    assert "Flag" in text


def test_higher_noise_higher_index():
    r_low = compute_activity_index(_noisy_flux(amplitude=0.0001))
    r_high = compute_activity_index(_noisy_flux(amplitude=0.01))
    assert r_high.activity_index > r_low.activity_index
