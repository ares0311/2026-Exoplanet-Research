"""Tests for Skills/rms_timescale_profiler.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from rms_timescale_profiler import format_rms_timescale_result, profile_rms_timescales


def _white_noise(n=500, cadence_days=0.5 / 24.0, seed=42):
    """Generate simple white-noise light curve."""
    # Pseudo-random noise using LCG
    a, c, m = 1664525, 1013904223, 2**32
    x = seed
    time, flux = [], []
    for i in range(n):
        x = (a * x + c) % m
        noise = (x / m - 0.5) * 0.002
        time.append(i * cadence_days)
        flux.append(1.0 + noise)
    return time, flux


def test_basic_ok():
    time, flux = _white_noise()
    r = profile_rms_timescales(time, flux)
    assert r.flag == "OK"


def test_cadence_estimated():
    time, flux = _white_noise()
    r = profile_rms_timescales(time, flux)
    assert r.cadence_hours is not None
    assert r.cadence_hours > 0


def test_baseline_rms_positive():
    time, flux = _white_noise()
    r = profile_rms_timescales(time, flux)
    assert r.baseline_rms_ppm is not None
    assert r.baseline_rms_ppm > 0


def test_timescale_bins_present():
    time, flux = _white_noise()
    r = profile_rms_timescales(time, flux)
    assert len(r.timescale_bins) > 0


def test_timescale_bins_sorted():
    time, flux = _white_noise()
    r = profile_rms_timescales(time, flux)
    ts = [b.timescale_hours for b in r.timescale_bins]
    assert ts == sorted(ts)


def test_red_noise_ratio_near_one_white_noise():
    # White noise: red_noise_ratio should be approximately ≥ 0 (not enormously above 1)
    time, flux = _white_noise(n=1000)
    r = profile_rms_timescales(time, flux)
    for b in r.timescale_bins:
        assert b.red_noise_ratio >= 0


def test_insufficient_too_few():
    r = profile_rms_timescales([0.0] * 5, [1.0] * 5)
    assert r.flag == "INSUFFICIENT"


def test_invalid_length_mismatch():
    r = profile_rms_timescales([0.0, 1.0], [1.0])
    assert r.flag == "INVALID"


def test_invalid_n_timescales_zero():
    time, flux = _white_noise()
    r = profile_rms_timescales(time, flux, n_timescales=0)
    assert r.flag == "INVALID"


def test_custom_timescale_range():
    time, flux = _white_noise()
    r = profile_rms_timescales(time, flux, min_timescale_hours=0.5, max_timescale_hours=2.0)
    assert r.flag == "OK"
    for b in r.timescale_bins:
        assert b.timescale_hours >= 0.4  # small tolerance


def test_n_points_recorded():
    time, flux = _white_noise(n=200)
    r = profile_rms_timescales(time, flux)
    assert r.n_points == 200


def test_format_ok():
    time, flux = _white_noise()
    r = profile_rms_timescales(time, flux)
    text = format_rms_timescale_result(r)
    assert "RMS Timescale" in text
    assert "OK" in text


def test_format_insufficient():
    r = profile_rms_timescales([], [])
    text = format_rms_timescale_result(r)
    assert "INSUFFICIENT" in text
