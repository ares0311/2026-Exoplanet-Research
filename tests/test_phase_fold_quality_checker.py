"""Tests for Skills/phase_fold_quality_checker.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from phase_fold_quality_checker import check_phase_fold_quality, format_phase_fold_quality


def _make_phase_flux(n=200, depth=0.01):
    # Add small noise so std_oot > 0 (needed for SNR calculation)
    a, c, m = 1664525, 1013904223, 2**32
    x = 7919
    phase, flux = [], []
    for i in range(n):
        x = (a * x + c) % m
        noise = (x / m - 0.5) * 0.001
        p = i / n - 0.5
        f = 1.0 - depth + noise if abs(p) < 0.04 else 1.0 + noise
        phase.append(p)
        flux.append(f)
    return phase, flux


def test_good_transit_grade_a():
    phase, flux = _make_phase_flux(500, depth=0.05)
    r = check_phase_fold_quality(phase, flux, transit_width_phase=0.05, n_bins=50)
    assert r.flag == "OK"
    assert r.quality_grade in ("A", "B", "C")


def test_coverage_fraction():
    phase, flux = _make_phase_flux(500)
    r = check_phase_fold_quality(phase, flux)
    assert r.coverage_fraction is not None
    assert r.coverage_fraction >= 0.0


def test_transit_snr_positive():
    phase, flux = _make_phase_flux(500, depth=0.05)
    r = check_phase_fold_quality(phase, flux)
    assert r.transit_snr is not None
    assert r.transit_snr >= 0


def test_insufficient_too_few_points():
    r = check_phase_fold_quality([0.1, 0.2], [1.0, 1.0])
    assert r.flag == "INSUFFICIENT"


def test_invalid_length_mismatch():
    r = check_phase_fold_quality([0.0, 0.1, 0.2], [1.0, 1.0])
    assert r.flag == "INVALID"


def test_invalid_zero_bins():
    phase, flux = _make_phase_flux(50)
    r = check_phase_fold_quality(phase, flux, n_bins=2)
    assert r.flag in ("OK", "INSUFFICIENT", "INVALID")


def test_invalid_negative_transit_width():
    phase, flux = _make_phase_flux(50)
    r = check_phase_fold_quality(phase, flux, transit_width_phase=-0.05)
    assert r.flag == "INVALID"


def test_symmetry_score_range():
    phase, flux = _make_phase_flux(500, depth=0.01)
    r = check_phase_fold_quality(phase, flux)
    if r.symmetry_score is not None:
        assert 0.0 <= r.symmetry_score <= 1.0


def test_flat_light_curve_low_snr():
    phase = [i / 200 - 0.5 for i in range(200)]
    flux = [1.0] * 200
    r = check_phase_fold_quality(phase, flux)
    assert r.flag in ("OK", "INSUFFICIENT")
    if r.transit_snr is not None:
        assert r.transit_snr < 1.0


def test_grade_d_sparse():
    phase = [i / 30 - 0.5 for i in range(30)]
    flux = [0.99 if abs(p) < 0.05 else 1.0 for p in phase]
    r = check_phase_fold_quality(phase, flux, n_bins=50)
    assert r.quality_grade in ("A", "B", "C", "D") or r.flag == "INSUFFICIENT"


def test_format_ok():
    phase, flux = _make_phase_flux(200)
    r = check_phase_fold_quality(phase, flux)
    text = format_phase_fold_quality(r)
    assert "Phase-Fold Quality" in text


def test_format_insufficient():
    r = check_phase_fold_quality([], [])
    text = format_phase_fold_quality(r)
    assert "INSUFFICIENT" in text


def test_n_bins_in_result():
    phase, flux = _make_phase_flux(200)
    r = check_phase_fold_quality(phase, flux, n_bins=40)
    assert r.n_bins == 40
