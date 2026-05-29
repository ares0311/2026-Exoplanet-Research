"""Tests for Skills/photometry_precision_budget.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from photometry_precision_budget import (  # noqa: E402
    PrecisionBudget,
    compute_precision_budget,
    format_precision_budget,
)


def test_returns_dataclass():
    r = compute_precision_budget(10000.0, 60.0)
    assert isinstance(r, PrecisionBudget)


def test_photon_noise_positive():
    r = compute_precision_budget(10000.0, 60.0)
    assert r.photon_noise_ppm > 0


def test_sky_noise_positive():
    r = compute_precision_budget(10000.0, 60.0)
    assert r.sky_noise_ppm > 0


def test_scint_noise_positive():
    r = compute_precision_budget(10000.0, 60.0)
    assert r.scint_noise_ppm > 0


def test_read_noise_positive():
    r = compute_precision_budget(10000.0, 60.0)
    assert r.read_noise_ppm > 0


def test_total_noise_exceeds_components():
    r = compute_precision_budget(10000.0, 60.0)
    max_component = max(
        r.photon_noise_ppm, r.sky_noise_ppm, r.scint_noise_ppm, r.read_noise_ppm
    )
    assert r.total_noise_ppm >= max_component


def test_flag_ok_for_photon_limited():
    # Faint star with suppressed sky/read noise: photon noise dominates → OK
    r = compute_precision_budget(
        1.0, 60.0, sky_adu_pix=0.001, read_noise_e=0.001,
        aperture_cm=200.0, airmass=1.0, elevation_m=4000.0,
    )
    assert r.flag == "OK"


def test_flag_sky_limited():
    # Very faint star with bright sky
    r = compute_precision_budget(10.0, 60.0, sky_adu_pix=1000.0)
    assert r.flag == "SKY_LIMITED"


def test_higher_airmass_more_scintillation():
    r1 = compute_precision_budget(10000.0, 60.0, airmass=1.2)
    r2 = compute_precision_budget(10000.0, 60.0, airmass=2.5)
    assert r2.scint_noise_ppm > r1.scint_noise_ppm


def test_longer_exposure_less_scintillation():
    r1 = compute_precision_budget(10000.0, 30.0)
    r2 = compute_precision_budget(10000.0, 120.0)
    assert r2.scint_noise_ppm < r1.scint_noise_ppm


def test_format_returns_string():
    r = compute_precision_budget(10000.0, 60.0)
    s = format_precision_budget(r)
    assert isinstance(s, str)


def test_format_contains_total():
    r = compute_precision_budget(10000.0, 60.0)
    s = format_precision_budget(r)
    assert "Total" in s


def test_format_contains_flag():
    r = compute_precision_budget(10000.0, 60.0)
    s = format_precision_budget(r)
    assert "Flag" in s
