"""Tests for Skills/stellar_properties_reporter.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

import pytest
from stellar_properties_reporter import (
    StellarReport,
    build_stellar_report,
    format_stellar_report,
)


def _solar():
    return build_stellar_report(1, teff_k=5778.0, radius_rsun=1.0, mass_msun=1.0)


def test_solar_luminosity_near_one():
    report = _solar()
    assert report.luminosity_lsun == pytest.approx(1.0, abs=0.01)


def test_solar_spectral_type_g():
    report = _solar()
    assert report.spectral_type == "G"


def test_hz_boundaries_computed():
    report = _solar()
    assert report.hz_inner_au is not None
    assert report.hz_outer_au is not None
    assert report.hz_inner_au < report.hz_outer_au


def test_hz_inner_near_one_au():
    report = _solar()
    assert 0.8 < report.hz_inner_au < 1.5


def test_flag_ok_with_all_params():
    report = _solar()
    assert report.flag == "OK"


def test_flag_incomplete_missing_params():
    report = build_stellar_report(1, teff_k=5778.0)
    assert report.flag == "INCOMPLETE"


def test_invalid_teff():
    report = build_stellar_report(1, teff_k=-100.0)
    assert report.flag == "INVALID"


def test_m_dwarf_spectral_type():
    report = build_stellar_report(1, teff_k=3200.0)
    assert report.spectral_type == "M"


def test_k_dwarf_spectral_type():
    report = build_stellar_report(1, teff_k=4500.0)
    assert report.spectral_type == "K"


def test_f_star_spectral_type():
    report = build_stellar_report(1, teff_k=6500.0)
    assert report.spectral_type == "F"


def test_luminosity_none_without_radius():
    report = build_stellar_report(1, teff_k=5778.0)
    assert report.luminosity_lsun is None


def test_returns_stellar_report():
    report = _solar()
    assert isinstance(report, StellarReport)


def test_metallicity_stored():
    report = build_stellar_report(1, teff_k=5778.0, radius_rsun=1.0,
                                  mass_msun=1.0, metallicity_dex=-0.2)
    assert report.metallicity_dex == pytest.approx(-0.2)


def test_format_contains_tic():
    report = build_stellar_report(55555, teff_k=5778.0)
    md = format_stellar_report(report)
    assert "55555" in md


def test_format_contains_spectral_type():
    report = _solar()
    md = format_stellar_report(report)
    assert "G" in md
