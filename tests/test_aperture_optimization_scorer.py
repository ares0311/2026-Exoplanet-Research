"""Tests for Skills/aperture_optimization_scorer.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from aperture_optimization_scorer import format_aperture_result, score_apertures


def test_basic_ok():
    r = score_apertures(12.0, [], [], psf_fwhm_arcsec=2.0)
    assert r.flag == "OK"
    assert r.optimal_radius_arcsec is not None


def test_optimal_radius_positive():
    r = score_apertures(12.0, [], [])
    assert r.optimal_radius_arcsec > 0


def test_snr_proxy_positive():
    r = score_apertures(12.0, [], [])
    assert r.optimal_snr_proxy > 0


def test_scores_count():
    r = score_apertures(12.0, [], [], candidate_radii_arcsec=[1.0, 2.0, 3.0, 4.0])
    assert len(r.scores) == 4


def test_contamination_zero_no_neighbours():
    r = score_apertures(12.0, [], [], candidate_radii_arcsec=[2.0])
    assert r.scores[0].contamination_fraction == 0.0


def test_contamination_with_neighbour():
    r = score_apertures(12.0, [12.0], [3.0], candidate_radii_arcsec=[4.0])
    assert r.scores[0].contamination_fraction > 0.0


def test_signal_fraction_increases_with_radius():
    r = score_apertures(12.0, [], [], candidate_radii_arcsec=[1.0, 2.0, 4.0])
    fracs = [s.signal_fraction for s in r.scores]
    assert fracs[0] < fracs[1] < fracs[2]


def test_invalid_negative_fwhm():
    r = score_apertures(12.0, [], [], psf_fwhm_arcsec=-1.0)
    assert r.flag == "INVALID"


def test_invalid_length_mismatch():
    r = score_apertures(12.0, [12.0, 13.0], [5.0])
    assert r.flag == "INVALID"


def test_invalid_negative_separation():
    r = score_apertures(12.0, [12.0], [-5.0])
    assert r.flag == "INVALID"


def test_insufficient_empty_radii():
    r = score_apertures(12.0, [], [], candidate_radii_arcsec=[])
    assert r.flag == "INSUFFICIENT"


def test_format_ok():
    r = score_apertures(12.0, [], [])
    text = format_aperture_result(r)
    assert "Aperture" in text
    assert "OK" in text


def test_format_shows_optimal():
    r = score_apertures(12.0, [], [])
    text = format_aperture_result(r)
    assert "Optimal" in text


def test_default_radii_used():
    r = score_apertures(12.0, [], [])
    assert len(r.scores) > 0
