"""Tests for Skills/snippet_quality_scorer.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from snippet_quality_scorer import (
    format_snippet_quality,
    score_snippet_batch,
    score_snippet_quality,
)


def _make_flat_snippet(n=201, value=1.0):
    """Flat OOT flux with tiny noise, small dip in center."""
    import math
    flux = []
    for i in range(n):
        # tiny sinusoidal ripple so OOT std > 0
        noise = 5e-4 * math.sin(i * 0.5)
        flux.append(value + noise)
    mid = n // 2
    for i in range(mid - 5, mid + 5):
        flux[i] = value - 0.01
    return flux


def test_ok_flag_full_coverage():
    flux = _make_flat_snippet()
    r = score_snippet_quality(flux)
    assert r.flag == "OK"
    assert r.n_bins == 201
    assert r.n_populated == 201
    assert r.coverage_fraction == 1.0


def test_insufficient_flag_sparse():
    """Fewer than 70% populated → INSUFFICIENT."""
    flux = [None] * 201
    # populate only 50%
    for i in range(100):
        flux[i] = 1.0
    r = score_snippet_quality(flux)
    assert r.flag == "INSUFFICIENT"
    assert r.coverage_fraction < 0.70


def test_invalid_empty_list():
    r = score_snippet_quality([])
    assert r.flag == "INVALID"
    assert r.n_bins == 0


def test_depth_snr_positive():
    flux = _make_flat_snippet()
    r = score_snippet_quality(flux)
    assert r.depth_snr is not None
    assert r.depth_snr > 0


def test_in_transit_dip_negative():
    """In-transit dip should be negative (flux below OOT)."""
    flux = _make_flat_snippet()
    r = score_snippet_quality(flux)
    assert r.in_transit_dip is not None
    assert r.in_transit_dip < 0


def test_oot_noise_nonnegative():
    flux = _make_flat_snippet()
    r = score_snippet_quality(flux)
    assert r.oot_noise is not None
    assert r.oot_noise >= 0.0


def test_quality_score_range():
    flux = _make_flat_snippet()
    r = score_snippet_quality(flux)
    assert 0.0 <= r.quality_score <= 1.0


def test_none_bins_counted():
    flux = [1.0] * 201
    for i in range(50):
        flux[i] = None
    r = score_snippet_quality(flux)
    assert r.n_populated == 151
    assert abs(r.coverage_fraction - 151 / 201) < 1e-6


def test_custom_min_coverage():
    flux = [1.0] * 201
    for i in range(60):
        flux[i] = None  # 141/201 ≈ 70.1%
    r_default = score_snippet_quality(flux)
    r_strict = score_snippet_quality(flux, min_coverage=0.80)
    assert r_default.flag == "OK"
    assert r_strict.flag == "INSUFFICIENT"


def test_batch_scoring():
    snippets = [_make_flat_snippet(), [None] * 201]
    results = score_snippet_batch(snippets)
    assert len(results) == 2
    assert results[0].flag == "OK"
    assert results[1].flag == "INSUFFICIENT"


def test_flat_signal_no_dip():
    """All-constant flux — dip should be zero or near zero."""
    flux = [1.0] * 201
    r = score_snippet_quality(flux)
    # oot_noise should be zero → depth_snr None or 0
    assert r.oot_noise == 0.0 or r.oot_noise is None


def test_format_contains_key_words():
    flux = _make_flat_snippet()
    r = score_snippet_quality(flux)
    text = format_snippet_quality(r)
    assert "Snippet Quality" in text
    assert "OK" in text
    assert "coverage" in text.lower()


def test_format_invalid():
    r = score_snippet_quality([])
    text = format_snippet_quality(r)
    assert "INVALID" in text
