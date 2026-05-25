"""Tests for Skills/target_crowding_estimator.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from target_crowding_estimator import (
    estimate_crowding,
    format_crowding,
)


def test_no_neighbors_ok():
    r = estimate_crowding(12.0, [], [])
    assert r.flag == "OK"
    assert r.flux_ratio == 0.0
    assert r.crowding_metric == 1.0
    assert r.contamination_fraction == 0.0
    assert r.n_neighbors == 0


def test_equal_magnitude_neighbor():
    """A neighbor with the same magnitude adds flux_ratio = 1.0 → 50% contamination."""
    r = estimate_crowding(12.0, [12.0], [10.0])
    assert r.flag == "CROWDED"  # 0.5 > 0.10
    assert abs(r.flux_ratio - 1.0) < 1e-6
    assert abs(r.contamination_fraction - 0.5) < 1e-6


def test_faint_neighbor_low_contamination():
    """Neighbor 5 mags fainter → flux_ratio = 10^(-2) = 0.01 → OK."""
    r = estimate_crowding(12.0, [17.0], [5.0])
    assert r.flag == "OK"
    assert r.contamination_fraction < 0.10


def test_neighbor_outside_aperture_excluded():
    """Neighbor beyond aperture radius should not count."""
    r = estimate_crowding(12.0, [12.0], [30.0], aperture_radius_arcsec=21.0)
    assert r.n_neighbors == 0
    assert r.flux_ratio == 0.0


def test_multiple_neighbors():
    r = estimate_crowding(12.0, [13.0, 14.0, 15.0], [5.0, 10.0, 15.0])
    assert r.n_neighbors == 3
    assert r.flux_ratio > 0


def test_crowded_threshold_custom():
    r = estimate_crowding(12.0, [12.0], [5.0], crowded_threshold=0.60)
    assert r.flag == "OK"  # 50% < 60% threshold


def test_invalid_nan_target_mag():
    r = estimate_crowding(float("nan"), [12.0], [5.0])
    assert r.flag == "INVALID"


def test_mismatched_lists():
    r = estimate_crowding(12.0, [12.0, 13.0], [5.0])  # len mismatch
    assert r.flag == "INVALID"


def test_flux_ratio_formula():
    """flux_ratio = sum(10^(-0.4*(m_i - target_mag)))."""
    target = 12.0
    neigh = 14.0
    expected_ratio = 10.0 ** (-0.4 * (neigh - target))
    r = estimate_crowding(target, [neigh], [10.0])
    assert abs(r.flux_ratio - expected_ratio) < 1e-5


def test_crowding_metric_formula():
    r = estimate_crowding(12.0, [12.0], [5.0])
    expected_metric = 1.0 / (1.0 + r.flux_ratio)
    assert abs(r.crowding_metric - expected_metric) < 1e-8


def test_contamination_plus_crowding_equals_one():
    r = estimate_crowding(12.0, [13.0], [5.0])
    assert abs(r.crowding_metric + r.contamination_fraction - 1.0) < 1e-6


def test_format_contains_keywords():
    r = estimate_crowding(12.0, [20.0], [5.0])
    text = format_crowding(r)
    assert "Crowding" in text
    assert "contamination" in text.lower()
    assert r.flag in text


def test_format_crowded():
    r = estimate_crowding(12.0, [12.0], [5.0])
    text = format_crowding(r)
    assert "CROWDED" in text
