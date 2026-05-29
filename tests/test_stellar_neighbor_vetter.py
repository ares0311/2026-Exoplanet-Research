"""Tests for Skills/stellar_neighbor_vetter.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from stellar_neighbor_vetter import (
    NeighborVettingResult,
    vet_stellar_neighbors,
    format_neighbor_vetting,
)

_NEIGHBORS_CLEAN = [
    {"catalog_id": "TIC-999", "separation_arcsec": 50.0, "delta_mag": 5.0},
]
_NEIGHBORS_CONTAMINATED = [
    {"catalog_id": "TIC-888", "separation_arcsec": 10.0, "delta_mag": 3.0},
]
_NEIGHBORS_SEVERE = [
    {"catalog_id": "TIC-777", "separation_arcsec": 5.0, "delta_mag": 0.0},
]


def test_clean_result():
    result = vet_stellar_neighbors(100, 1000.0, _NEIGHBORS_CLEAN)
    assert result.flag == "CLEAN"


def test_contaminated_result():
    result = vet_stellar_neighbors(100, 1000.0, _NEIGHBORS_CONTAMINATED)
    assert result.flag == "CONTAMINATED"


def test_severe_contamination():
    result = vet_stellar_neighbors(100, 1000.0, _NEIGHBORS_SEVERE)
    assert result.flag == "SEVERE_CONTAMINATION"


def test_no_neighbors():
    result = vet_stellar_neighbors(100, 1000.0, [])
    assert result.flag == "NO_NEIGHBORS"


def test_n_neighbors_correct():
    result = vet_stellar_neighbors(100, 1000.0, _NEIGHBORS_CONTAMINATED)
    assert result.n_neighbors == 1


def test_n_contaminants():
    result = vet_stellar_neighbors(100, 1000.0, _NEIGHBORS_CONTAMINATED)
    assert result.n_contaminants == 1


def test_flux_fraction_computed():
    result = vet_stellar_neighbors(100, 1000.0, _NEIGHBORS_CONTAMINATED)
    n = result.neighbors[0]
    # delta_mag=3.0 → flux_fraction = 10^(-1.2) ≈ 0.063
    assert 0.04 < n.flux_fraction < 0.10


def test_diluted_depth_positive():
    result = vet_stellar_neighbors(100, 1000.0, _NEIGHBORS_CONTAMINATED)
    assert result.neighbors[0].diluted_depth_ppm > 0


def test_sorted_by_separation():
    neighbors = [
        {"catalog_id": "far", "separation_arcsec": 30.0, "delta_mag": 2.0},
        {"catalog_id": "close", "separation_arcsec": 5.0, "delta_mag": 1.0},
    ]
    result = vet_stellar_neighbors(100, 1000.0, neighbors)
    assert result.neighbors[0].separation_arcsec <= result.neighbors[1].separation_arcsec


def test_tic_id_stored():
    result = vet_stellar_neighbors(12345, 1000.0, [])
    assert result.tic_id == 12345


def test_aperture_radius_respected():
    # Neighbor at 25 arcsec; aperture 21 → outside → CLEAN
    neighbors = [{"catalog_id": "X", "separation_arcsec": 25.0, "delta_mag": 0.5}]
    result = vet_stellar_neighbors(100, 1000.0, neighbors, aperture_radius_arcsec=21.0)
    assert result.flag == "CLEAN"


def test_format_returns_string():
    result = vet_stellar_neighbors(100, 1000.0, _NEIGHBORS_CONTAMINATED)
    text = format_neighbor_vetting(result)
    assert isinstance(text, str)
    assert "Neighbor" in text


def test_format_no_neighbors():
    result = vet_stellar_neighbors(100, 1000.0, [])
    text = format_neighbor_vetting(result)
    assert "No neighbors" in text
