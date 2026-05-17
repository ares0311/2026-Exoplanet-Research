"""Tests for Skills.nearby_star_checker."""
from __future__ import annotations

import pytest
from Skills.nearby_star_checker import (
    NearbyStar,
    NearbyStarResult,
    check_nearby_stars,
    format_nearby_result,
)


def _make_catalog(*rows):
    """Return a catalog_fn returning the given row dicts."""
    def catalog_fn(ra, dec, radius):
        return list(rows)
    return catalog_fn


_TARGET = {"tic_id": 99, "ra": 10.0, "dec": 20.0, "tmag": 11.0}


class TestCheckNearbyStars:
    def test_returns_result(self) -> None:
        r = check_nearby_stars(1, 0.0, 0.0, 12.0)
        assert isinstance(r, NearbyStarResult)

    def test_no_neighbors_empty(self) -> None:
        r = check_nearby_stars(1, 0.0, 0.0, 12.0,
                               catalog_fn=_make_catalog())
        assert len(r.neighbors) == 0

    def test_target_excluded_from_neighbors(self) -> None:
        same = {"tic_id": 1, "ra": 0.0, "dec": 0.0, "tmag": 12.0}
        r = check_nearby_stars(1, 0.0, 0.0, 12.0,
                               catalog_fn=_make_catalog(same))
        assert len(r.neighbors) == 0

    def test_neighbor_found(self) -> None:
        neighbor = {"tic_id": 2, "ra": 0.001, "dec": 0.001, "tmag": 13.0}
        r = check_nearby_stars(1, 0.0, 0.0, 12.0,
                               catalog_fn=_make_catalog(neighbor))
        assert len(r.neighbors) == 1

    def test_too_faint_excluded(self) -> None:
        faint = {"tic_id": 3, "ra": 0.001, "dec": 0.001, "tmag": 22.0}
        r = check_nearby_stars(1, 0.0, 0.0, 12.0,
                               delta_tmag_limit=8.0,
                               catalog_fn=_make_catalog(faint))
        assert len(r.neighbors) == 0

    def test_dilution_fraction_positive(self) -> None:
        neighbor = {"tic_id": 2, "ra": 0.001, "dec": 0.0, "tmag": 13.0}
        r = check_nearby_stars(1, 0.0, 0.0, 12.0,
                               catalog_fn=_make_catalog(neighbor))
        assert r.neighbors[0].dilution_fraction > 0.0

    def test_total_dilution_sum(self) -> None:
        n1 = {"tic_id": 2, "ra": 0.001, "dec": 0.0, "tmag": 13.0}
        n2 = {"tic_id": 3, "ra": 0.0, "dec": 0.001, "tmag": 14.0}
        r = check_nearby_stars(1, 0.0, 0.0, 12.0,
                               catalog_fn=_make_catalog(n1, n2))
        expected = sum(s.dilution_fraction for s in r.neighbors)
        assert r.total_dilution_fraction == pytest.approx(expected, rel=1e-5)

    def test_n_significant_count(self) -> None:
        close = {"tic_id": 2, "ra": 0.001, "dec": 0.0, "tmag": 12.5}  # Δ=0.5 <5
        r = check_nearby_stars(1, 0.0, 0.0, 12.0,
                               catalog_fn=_make_catalog(close))
        assert r.n_significant >= 1

    def test_sorted_by_separation(self) -> None:
        n1 = {"tic_id": 2, "ra": 0.002, "dec": 0.0, "tmag": 13.0}
        n2 = {"tic_id": 3, "ra": 0.001, "dec": 0.0, "tmag": 13.5}
        r = check_nearby_stars(1, 0.0, 0.0, 12.0,
                               catalog_fn=_make_catalog(n1, n2))
        if len(r.neighbors) >= 2:
            assert r.neighbors[0].separation_arcsec <= r.neighbors[1].separation_arcsec

    def test_target_tic_id_stored(self) -> None:
        r = check_nearby_stars(42, 0.0, 0.0, 12.0)
        assert r.target_tic_id == 42

    def test_target_tmag_stored(self) -> None:
        r = check_nearby_stars(1, 0.0, 0.0, 11.5)
        assert r.target_tmag == pytest.approx(11.5)


class TestFormatNearbyResult:
    def test_returns_string(self) -> None:
        r = check_nearby_stars(1, 0.0, 0.0, 12.0)
        assert isinstance(format_nearby_result(r), str)

    def test_target_tic_id_in_output(self) -> None:
        r = check_nearby_stars(999, 0.0, 0.0, 12.0)
        assert "999" in format_nearby_result(r)
