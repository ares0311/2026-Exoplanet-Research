"""Tests for Skills/comparison_star_finder.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from comparison_star_finder import (
    ComparisonStarResult,
    NeighbourStar,
    find_comparison_stars,
    format_comparison_star_result,
)


class TestNeighbourStar:
    def test_frozen_dataclass(self):
        s = NeighbourStar(mag=10.0, ra_deg=180.0, dec_deg=0.0)
        assert s.mag == 10.0
        try:
            s.mag = 0.0
            raise AssertionError()
        except Exception:
            pass


class TestComparisonStarResult:
    def test_dataclass_fields(self):
        r = ComparisonStarResult(n_candidates=5, n_selected=2, selected_indices=(0, 2))
        assert r.n_candidates == 5
        assert r.n_selected == 2


class TestFindComparisonStars:
    def _make_neighbour(self, mag, ra_offset_arcmin=5.0):
        """Star slightly offset in RA from (0, 0)."""
        return NeighbourStar(
            mag=mag,
            ra_deg=ra_offset_arcmin / 60.0,
            dec_deg=0.0,
        )

    def test_single_good_star(self):
        star = self._make_neighbour(10.0, 5.0)  # 5 arcmin offset, same mag
        r = find_comparison_stars(10.0, 0.0, 0.0, [star], delta_mag=1.0)
        assert r.n_selected == 1
        assert r.flag == "OK"

    def test_mag_too_different(self):
        star = self._make_neighbour(13.0, 5.0)
        r = find_comparison_stars(10.0, 0.0, 0.0, [star], delta_mag=1.0)
        assert r.n_selected == 0
        assert r.flag == "WARNING"

    def test_too_close(self):
        star = NeighbourStar(mag=10.0, ra_deg=0.001, dec_deg=0.0)  # < 10 arcsec
        r = find_comparison_stars(10.0, 0.0, 0.0, [star], min_sep_arcsec=10.0)
        assert r.n_selected == 0

    def test_too_far(self):
        star = NeighbourStar(mag=10.0, ra_deg=1.0, dec_deg=0.0)  # >60 arcmin
        r = find_comparison_stars(10.0, 0.0, 0.0, [star], max_sep_arcmin=15.0)
        assert r.n_selected == 0

    def test_empty_neighbours(self):
        r = find_comparison_stars(10.0, 0.0, 0.0, [])
        assert r.n_selected == 0
        assert r.flag == "WARNING"

    def test_n_candidates_count(self):
        stars = [self._make_neighbour(10.0, 5.0), self._make_neighbour(13.0, 5.0)]
        r = find_comparison_stars(10.0, 0.0, 0.0, stars)
        assert r.n_candidates == 2

    def test_multiple_selected(self):
        stars = [
            self._make_neighbour(10.0, 5.0),
            self._make_neighbour(9.5, 8.0),
        ]
        r = find_comparison_stars(10.0, 0.0, 0.0, stars)
        assert r.n_selected == 2

    def test_selected_indices_type(self):
        star = self._make_neighbour(10.0, 5.0)
        r = find_comparison_stars(10.0, 0.0, 0.0, [star])
        assert isinstance(r.selected_indices, tuple)

    def test_index_values_correct(self):
        stars = [self._make_neighbour(13.0, 5.0), self._make_neighbour(10.0, 5.0)]
        r = find_comparison_stars(10.0, 0.0, 0.0, stars)
        assert 1 in r.selected_indices
        assert 0 not in r.selected_indices


class TestFormatComparisonStar:
    def test_returns_string(self):
        r = find_comparison_stars(10.0, 0.0, 0.0, [])
        s = format_comparison_star_result(r)
        assert isinstance(s, str)

    def test_contains_n_selected(self):
        r = find_comparison_stars(10.0, 0.0, 0.0, [])
        s = format_comparison_star_result(r)
        assert "0" in s

    def test_no_stars_shows_none(self):
        r = find_comparison_stars(10.0, 0.0, 0.0, [])
        s = format_comparison_star_result(r)
        assert "none" in s.lower()
