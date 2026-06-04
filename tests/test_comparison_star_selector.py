"""Tests for Skills/comparison_star_selector.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from comparison_star_selector import ComparisonSelectionResult, select_comparison_stars

_CANDIDATES = [
    {"star_id": "A", "magnitude": 10.0, "separation_arcsec": 60.0},
    {"star_id": "B", "magnitude": 10.5, "separation_arcsec": 120.0},
    {"star_id": "C", "magnitude": 11.0, "separation_arcsec": 180.0},
    {"star_id": "D", "magnitude": 9.5, "separation_arcsec": 90.0},
    {"star_id": "E", "magnitude": 10.2, "separation_arcsec": 300.0},
    {"star_id": "F", "magnitude": 10.1, "separation_arcsec": 45.0},
]


class TestComparisonStarSelector:
    def test_basic_ok(self) -> None:
        r = select_comparison_stars(10.0, _CANDIDATES)
        assert r.flag == "OK"

    def test_no_candidates(self) -> None:
        r = select_comparison_stars(10.0, [])
        assert r.flag == "NO_CANDIDATES"

    def test_invalid_target_mag(self) -> None:
        r = select_comparison_stars(float("nan"), _CANDIDATES)
        assert r.flag == "INVALID_TARGET_MAG"

    def test_n_recommended_respected(self) -> None:
        r = select_comparison_stars(10.0, _CANDIDATES, n_recommend=3)
        assert r.n_recommended <= 3
        rec = [s for s in r.stars if s.recommended]
        assert len(rec) == r.n_recommended

    def test_scores_sorted_ascending(self) -> None:
        r = select_comparison_stars(10.0, _CANDIDATES)
        scores = [s.composite_score for s in r.stars]
        assert scores == sorted(scores)

    def test_mag_filter_applied(self) -> None:
        r = select_comparison_stars(10.0, _CANDIDATES, max_mag_diff=0.3)
        for s in r.stars:
            assert s.magnitude_diff <= 0.3 + 1e-6

    def test_separation_filter_applied(self) -> None:
        r = select_comparison_stars(10.0, _CANDIDATES, max_separation_arcsec=100.0)
        for s in r.stars:
            assert s.separation_arcsec <= 100.0

    def test_color_scoring_used(self) -> None:
        cands = [{"star_id": "X", "magnitude": 10.0, "separation_arcsec": 60.0,
                  "color_index": 0.6}]
        r = select_comparison_stars(10.0, cands, target_color=0.5)
        assert r.flag == "OK"
        assert r.stars[0].color_diff is not None

    def test_result_frozen(self) -> None:
        r = select_comparison_stars(10.0, _CANDIDATES)
        assert isinstance(r, ComparisonSelectionResult)
        try:
            object.__setattr__(r, "flag", "x")
            raise AssertionError()
        except Exception:
            pass

    def test_format_output(self) -> None:
        from comparison_star_selector import format_comparison_result
        r = select_comparison_stars(10.0, _CANDIDATES)
        s = format_comparison_result(r)
        assert "|" in s

    def test_too_close_excluded(self) -> None:
        cands = [{"star_id": "X", "magnitude": 10.0, "separation_arcsec": 1.0}]
        r = select_comparison_stars(10.0, cands, min_separation_arcsec=5.0)
        assert r.n_candidates == 0

    def test_n_candidates_count(self) -> None:
        r = select_comparison_stars(10.0, _CANDIDATES)
        assert r.n_candidates == len(r.stars)
