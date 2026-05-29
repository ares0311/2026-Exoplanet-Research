"""Tests for Skills/guiding_star_selector.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from guiding_star_selector import (  # noqa: E402
    GuideStarResult,
    select_guide_star,
    format_guide_star,
)


STARS = [
    {"mag": 11.0, "sep_arcsec": 120.0},
    {"mag": 13.5, "sep_arcsec": 300.0},
    {"mag": 9.0, "sep_arcsec": 50.0},   # too bright? No — min 8.0
    {"mag": 15.0, "sep_arcsec": 200.0},  # too faint
]


def test_returns_dataclass():
    r = select_guide_star(STARS)
    assert isinstance(r, GuideStarResult)


def test_no_stars_returns_no_guide():
    r = select_guide_star([])
    assert r.flag == "NO_GUIDE_STAR"
    assert r.selected_index is None


def test_selects_within_mag_range():
    r = select_guide_star(STARS)
    if r.selected_index is not None:
        assert 8.0 <= r.selected_mag <= 14.0


def test_selects_within_sep_range():
    r = select_guide_star(STARS)
    if r.selected_index is not None:
        assert 30.0 <= r.selected_sep_arcsec <= 600.0


def test_ok_flag_multiple_candidates():
    # Provide multiple valid stars
    stars = [
        {"mag": 10.0, "sep_arcsec": 100.0},
        {"mag": 11.0, "sep_arcsec": 200.0},
    ]
    r = select_guide_star(stars)
    assert r.flag == "OK"
    assert r.n_candidates == 2


def test_too_few_candidates_flag():
    stars = [{"mag": 11.0, "sep_arcsec": 200.0}]
    r = select_guide_star(stars)
    assert r.flag == "TOO_FEW_CANDIDATES"


def test_no_guide_when_all_too_faint():
    stars = [{"mag": 16.0, "sep_arcsec": 200.0}]
    r = select_guide_star(stars)
    assert r.flag == "NO_GUIDE_STAR"


def test_no_guide_when_too_close():
    stars = [{"mag": 11.0, "sep_arcsec": 10.0}]
    r = select_guide_star(stars)
    assert r.flag == "NO_GUIDE_STAR"


def test_optimal_mag_selection():
    # The star closest to midpoint (11.0) should be preferred
    stars = [
        {"mag": 8.1, "sep_arcsec": 100.0},   # far from midpoint
        {"mag": 11.0, "sep_arcsec": 100.0},  # close to midpoint
    ]
    r = select_guide_star(stars)
    assert r.selected_mag == 11.0


def test_format_returns_string():
    r = select_guide_star(STARS)
    s = format_guide_star(r)
    assert isinstance(s, str)


def test_format_contains_flag():
    r = select_guide_star(STARS)
    s = format_guide_star(r)
    assert "Flag" in s


def test_custom_mag_range():
    stars = [{"mag": 7.0, "sep_arcsec": 100.0}]
    r = select_guide_star(stars, guide_mag_min=6.0, guide_mag_max=8.0)
    assert r.selected_mag == 7.0
