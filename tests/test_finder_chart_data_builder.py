"""Tests for Skills/finder_chart_data_builder.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from finder_chart_data_builder import (  # noqa: E402
    FinderChartData,
    build_finder_chart_data,
    format_finder_chart_data,
)


STARS = [
    {"ra_deg": 180.01, "dec_deg": 30.01, "mag": 12.0},
    {"ra_deg": 180.05, "dec_deg": 30.05, "mag": 11.0},
    {"ra_deg": 181.0, "dec_deg": 30.0, "mag": 10.0},  # outside 10 arcmin
]


def test_returns_dataclass():
    r = build_finder_chart_data(180.0, 30.0, STARS)
    assert isinstance(r, FinderChartData)


def test_filters_by_field_radius():
    r = build_finder_chart_data(180.0, 30.0, STARS, field_arcmin=10.0)
    assert r.n_stars < len(STARS)


def test_sorted_by_magnitude():
    r = build_finder_chart_data(180.0, 30.0, STARS[:2])
    mags = [s["mag"] for s in r.stars_in_field]
    assert mags == sorted(mags)


def test_empty_catalog_flag():
    r = build_finder_chart_data(180.0, 30.0, [])
    assert r.flag == "NO_CATALOG"
    assert r.n_stars == 0


def test_empty_field_flag():
    far_stars = [{"ra_deg": 185.0, "dec_deg": 30.0, "mag": 12.0}]
    r = build_finder_chart_data(180.0, 30.0, far_stars, field_arcmin=5.0)
    assert r.flag == "EMPTY_FIELD"


def test_ok_flag_when_stars_found():
    r = build_finder_chart_data(180.0, 30.0, STARS[:2], field_arcmin=20.0)
    assert r.flag == "OK"


def test_sep_arcmin_added():
    r = build_finder_chart_data(180.0, 30.0, STARS[:2], field_arcmin=20.0)
    for s in r.stars_in_field:
        assert "sep_arcmin" in s


def test_field_radius_attribute():
    r = build_finder_chart_data(180.0, 30.0, STARS, field_arcmin=15.0)
    assert r.field_arcmin == 15.0


def test_format_returns_string():
    r = build_finder_chart_data(180.0, 30.0, STARS[:2])
    s = format_finder_chart_data(r)
    assert isinstance(s, str)


def test_format_contains_flag():
    r = build_finder_chart_data(180.0, 30.0, STARS[:2])
    s = format_finder_chart_data(r)
    assert "Flag" in s


def test_target_ra_dec_stored():
    r = build_finder_chart_data(123.45, -10.0, STARS)
    assert r.target_ra == 123.45
    assert r.target_dec == -10.0


def test_large_field_includes_more_stars():
    r_small = build_finder_chart_data(180.0, 30.0, STARS, field_arcmin=5.0)
    r_large = build_finder_chart_data(180.0, 30.0, STARS, field_arcmin=100.0)
    assert r_large.n_stars >= r_small.n_stars
