"""Tests for Skills/differential_photometry_optimizer.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from differential_photometry_optimizer import (  # noqa: E402
    DiffPhotResult,
    format_diff_phot,
    optimize_diff_photometry,
)

STARS = [
    {"mag": 12.0, "variable": False, "in_field": True},
    {"mag": 12.5, "variable": False, "in_field": True},
    {"mag": 13.0, "variable": False, "in_field": True},
    {"mag": 14.5, "variable": False, "in_field": True},  # too faint (delta > 1.5)
    {"mag": 12.2, "variable": True, "in_field": True},   # variable — excluded
    {"mag": 11.8, "variable": False, "in_field": False},  # out of field — excluded
]


def test_returns_dataclass():
    r = optimize_diff_photometry(12.0, STARS)
    assert isinstance(r, DiffPhotResult)


def test_no_stars_flag():
    r = optimize_diff_photometry(12.0, [])
    assert r.flag == "NO_STARS"
    assert r.n_selected == 0


def test_excludes_variable_stars():
    stars = [
        {"mag": 12.0, "variable": True, "in_field": True},
    ]
    r = optimize_diff_photometry(12.0, stars)
    assert r.flag == "NO_STARS"


def test_excludes_out_of_field_stars():
    stars = [
        {"mag": 12.0, "variable": False, "in_field": False},
    ]
    r = optimize_diff_photometry(12.0, stars)
    assert r.flag == "NO_STARS"


def test_excludes_too_faint_stars():
    stars = [{"mag": 15.0, "variable": False, "in_field": True}]
    r = optimize_diff_photometry(12.0, stars, max_delta_mag=1.5)
    assert r.flag == "NO_STARS"


def test_ok_flag_enough_stars():
    stars = [
        {"mag": 12.0 + i * 0.2, "variable": False, "in_field": True}
        for i in range(5)
    ]
    r = optimize_diff_photometry(12.0, stars, min_stars=3)
    assert r.flag == "OK"


def test_few_stars_flag():
    stars = [{"mag": 12.0, "variable": False, "in_field": True}]
    r = optimize_diff_photometry(12.0, stars, min_stars=3)
    assert r.flag == "FEW_STARS"


def test_selected_indices_valid():
    r = optimize_diff_photometry(12.0, STARS)
    for idx in r.selected_indices:
        assert 0 <= idx < len(STARS)


def test_precision_positive():
    r = optimize_diff_photometry(12.0, STARS)
    assert r.expected_precision_ppm > 0


def test_more_stars_better_precision():
    few = [{"mag": 12.0, "variable": False, "in_field": True}]
    many = [{"mag": 12.0 + i * 0.1, "variable": False, "in_field": True} for i in range(9)]
    r1 = optimize_diff_photometry(12.0, few)
    r2 = optimize_diff_photometry(12.0, many)
    assert r2.expected_precision_ppm < r1.expected_precision_ppm


def test_format_returns_string():
    r = optimize_diff_photometry(12.0, STARS)
    s = format_diff_phot(r)
    assert isinstance(s, str)


def test_format_contains_flag():
    r = optimize_diff_photometry(12.0, STARS)
    s = format_diff_phot(r)
    assert "Flag" in s
