"""Tests for Skills/ground_obs_report_builder.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from ground_obs_report_builder import (  # noqa: E402
    GroundObsReport,
    build_ground_obs_report,
    format_ground_obs_report,
)


def _meta(**kwargs):
    base = {
        "target": "TIC-12345",
        "date": "2026-06-01",
        "depth_ppm_measured": 5000.0,
        "depth_ppm_expected": 5000.0,
        "timing_offset_min": 0.0,
    }
    base.update(kwargs)
    return base


def test_returns_dataclass():
    r = build_ground_obs_report(_meta())
    assert isinstance(r, GroundObsReport)


def test_perfect_match_grade_a():
    r = build_ground_obs_report(_meta(timing_offset_min=2.0))
    assert r.quality_grade == "A"


def test_grade_b_slight_offset():
    r = build_ground_obs_report(_meta(
        depth_ppm_measured=6000.0, depth_ppm_expected=5000.0, timing_offset_min=10.0
    ))
    assert r.quality_grade == "B"


def test_grade_d_large_depth_mismatch():
    r = build_ground_obs_report(_meta(
        depth_ppm_measured=100.0, depth_ppm_expected=5000.0
    ))
    assert r.quality_grade == "D"


def test_flag_ok_for_good_match():
    r = build_ground_obs_report(_meta())
    assert r.flag == "OK"


def test_flag_depth_mismatch():
    r = build_ground_obs_report(_meta(
        depth_ppm_measured=500.0, depth_ppm_expected=5000.0
    ))
    assert r.flag == "DEPTH_MISMATCH"


def test_flag_timing_offset():
    r = build_ground_obs_report(_meta(
        depth_ppm_measured=5000.0, depth_ppm_expected=5000.0,
        timing_offset_min=20.0
    ))
    assert r.flag == "TIMING_OFFSET"


def test_depth_ratio_computed():
    r = build_ground_obs_report(_meta(
        depth_ppm_measured=4000.0, depth_ppm_expected=5000.0
    ))
    assert abs(r.depth_ratio - 0.8) < 0.001


def test_target_stored():
    r = build_ground_obs_report(_meta(target="TIC-99999"))
    assert r.target == "TIC-99999"


def test_date_stored():
    r = build_ground_obs_report(_meta(date="2026-12-25"))
    assert r.date == "2026-12-25"


def test_format_returns_string():
    r = build_ground_obs_report(_meta())
    s = format_ground_obs_report(r)
    assert isinstance(s, str)


def test_format_contains_grade():
    r = build_ground_obs_report(_meta())
    s = format_ground_obs_report(r)
    assert "grade" in s.lower() or "Grade" in s
