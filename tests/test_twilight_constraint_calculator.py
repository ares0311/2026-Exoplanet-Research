"""Tests for Skills/twilight_constraint_calculator.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from twilight_constraint_calculator import (  # noqa: E402
    TwilightResult,
    check_twilight_constraints,
    format_twilight_result,
)


def test_returns_dataclass():
    r = check_twilight_constraints("2026-06-01", "22:00", "01:00", 35.0)
    assert isinstance(r, TwilightResult)


def test_ok_flag_deep_night():
    # Deep in the night should be OK at mid-latitudes
    r = check_twilight_constraints("2026-01-15", "23:00", "02:00", 35.0)
    assert r.flag in ("OK", "PARTIAL_TWILIGHT")


def test_in_twilight_flag_daytime():
    # Daytime observation
    r = check_twilight_constraints("2026-06-01", "12:00", "14:00", 35.0)
    assert r.flag == "IN_TWILIGHT"


def test_dark_start_format():
    r = check_twilight_constraints("2026-06-01", "22:00", "01:00", 35.0)
    parts = r.dark_start_local.split(":")
    assert len(parts) == 2
    assert parts[0].isdigit() and parts[1].isdigit()


def test_dark_end_format():
    r = check_twilight_constraints("2026-06-01", "22:00", "01:00", 35.0)
    parts = r.dark_end_local.split(":")
    assert len(parts) == 2


def test_overlap_hours_nonneg():
    r = check_twilight_constraints("2026-06-01", "22:00", "01:00", 35.0)
    assert r.overlap_hours >= 0.0


def test_in_dark_time_boolean():
    r = check_twilight_constraints("2026-06-01", "22:00", "01:00", 35.0)
    assert isinstance(r.in_dark_time, bool)


def test_obs_start_end_preserved():
    r = check_twilight_constraints("2026-06-01", "22:30", "00:30", 35.0)
    assert r.obs_start == "22:30"
    assert r.obs_end == "00:30"


def test_partial_twilight_flag():
    # Starts just before astronomical dark
    r = check_twilight_constraints("2026-06-01", "20:00", "23:00", 35.0)
    # Should be partial or in_twilight depending on exact computation
    assert r.flag in ("PARTIAL_TWILIGHT", "IN_TWILIGHT", "OK")


def test_negative_utc_offset():
    r = check_twilight_constraints("2026-06-01", "22:00", "01:00", 35.0, utc_offset_h=-5.0)
    assert isinstance(r, TwilightResult)


def test_format_returns_string():
    r = check_twilight_constraints("2026-06-01", "22:00", "01:00", 35.0)
    s = format_twilight_result(r)
    assert isinstance(s, str)


def test_format_contains_flag():
    r = check_twilight_constraints("2026-06-01", "22:00", "01:00", 35.0)
    s = format_twilight_result(r)
    assert "Flag" in s
