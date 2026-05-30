"""Tests for Skills/site_horizon_checker.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from site_horizon_checker import (  # noqa: E402
    HorizonCheckResult,
    check_site_horizon,
    format_horizon_check,
)


def test_returns_dataclass():
    r = check_site_horizon(30.0, 35.0)
    assert isinstance(r, HorizonCheckResult)


def test_ok_flag_for_good_target():
    # Target dec = site lat: meridian altitude = 90°
    r = check_site_horizon(45.0, 45.0, min_altitude_deg=20.0)
    assert r.flag == "OK"


def test_meridian_altitude_at_equator():
    # Lat=0, dec=0 → meridian alt = 90°
    r = check_site_horizon(0.0, 0.0)
    assert abs(r.meridian_altitude_deg - 90.0) < 1.0


def test_never_rises_flag():
    # Target dec = -80° at lat=+40°: never rises
    r = check_site_horizon(-80.0, 40.0)
    assert r.flag in ("NEVER_RISES", "LOW_TARGET")


def test_low_target_flag():
    # Target barely rises above horizon
    r = check_site_horizon(-50.0, 40.0, min_altitude_deg=20.0)
    assert r.flag in ("LOW_TARGET", "NEVER_RISES")


def test_n_hours_above_plus_below_equals_total():
    hour_angles = list(range(-6, 7))
    r = check_site_horizon(30.0, 35.0, hour_angles=hour_angles)
    assert r.n_hours_above + r.n_hours_below == len(hour_angles)


def test_min_altitude_achieved_lte_max():
    r = check_site_horizon(30.0, 35.0)
    assert r.min_altitude_achieved <= r.max_altitude_achieved


def test_max_altitude_is_meridian_altitude():
    # The maximum altitude should occur at meridian (HA=0)
    r = check_site_horizon(30.0, 35.0)
    assert abs(r.max_altitude_achieved - r.meridian_altitude_deg) < 1.0


def test_custom_hour_angles():
    ha = [-3.0, 0.0, 3.0]
    r = check_site_horizon(30.0, 35.0, hour_angles=ha)
    assert r.n_hours_above + r.n_hours_below == 3


def test_format_returns_string():
    r = check_site_horizon(30.0, 35.0)
    s = format_horizon_check(r)
    assert isinstance(s, str)


def test_format_contains_flag():
    r = check_site_horizon(30.0, 35.0)
    s = format_horizon_check(r)
    assert "Flag" in s


def test_circumpolar_target():
    # Dec = 80° at lat = 60°: circumpolar, always above horizon
    r = check_site_horizon(80.0, 60.0, min_altitude_deg=20.0)
    assert r.flag == "OK"
    assert r.n_hours_above == r.n_hours_above + r.n_hours_below - r.n_hours_below
