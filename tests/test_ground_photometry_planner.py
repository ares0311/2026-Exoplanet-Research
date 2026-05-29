"""Tests for Skills/ground_photometry_planner.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from ground_photometry_planner import (  # noqa: E402
    GroundPhotPlan,
    plan_ground_photometry,
    format_ground_phot_plan,
)


def test_basic_plan_returns_dataclass():
    r = plan_ground_photometry(180.0, 30.0, 2458000.0, 2.0, 30.0, 0.0)
    assert isinstance(r, GroundPhotPlan)


def test_ok_flag_for_high_altitude():
    r = plan_ground_photometry(180.0, 45.0, 2458000.0, 2.0, 45.0, 0.0)
    assert r.flag == "OK"
    assert r.min_altitude_deg > 20.0


def test_unobservable_flag_for_zero_duration():
    r = plan_ground_photometry(180.0, 30.0, 2458000.0, 0.0, 30.0, 0.0)
    assert r.flag == "UNOBSERVABLE"


def test_low_altitude_flag():
    # target at declination far from latitude
    r = plan_ground_photometry(0.0, -60.0, 2458000.0, 2.0, 50.0, 0.0)
    assert r.flag in ("LOW_ALTITUDE", "UNOBSERVABLE")


def test_baseline_padding():
    r = plan_ground_photometry(180.0, 40.0, 2458000.0, 3.0, 40.0, 0.0)
    # 3h transit + 2*3h baseline = 9h total
    assert abs(r.duration_with_baseline_hours - 9.0) < 0.01


def test_obs_start_end_are_strings():
    r = plan_ground_photometry(180.0, 30.0, 2458000.0, 2.0, 30.0, 0.0)
    assert isinstance(r.obs_start_utc, str)
    assert isinstance(r.obs_end_utc, str)


def test_airmass_positive():
    r = plan_ground_photometry(180.0, 35.0, 2458000.0, 2.0, 35.0, 0.0)
    assert r.max_airmass >= 1.0


def test_format_returns_string():
    r = plan_ground_photometry(180.0, 30.0, 2458000.0, 2.0, 30.0, 0.0)
    s = format_ground_phot_plan(r)
    assert isinstance(s, str)
    assert "Flag" in s


def test_format_contains_key_fields():
    r = plan_ground_photometry(180.0, 30.0, 2458000.0, 2.0, 30.0, 0.0)
    s = format_ground_phot_plan(r)
    assert "Obs start" in s
    assert "Obs end" in s


def test_polar_target_unobservable():
    # Target dec far outside site lat range
    r = plan_ground_photometry(180.0, 89.0, 2458000.0, 2.0, -30.0, 0.0)
    assert r.flag in ("LOW_ALTITUDE", "UNOBSERVABLE")


def test_various_durations():
    for dur in [0.5, 1.0, 4.0, 8.0]:
        r = plan_ground_photometry(180.0, 45.0, 2458000.0, dur, 45.0, 0.0)
        assert r.duration_with_baseline_hours > dur


def test_min_observable_alt_kwarg():
    r = plan_ground_photometry(
        180.0, 30.0, 2458000.0, 2.0, 30.0, min_observable_alt_deg=30.0
    )
    assert r.flag in ("OK", "LOW_ALTITUDE", "UNOBSERVABLE")
