"""Tests for Skills/seasonal_visibility_planner.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from seasonal_visibility_planner import format_seasonal_visibility, plan_seasonal_visibility


def test_basic_ok():
    r = plan_seasonal_visibility(90.0, 30.0)
    assert r.flag == "OK"
    assert len(r.monthly) == 12


def test_circumpolar_always_high():
    # Object near north celestial pole, seen from mid-latitude
    r = plan_seasonal_visibility(0.0, 89.0, site_lat_deg=45.0, min_altitude_deg=10.0)
    assert r.n_observable_months > 6


def test_never_visible_from_north():
    # Very southern object, viewed from high north lat
    r = plan_seasonal_visibility(0.0, -89.0, site_lat_deg=70.0, min_altitude_deg=20.0)
    assert r.n_observable_months == 0


def test_twelve_months_returned():
    r = plan_seasonal_visibility(180.0, 0.0)
    assert len(r.monthly) == 12


def test_month_numbers_sequential():
    r = plan_seasonal_visibility(180.0, 0.0)
    months = [m.month for m in r.monthly]
    assert months == list(range(1, 13))


def test_sun_ra_changes_by_month():
    r = plan_seasonal_visibility(180.0, 0.0)
    sun_ras = [m.sun_ra_deg for m in r.monthly]
    assert len(set(sun_ras)) > 6  # Sun RA changes through the year


def test_max_altitude_consistent():
    # Max altitude should not depend on month
    r = plan_seasonal_visibility(90.0, 30.0)
    alts = {m.max_altitude_deg for m in r.monthly}
    assert len(alts) == 1


def test_invalid_dec_too_large():
    r = plan_seasonal_visibility(0.0, 95.0)
    assert r.flag == "INVALID"


def test_invalid_dec_too_small():
    r = plan_seasonal_visibility(0.0, -91.0)
    assert r.flag == "INVALID"


def test_invalid_lat_out_of_range():
    r = plan_seasonal_visibility(0.0, 0.0, site_lat_deg=95.0)
    assert r.flag == "INVALID"


def test_invalid_nan_ra():
    r = plan_seasonal_visibility(float("nan"), 0.0)
    assert r.flag == "INVALID"


def test_format_ok():
    r = plan_seasonal_visibility(90.0, 30.0)
    text = format_seasonal_visibility(r)
    assert "Seasonal Visibility" in text
    assert "OK" in text


def test_format_shows_months():
    r = plan_seasonal_visibility(90.0, 30.0)
    text = format_seasonal_visibility(r)
    assert "Jan" in text


def test_observable_months_count():
    r = plan_seasonal_visibility(90.0, 30.0)
    manual_count = sum(1 for m in r.monthly if m.is_observable)
    assert r.n_observable_months == manual_count
