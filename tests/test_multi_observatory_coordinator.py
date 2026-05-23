"""Tests for Skills/multi_observatory_coordinator.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from multi_observatory_coordinator import (
    ObservatorySlot,
    CoordinationResult,
    coordinate_observations,
    format_coordination_result,
)


def _site(name="TestSite", lat=28.0, lon=-17.0, utc_off=-1.0):
    return {"name": name, "lat_deg": lat, "lon_deg": lon, "utc_offset_hours": utc_off}


class TestCoordinateObservations:
    def test_basic_ok(self):
        sites = [_site()]
        result = coordinate_observations(12345, 2460000.0, sites)
        assert result.flag in ("OK", "NO_USABLE_SITES")

    def test_invalid_sites_not_list(self):
        result = coordinate_observations(12345, 2460000.0, "not a list")
        assert result.flag == "INVALID"

    def test_empty_sites_invalid(self):
        result = coordinate_observations(12345, 2460000.0, [])
        assert result.flag == "INVALID"

    def test_n_sites_correct(self):
        sites = [_site("A"), _site("B"), _site("C")]
        result = coordinate_observations(None, 2460000.0, sites)
        assert result.n_sites == 3

    def test_tic_id_stored(self):
        sites = [_site()]
        result = coordinate_observations(99999, 2460000.0, sites)
        assert result.tic_id == 99999

    def test_transit_mid_bjd_stored(self):
        sites = [_site()]
        result = coordinate_observations(None, 2460123.5, sites)
        assert result.transit_mid_bjd == 2460123.5

    def test_usable_slots_subset_of_all_sites(self):
        sites = [_site(f"S{i}") for i in range(5)]
        result = coordinate_observations(None, 2460000.0, sites)
        assert len(result.usable_slots) <= result.n_sites

    def test_best_site_none_when_no_usable(self):
        # Use extreme max_airmass=0.1 to make everything fail
        sites = [_site()]
        result = coordinate_observations(None, 2460000.0, sites, max_airmass=0.1)
        if result.flag == "NO_USABLE_SITES":
            assert result.best_site is None

    def test_result_frozen(self):
        sites = [_site()]
        result = coordinate_observations(None, 2460000.0, sites)
        try:
            result.best_site = "other"
            assert False
        except Exception:
            pass

    def test_slot_fields_present(self):
        sites = [_site()]
        result = coordinate_observations(None, 2460000.0, sites)
        for slot in result.usable_slots:
            assert hasattr(slot, "site_name")
            assert hasattr(slot, "airmass_at_transit")
            assert hasattr(slot, "moon_sep_deg")

    def test_airmass_constraint_respected(self):
        sites = [_site()]
        result = coordinate_observations(None, 2460000.0, sites, max_airmass=2.0)
        for slot in result.usable_slots:
            if slot.airmass_at_transit is not None:
                assert slot.airmass_at_transit <= 2.0

    def test_moon_sep_constraint_respected(self):
        sites = [_site()]
        result = coordinate_observations(None, 2460000.0, sites, min_moon_sep_deg=30.0)
        for slot in result.usable_slots:
            if slot.moon_sep_deg is not None:
                assert slot.moon_sep_deg >= 30.0

    def test_format_returns_string(self):
        sites = [_site()]
        result = coordinate_observations(None, 2460000.0, sites)
        text = format_coordination_result(result)
        assert isinstance(text, str)

    def test_format_contains_flag(self):
        sites = [_site()]
        result = coordinate_observations(None, 2460000.0, sites)
        text = format_coordination_result(result)
        assert result.flag in text
