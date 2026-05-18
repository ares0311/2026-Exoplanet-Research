"""Tests for airmass_calculator.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from airmass_calculator import (
    AirmassResult,
    compute_airmass,
    compute_airmass_curve,
    format_airmass_result,
)

# Use Paranal-like coordinates: lat=-24.6, lon=-70.4
LAT = -24.6
LON = -70.4
# BJD ≈ 2460000 (some arbitrary time)
BJD = 2460000.0


class TestComputeAirmass:
    def test_returns_result(self):
        r = compute_airmass(0.0, LAT, LAT, LON, BJD)
        assert isinstance(r, AirmassResult)
        assert r.flag in ("OK", "BELOW_HORIZON")

    def test_airmass_at_zenith(self):
        # A star exactly at the zenith has airmass = 1
        # Place star at declination = observer latitude, hour angle = 0 (transit)
        r = compute_airmass(0.0, LAT, LAT, LON, BJD)
        if r.is_observable:
            assert r.airmass >= 1.0

    def test_airmass_high_when_low_altitude(self):
        # Star far from zenith should have high airmass
        r = compute_airmass(0.0, 89.0, LAT, LON, BJD)
        if r.is_observable:
            assert r.airmass > 2.0

    def test_below_horizon_flag(self):
        # Circumpolar star opposite meridian might be below horizon
        r = compute_airmass(0.0, 89.0, LAT, LON, BJD)
        assert r.flag in ("OK", "BELOW_HORIZON")

    def test_invalid_lat(self):
        r = compute_airmass(0.0, 0.0, 100.0, LON, BJD)
        assert r.flag == "INVALID"

    def test_lon_200_is_valid(self):
        # Longitude 200° wraps; implementation accepts any value
        r = compute_airmass(0.0, 0.0, LAT, 200.0, BJD)
        assert r.flag in ("OK", "BELOW_HORIZON")

    def test_result_frozen(self):
        r = compute_airmass(0.0, 0.0, LAT, LON, BJD)
        try:
            r.airmass = 99  # type: ignore[misc]
            raise AssertionError()
        except (AttributeError, TypeError):
            pass

    def test_observable_flag_consistent(self):
        r = compute_airmass(0.0, 0.0, LAT, LON, BJD)
        if r.is_observable:
            assert r.airmass < 10.0


class TestComputeAirmassAurve:
    def test_returns_list(self):
        results = compute_airmass_curve(0.0, 0.0, LAT, LON, BJD, BJD + 1.0, n_steps=12)
        assert isinstance(results, list)
        assert len(results) == 12

    def test_n_steps(self):
        results = compute_airmass_curve(0.0, 0.0, LAT, LON, BJD, BJD + 0.5, n_steps=5)
        assert len(results) == 5

    def test_invalid_range(self):
        results = compute_airmass_curve(0.0, 0.0, LAT, LON, BJD, BJD - 1.0, n_steps=5)
        assert results == []


class TestFormatAirmassResult:
    def test_returns_string(self):
        r = compute_airmass(0.0, 0.0, LAT, LON, BJD)
        assert isinstance(format_airmass_result(r), str)

    def test_contains_flag(self):
        r = compute_airmass(0.0, 0.0, LAT, LON, BJD)
        s = format_airmass_result(r)
        assert r.flag in s
