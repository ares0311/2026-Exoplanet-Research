"""Tests for Skills/astroimagej_region_writer.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from astroimagej_region_writer import (
    RegionEntry,
    format_aij_region_result,
    write_aij_region,
)


class TestRegionEntry:
    def test_basic_creation(self):
        e = RegionEntry(ra_deg=10.0, dec_deg=-20.0, radius_arcsec=5.0, label="T", color="green")
        assert e.ra_deg == 10.0
        assert e.label == "T"

    def test_frozen(self):
        e = RegionEntry(ra_deg=1.0, dec_deg=2.0, radius_arcsec=3.0, label="X", color="red")
        try:
            e.ra_deg = 99.0
            raise AssertionError("Should be frozen")
        except Exception:
            pass


class TestWriteAijRegion:
    def _make_entry(self, ra=10.0, dec=-5.0, radius=5.0, label="T", color="green"):
        return RegionEntry(ra_deg=ra, dec_deg=dec, radius_arcsec=radius, label=label, color=color)

    def test_empty_entries_empty(self):
        result = write_aij_region([])
        assert result.flag == "EMPTY"
        assert result.n_entries == 0

    def test_single_entry(self):
        result = write_aij_region([self._make_entry()])
        assert result.flag == "OK"
        assert result.n_entries == 1

    def test_region_text_is_string(self):
        result = write_aij_region([self._make_entry()])
        assert isinstance(result.region_text, str)

    def test_ra_in_output(self):
        result = write_aij_region([self._make_entry(ra=123.456)])
        assert "123.456" in result.region_text

    def test_dec_in_output(self):
        result = write_aij_region([self._make_entry(dec=-45.678)])
        assert "-45.678" in result.region_text

    def test_multiple_entries(self):
        entries = [self._make_entry(ra=i * 10.0) for i in range(3)]
        result = write_aij_region(entries)
        assert result.n_entries == 3

    def test_invalid_annulus_inner_ge_outer(self):
        result = write_aij_region(
            [self._make_entry()],
            inner_annulus_arcsec=50.0,
            outer_annulus_arcsec=30.0,
        )
        assert result.flag == "INVALID"

    def test_annulus_equal_invalid(self):
        result = write_aij_region(
            [self._make_entry()],
            inner_annulus_arcsec=30.0,
            outer_annulus_arcsec=30.0,
        )
        assert result.flag == "INVALID"

    def test_custom_annulus_values(self):
        result = write_aij_region(
            [self._make_entry()],
            inner_annulus_arcsec=25.0,
            outer_annulus_arcsec=45.0,
        )
        assert result.flag == "OK"

    def test_invalid_entries_type(self):
        result = write_aij_region("not a list")
        assert result.flag == "INVALID"

    def test_label_in_output(self):
        result = write_aij_region([self._make_entry(label="COMP1")])
        assert "COMP1" in result.region_text


class TestFormatRegionResult:
    def test_returns_string(self):
        e = RegionEntry(ra_deg=1.0, dec_deg=2.0, radius_arcsec=3.0, label="T", color="green")
        result = write_aij_region([e])
        text = format_aij_region_result(result)
        assert isinstance(text, str)

    def test_contains_flag(self):
        e = RegionEntry(ra_deg=1.0, dec_deg=2.0, radius_arcsec=3.0, label="T", color="green")
        result = write_aij_region([e])
        text = format_aij_region_result(result)
        assert result.flag in text

    def test_contains_entry_count(self):
        e = RegionEntry(ra_deg=1.0, dec_deg=2.0, radius_arcsec=3.0, label="T", color="green")
        result = write_aij_region([e, e])
        text = format_aij_region_result(result)
        assert "2" in text
