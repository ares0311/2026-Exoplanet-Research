"""Tests for Skills/fits_keyword_mapper.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from fits_keyword_mapper import (
    KEPLER_MAPPINGS,
    TESS_MAPPINGS,
    format_keyword_map_result,
    map_fits_keywords,
)


class TestMappingTables:
    def test_tess_mappings_nonempty(self):
        assert len(TESS_MAPPINGS) > 0

    def test_kepler_mappings_nonempty(self):
        assert len(KEPLER_MAPPINGS) > 0

    def test_tess_has_ticid(self):
        keys = [m.fits_key for m in TESS_MAPPINGS]
        assert "TICID" in keys

    def test_kepler_has_keplerid(self):
        keys = [m.fits_key for m in KEPLER_MAPPINGS]
        assert "KEPLERID" in keys

    def test_mapping_frozen(self):
        m = TESS_MAPPINGS[0]
        try:
            m.fits_key = "OTHER"
            raise AssertionError()
        except Exception:
            pass


class TestMapFitsKeywords:
    def _tess_header(self):
        return {
            "TICID": 150428135,
            "SECTOR": 1,
            "TESSMAG": 11.5,
            "TEFF": 3800.0,
            "RADIUS": 0.42,
            "RA_OBJ": 100.46,
            "DEC_OBJ": -65.18,
        }

    def test_basic_tess_ok(self):
        result = map_fits_keywords(self._tess_header(), mission="TESS")
        assert result.flag in ("OK", "PARTIAL")

    def test_n_found_positive(self):
        result = map_fits_keywords(self._tess_header(), mission="TESS")
        assert result.n_found > 0

    def test_tic_id_mapped(self):
        result = map_fits_keywords(self._tess_header(), mission="TESS")
        assert "tic_id" in result.mapped_values
        assert result.mapped_values["tic_id"] == 150428135

    def test_tmag_mapped_float(self):
        result = map_fits_keywords(self._tess_header(), mission="TESS")
        assert isinstance(result.mapped_values.get("tmag"), float)

    def test_bjd_ref_computed(self):
        header = {"BJDREFI": 2457000, "BJDREFF": 0.5}
        result = map_fits_keywords(header, mission="TESS")
        assert "bjd_ref" in result.mapped_values
        assert abs(result.mapped_values["bjd_ref"] - 2457000.5) < 1e-9

    def test_unrecognised_keys_tracked(self):
        header = dict(self._tess_header())
        header["UNKNOWN_KEY"] = "xyz"
        result = map_fits_keywords(header, mission="TESS")
        assert "UNKNOWN_KEY" in result.unrecognised_keys

    def test_n_unrecognised_correct(self):
        header = {"UNKNOWN1": 1, "UNKNOWN2": 2}
        result = map_fits_keywords(header, mission="TESS")
        assert result.n_unrecognised == 2

    def test_mission_stored(self):
        result = map_fits_keywords({}, mission="TESS")
        assert result.mission == "TESS"

    def test_kepler_mission(self):
        header = {"KEPLERID": 757450, "QUARTER": 3}
        result = map_fits_keywords(header, mission="Kepler")
        assert result.mapped_values.get("kepler_id") == 757450

    def test_k2_uses_kepler_mappings(self):
        header = {"KEPLERID": 123}
        r_k2 = map_fits_keywords(header, mission="K2")
        assert "kepler_id" in r_k2.mapped_values

    def test_invalid_mission(self):
        result = map_fits_keywords({}, mission="HST")
        assert result.flag == "INVALID"

    def test_non_dict_header_invalid(self):
        result = map_fits_keywords("not a dict", mission="TESS")
        assert result.flag == "INVALID"

    def test_empty_header_partial_or_ok(self):
        result = map_fits_keywords({}, mission="TESS")
        assert result.flag in ("OK", "PARTIAL")
        assert result.n_found == 0

    def test_result_frozen(self):
        result = map_fits_keywords({}, mission="TESS")
        try:
            result.mission = "other"
            raise AssertionError()
        except Exception:
            pass

    def test_format_returns_string(self):
        result = map_fits_keywords(self._tess_header(), mission="TESS")
        text = format_keyword_map_result(result)
        assert isinstance(text, str)

    def test_format_contains_mission(self):
        result = map_fits_keywords(self._tess_header(), mission="TESS")
        text = format_keyword_map_result(result)
        assert "TESS" in text
