"""Tests for Skills.catalog_crossmatch."""
from __future__ import annotations

import pytest
from Skills.catalog_crossmatch import CatalogMatch, crossmatch, format_crossmatch


def _noop_sim(ra: float, dec: float) -> dict:
    return {}

def _noop_gaia(ra: float, dec: float) -> dict:
    return {}

def _noop_nea(tic_id: int) -> dict:
    return {}


class TestCrossmatch:
    def test_returns_catalog_match(self) -> None:
        m = crossmatch(1, 10.0, 20.0,
                       simbad_fn=_noop_sim, gaia_fn=_noop_gaia, nea_fn=_noop_nea)
        assert isinstance(m, CatalogMatch)

    def test_tic_id_stored(self) -> None:
        m = crossmatch(99999, 10.0, 20.0,
                       simbad_fn=_noop_sim, gaia_fn=_noop_gaia, nea_fn=_noop_nea)
        assert m.tic_id == 99999

    def test_ra_dec_stored(self) -> None:
        m = crossmatch(1, 123.45, -67.89,
                       simbad_fn=_noop_sim, gaia_fn=_noop_gaia, nea_fn=_noop_nea)
        assert m.ra == pytest.approx(123.45)
        assert m.dec == pytest.approx(-67.89)

    def test_all_none_when_no_hits(self) -> None:
        m = crossmatch(1, 0.0, 0.0,
                       simbad_fn=_noop_sim, gaia_fn=_noop_gaia, nea_fn=_noop_nea)
        assert m.simbad_name is None
        assert m.gaia_source_id is None
        assert m.nea_planet_name is None

    def test_simbad_name_populated(self) -> None:
        m = crossmatch(1, 0.0, 0.0,
                       simbad_fn=lambda *a: {"main_id": "HD 123"},
                       gaia_fn=_noop_gaia, nea_fn=_noop_nea)
        assert m.simbad_name == "HD 123"

    def test_gaia_source_id_populated(self) -> None:
        m = crossmatch(1, 0.0, 0.0,
                       simbad_fn=_noop_sim,
                       gaia_fn=lambda *a: {"source_id": "1234567890"},
                       nea_fn=_noop_nea)
        assert m.gaia_source_id == "1234567890"

    def test_gaia_distance_populated(self) -> None:
        m = crossmatch(1, 0.0, 0.0,
                       simbad_fn=_noop_sim,
                       gaia_fn=lambda *a: {"source_id": "X", "distance_pc": 42.0},
                       nea_fn=_noop_nea)
        assert m.distance_pc == pytest.approx(42.0)

    def test_nea_planet_name_populated(self) -> None:
        m = crossmatch(1, 0.0, 0.0,
                       simbad_fn=_noop_sim, gaia_fn=_noop_gaia,
                       nea_fn=lambda t: {"planet_name": "TOI-700 d"})
        assert m.nea_planet_name == "TOI-700 d"

    def test_found_in_empty_when_no_hits(self) -> None:
        m = crossmatch(1, 0.0, 0.0,
                       simbad_fn=_noop_sim, gaia_fn=_noop_gaia, nea_fn=_noop_nea)
        assert m.found_in == ()

    def test_found_in_contains_simbad(self) -> None:
        m = crossmatch(1, 0.0, 0.0,
                       simbad_fn=lambda *a: {"main_id": "HD 1"},
                       gaia_fn=_noop_gaia, nea_fn=_noop_nea)
        assert "simbad" in m.found_in

    def test_twomass_extracted_from_simbad(self) -> None:
        m = crossmatch(1, 0.0, 0.0,
                       simbad_fn=lambda *a: {"main_id": "2MASS J12345+6789"},
                       gaia_fn=_noop_gaia, nea_fn=_noop_nea)
        assert m.twomass_id is not None
        assert "2MASS" in m.twomass_id

    def test_exception_in_catalog_does_not_crash(self) -> None:
        def _bad(*a: object) -> dict:
            raise RuntimeError("network error")
        m = crossmatch(1, 0.0, 0.0,
                       simbad_fn=_bad, gaia_fn=_bad, nea_fn=lambda t: _bad())
        assert isinstance(m, CatalogMatch)


class TestFormatCrossmatch:
    def test_format_contains_tic_id(self) -> None:
        m = crossmatch(42, 0.0, 0.0,
                       simbad_fn=_noop_sim, gaia_fn=_noop_gaia, nea_fn=_noop_nea)
        text = format_crossmatch(m)
        assert "42" in text
