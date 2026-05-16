"""Tests for Skills/fits_header_extractor.py."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Skills.fits_header_extractor import FITSStellarParams, extract_from_header  # noqa: E402


def _full_header() -> dict:
    return {
        "TICID": 150428135,
        "RADIUS": 0.42,
        "MASS": 0.40,
        "TEFF": 3480.0,
        "LOGG": 4.80,
        "CROWDSAP": 0.95,
        "SECTOR": 12,
    }


class TestExtractFromHeader:
    def test_all_keys_present_populates_all_fields(self) -> None:
        p = extract_from_header(_full_header())
        assert p.tic_id == 150428135
        assert p.stellar_radius_rsun == 0.42
        assert p.stellar_mass_msun == 0.40
        assert p.stellar_teff_k == 3480.0
        assert p.stellar_logg == 4.80
        assert p.sector == 12

    def test_missing_radius_gives_none(self) -> None:
        h = _full_header()
        del h["RADIUS"]
        p = extract_from_header(h)
        assert p.stellar_radius_rsun is None

    def test_missing_teff_gives_none(self) -> None:
        h = _full_header()
        del h["TEFF"]
        p = extract_from_header(h)
        assert p.stellar_teff_k is None

    def test_missing_crowdsap_gives_none(self) -> None:
        h = _full_header()
        del h["CROWDSAP"]
        p = extract_from_header(h)
        assert p.contamination_ratio is None

    def test_non_numeric_radius_gives_none(self) -> None:
        h = _full_header()
        h["RADIUS"] = "bad"
        p = extract_from_header(h)
        assert p.stellar_radius_rsun is None

    def test_to_vet_kwargs_excludes_none_fields(self) -> None:
        p = FITSStellarParams(
            tic_id=1, stellar_radius_rsun=None, stellar_mass_msun=1.0,
            stellar_teff_k=None, stellar_logg=None, contamination_ratio=0.1,
            sector=1,
        )
        kwargs = p.to_vet_kwargs()
        assert "stellar_radius_rsun" not in kwargs
        assert "stellar_mass_msun" in kwargs

    def test_to_vet_kwargs_keys_match_vet_params(self) -> None:
        p = extract_from_header(_full_header())
        kwargs = p.to_vet_kwargs()
        valid_keys = {
            "stellar_radius_rsun", "stellar_mass_msun", "stellar_teff_k", "contamination_ratio"
        }
        for k in kwargs:
            assert k in valid_keys

    def test_sector_extracted_as_int(self) -> None:
        p = extract_from_header(_full_header())
        assert isinstance(p.sector, int)
        assert p.sector == 12

    def test_ticid_extracted_as_int(self) -> None:
        p = extract_from_header(_full_header())
        assert isinstance(p.tic_id, int)

    def test_all_none_header_gives_all_none_dataclass(self) -> None:
        p = extract_from_header({
            "TICID": None, "RADIUS": None, "MASS": None,
            "TEFF": None, "LOGG": None, "CROWDSAP": None, "SECTOR": None,
        })
        assert p.stellar_radius_rsun is None
        assert p.stellar_mass_msun is None

    def test_empty_header_gives_all_none(self) -> None:
        p = extract_from_header({})
        assert p.tic_id is None
        assert p.stellar_radius_rsun is None

    def test_to_vet_kwargs_empty_when_all_none(self) -> None:
        p = extract_from_header({})
        assert p.to_vet_kwargs() == {}
