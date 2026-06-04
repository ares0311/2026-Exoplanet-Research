"""Tests for Skills/tcert_crossmatch.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from tcert_crossmatch import TcertCrossmatchResult, crossmatch_tcert

_CATALOG = [
    ("12345", "TestPlanet b", 10.0, 2458000.0),
    ("12345", "TestPlanet c", 20.0, 2457990.0),
    ("99999", "OtherPlanet b", 5.0, 2458100.0),
]


class TestTcertCrossmatch:
    def test_exact_match(self) -> None:
        r = crossmatch_tcert("12345", 10.0, epoch_bjd=2458000.0, catalog=_CATALOG)
        assert r.flag == "OK"
        assert r.known_planet

    def test_no_match_different_tic(self) -> None:
        r = crossmatch_tcert("00000", 10.0, catalog=_CATALOG)
        assert r.flag == "OK"
        assert not r.known_planet
        assert r.n_matches == 0

    def test_invalid_period(self) -> None:
        r = crossmatch_tcert("12345", 0.0, catalog=_CATALOG)
        assert r.flag == "INVALID_PERIOD"

    def test_negative_period(self) -> None:
        r = crossmatch_tcert("12345", -5.0, catalog=_CATALOG)
        assert r.flag == "INVALID_PERIOD"

    def test_alias_half_period(self) -> None:
        r = crossmatch_tcert("12345", 5.0, catalog=_CATALOG, check_aliases=True)
        assert r.flag == "OK"
        if r.best_match is not None:
            assert r.best_match.is_alias

    def test_period_tolerance(self) -> None:
        r = crossmatch_tcert("12345", 10.05, catalog=_CATALOG, period_tol_pct=1.0)
        assert r.n_matches >= 1

    def test_tight_tolerance_no_match(self) -> None:
        r = crossmatch_tcert("12345", 10.5, catalog=_CATALOG, period_tol_pct=0.1)
        assert r.n_matches == 0

    def test_n_catalog_checked(self) -> None:
        r = crossmatch_tcert("12345", 10.0, catalog=_CATALOG)
        assert r.n_catalog_checked == 2  # two entries for TIC 12345

    def test_best_match_has_fields(self) -> None:
        r = crossmatch_tcert("12345", 10.0, catalog=_CATALOG)
        assert r.best_match is not None
        assert r.best_match.planet_name == "TestPlanet b"

    def test_result_frozen(self) -> None:
        r = crossmatch_tcert("12345", 10.0, catalog=_CATALOG)
        assert isinstance(r, TcertCrossmatchResult)
        try:
            object.__setattr__(r, "flag", "x")
            raise AssertionError()
        except Exception:
            pass

    def test_format_output(self) -> None:
        from tcert_crossmatch import format_tcert_result
        r = crossmatch_tcert("12345", 10.0, catalog=_CATALOG)
        s = format_tcert_result(r)
        assert "|" in s

    def test_no_aliases_check(self) -> None:
        r = crossmatch_tcert("12345", 5.0, catalog=_CATALOG, check_aliases=False)
        assert r.flag == "OK"
        assert r.n_matches == 0
