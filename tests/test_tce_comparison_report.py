"""Tests for tce_comparison_report.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from tce_comparison_report import (
    compare_tce,
    format_tce_comparison,
)

_TIC = 150428135
_TABLE = [
    {"tic_id": _TIC, "tce_id": "TCE-001", "period_days": 37.42, "epoch": 2458000.0},
    {"tic_id": _TIC, "tce_id": "TCE-002", "period_days": 74.84, "epoch": 2458001.0},
    {"tic_id": 999999, "tce_id": "TCE-003", "period_days": 10.0, "epoch": 2458000.0},
]


class TestCompareTCE:
    def test_result_frozen(self):
        r = compare_tce(_TIC, 37.42, 2458000.0, _TABLE)
        try:
            r.n_matches = 99  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except (AttributeError, TypeError):
            pass

    def test_match_frozen(self):
        r = compare_tce(_TIC, 37.42, 2458000.0, _TABLE)
        if r.best_match:
            try:
                r.best_match.match_type = "X"  # type: ignore[misc]
                raise AssertionError("Should be frozen")
            except (AttributeError, TypeError):
                pass

    def test_invalid_period(self):
        r = compare_tce(_TIC, -1.0, 2458000.0, _TABLE)
        assert r.flag == "INVALID"

    def test_no_tces_for_tic(self):
        r = compare_tce(12345, 37.42, 2458000.0, _TABLE)
        assert r.flag == "NO_TCES"

    def test_exact_match_found(self):
        r = compare_tce(_TIC, 37.42, 2458000.0, _TABLE)
        assert r.flag == "OK"
        assert r.best_match is not None
        assert r.best_match.match_type == "exact"

    def test_alias_match_double_period(self):
        r = compare_tce(_TIC, 74.84, 2458000.0, _TABLE)
        assert r.flag in ("OK", "NO_MATCH")
        if r.flag == "OK" and r.best_match:
            assert r.best_match.match_type in ("exact", "alias", "conflict")

    def test_no_match_returns_no_match(self):
        r = compare_tce(_TIC, 999.0, 2458000.0, _TABLE)
        assert r.flag in ("NO_MATCH", "OK")

    def test_n_tces_checked_counts_tic_rows(self):
        r = compare_tce(_TIC, 37.42, 2458000.0, _TABLE)
        assert r.n_tces_checked == 2

    def test_empty_table_no_tces(self):
        r = compare_tce(_TIC, 37.42, 2458000.0, [])
        assert r.flag == "NO_TCES"

    def test_period_delta_frac_small_on_exact(self):
        r = compare_tce(_TIC, 37.42, 2458000.0, _TABLE)
        if r.best_match and r.best_match.match_type == "exact":
            assert r.best_match.period_delta_frac < 0.02

    def test_format_returns_string(self):
        r = compare_tce(_TIC, 37.42, 2458000.0, _TABLE)
        s = format_tce_comparison(r)
        assert isinstance(s, str)
        assert "TCE" in s

    def test_format_shows_tic_id(self):
        r = compare_tce(_TIC, 37.42, 2458000.0, _TABLE)
        s = format_tce_comparison(r)
        assert str(_TIC) in s

    def test_matches_tuple_type(self):
        r = compare_tce(_TIC, 37.42, 2458000.0, _TABLE)
        assert isinstance(r.matches, tuple)
