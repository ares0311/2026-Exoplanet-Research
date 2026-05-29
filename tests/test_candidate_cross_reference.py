"""Tests for Skills/candidate_cross_reference.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from candidate_cross_reference import (
    cross_reference,
    format_cross_ref_result,
)

_CATALOG = [
    {"tic_id": 100, "period_days": 5.0, "catalog_id": "TOI-1.01",
     "disposition": "CP"},
    {"tic_id": 100, "period_days": 5.01, "catalog_id": "TOI-1.02",
     "disposition": "PC"},
    {"tic_id": 200, "period_days": 10.0, "catalog_id": "TOI-2.01",
     "disposition": "FP"},
]


def test_known_tic_found():
    result = cross_reference(100, 5.0, _CATALOG)
    assert result.flag == "KNOWN"
    assert result.n_matches >= 1


def test_not_found():
    result = cross_reference(999, 5.0, _CATALOG)
    assert result.flag == "NOT_FOUND"
    assert result.n_matches == 0


def test_period_confirmed():
    result = cross_reference(100, 5.0, _CATALOG, period_rtol=0.01)
    assert result.period_confirmed


def test_period_conflict():
    result = cross_reference(100, 9.0, _CATALOG, period_rtol=0.01)
    assert result.flag == "PERIOD_CONFLICT"


def test_invalid_catalog():
    result = cross_reference(100, 5.0, "not-a-list")  # type: ignore[arg-type]
    assert result.flag == "INVALID"


def test_best_match_set():
    result = cross_reference(100, 5.0, _CATALOG)
    assert result.best_match is not None


def test_best_match_closest_period():
    result = cross_reference(100, 5.0, _CATALOG, period_rtol=0.02)
    assert result.best_match is not None
    assert result.best_match.period_delta_frac is not None
    assert result.best_match.period_delta_frac < 0.01


def test_period_none_gives_known():
    result = cross_reference(100, None, _CATALOG)
    assert result.flag == "KNOWN"
    assert result.period_confirmed is False


def test_format_returns_string():
    result = cross_reference(100, 5.0, _CATALOG)
    text = format_cross_ref_result(result)
    assert isinstance(text, str)
    assert "Cross-Reference" in text


def test_format_not_found():
    result = cross_reference(999, 5.0, _CATALOG)
    text = format_cross_ref_result(result)
    assert "NOT_FOUND" in text or "No catalog" in text


def test_tic_id_stored():
    result = cross_reference(100, 5.0, _CATALOG)
    assert result.tic_id == 100


def test_empty_catalog():
    result = cross_reference(100, 5.0, [])
    assert result.flag == "NOT_FOUND"
