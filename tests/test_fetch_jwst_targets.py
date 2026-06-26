"""Tests for Skills/fetch_jwst_targets.py (Option A1: JWST target discovery)."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from fetch_jwst_targets import (
    JwstObservation,
    _enrich_with_products,
    _safe_float,
    format_summary,
    query_jwst_timeseries,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_raw(
    obsid: str = "111",
    target: str = "WASP-39",
    instrument: str = "NIRISS/SOSS",
    exptime: float = 7200.0,
    ra: float | None = 185.0,
    dec: float | None = -3.4,
) -> dict[str, Any]:
    return {
        "obsid": obsid,
        "target_name": target,
        "ra": ra,
        "dec": dec,
        "instrument": instrument,
        "program_id": "1366",
        "t_min": 59800.0,
        "t_max": 59802.0,
        "t_exptime": exptime,
        "filters": "CLEAR;GR700XD",
        "obsid_raw": obsid,
    }


def _product_fn_calints(obsids: list[str]) -> dict[str, list[str]]:
    return {oid: [f"jw01366001001_04101_00001_{oid}_calints.fits"] for oid in obsids}


def _product_fn_x1dints(obsids: list[str]) -> dict[str, list[str]]:
    return {oid: [f"jw01366001001_04101_00001_{oid}_x1dints.fits"] for oid in obsids}


def _product_fn_both(obsids: list[str]) -> dict[str, list[str]]:
    return {
        oid: [
            f"jw01366_{oid}_calints.fits",
            f"jw01366_{oid}_x1dints.fits",
        ]
        for oid in obsids
    }


def _product_fn_empty(obsids: list[str]) -> dict[str, list[str]]:
    return {oid: [] for oid in obsids}


def _mock_search(instrument: str | None, min_exptime: float) -> list[dict[str, Any]]:
    rows = [
        _make_raw("111", "WASP-39", "NIRISS/SOSS", 7200.0),
        _make_raw("222", "TRAPPIST-1", "NIRCAM/GRISM TIME SERIES", 3600.0),
        _make_raw("333", "short", "NIRISS/SOSS", 100.0),  # below min_exptime
    ]
    return [r for r in rows if r["t_exptime"] >= min_exptime]


# ---------------------------------------------------------------------------
# JwstObservation dataclass
# ---------------------------------------------------------------------------

def test_observation_fields() -> None:
    obs = JwstObservation(
        obsid="111", target_name="WASP-39", ra=185.0, dec=-3.4,
        instrument="NIRISS/SOSS", program_id="1366",
        t_min=59800.0, t_max=59802.0, t_exptime=7200.0,
        n_products=2, has_calints=True, has_x1dints=False,
        filters="CLEAR;GR700XD",
    )
    assert obs.obsid == "111"
    assert obs.has_calints is True
    assert obs.has_x1dints is False


# ---------------------------------------------------------------------------
# _safe_float
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("val,expected", [
    (3.14, 3.14),
    ("2.71", 2.71),
    (None, None),
    ("nan", None),
    (float("nan"), None),
    ("invalid", None),
])
def test_safe_float(val: Any, expected: float | None) -> None:
    assert _safe_float(val) == expected


# ---------------------------------------------------------------------------
# _enrich_with_products
# ---------------------------------------------------------------------------

def test_enrich_calints_flag() -> None:
    raw = [_make_raw("111")]
    obs = _enrich_with_products(raw, _product_fn_calints)
    assert len(obs) == 1
    assert obs[0].has_calints is True
    assert obs[0].has_x1dints is False


def test_enrich_x1dints_flag() -> None:
    raw = [_make_raw("222")]
    obs = _enrich_with_products(raw, _product_fn_x1dints)
    assert obs[0].has_x1dints is True
    assert obs[0].has_calints is False


def test_enrich_both_flags() -> None:
    raw = [_make_raw("333")]
    obs = _enrich_with_products(raw, _product_fn_both)
    assert obs[0].has_calints is True
    assert obs[0].has_x1dints is True


def test_enrich_no_products() -> None:
    raw = [_make_raw("444")]
    obs = _enrich_with_products(raw, _product_fn_empty)
    assert obs[0].n_products == 0
    assert obs[0].has_calints is False


def test_enrich_empty_input() -> None:
    assert _enrich_with_products([], _product_fn_calints) == []


def test_enrich_product_count() -> None:
    raw = [_make_raw("555")]
    obs = _enrich_with_products(raw, _product_fn_both)
    assert obs[0].n_products == 2


# ---------------------------------------------------------------------------
# query_jwst_timeseries
# ---------------------------------------------------------------------------

def test_query_returns_list() -> None:
    obs = query_jwst_timeseries(search_fn=_mock_search, product_fn=_product_fn_calints)
    assert isinstance(obs, list)


def test_query_min_exptime_filter() -> None:
    obs = query_jwst_timeseries(
        min_exptime=1800.0, search_fn=_mock_search, product_fn=_product_fn_calints
    )
    # The 100s target should be filtered out
    assert all((o.t_exptime or 0) >= 1800.0 for o in obs)


def test_query_sorted_by_t_min() -> None:
    obs = query_jwst_timeseries(search_fn=_mock_search, product_fn=_product_fn_calints)
    t_mins = [o.t_min or 0 for o in obs]
    assert t_mins == sorted(t_mins)


def test_query_instrument_preserved() -> None:
    obs = query_jwst_timeseries(search_fn=_mock_search, product_fn=_product_fn_calints)
    instruments = {o.instrument for o in obs}
    assert "NIRISS/SOSS" in instruments


def test_query_zero_results_on_empty_search() -> None:
    obs = query_jwst_timeseries(
        search_fn=lambda i, e: [], product_fn=_product_fn_calints
    )
    assert obs == []


# ---------------------------------------------------------------------------
# format_summary
# ---------------------------------------------------------------------------

def test_format_summary_empty() -> None:
    result = format_summary([])
    assert "_No JWST time-series" in result


def test_format_summary_contains_target() -> None:
    obs = query_jwst_timeseries(search_fn=_mock_search, product_fn=_product_fn_calints)
    table = format_summary(obs)
    assert "WASP-39" in table


def test_format_summary_header_row() -> None:
    obs = query_jwst_timeseries(search_fn=_mock_search, product_fn=_product_fn_calints)
    table = format_summary(obs)
    assert "| Target |" in table
    assert "calints" in table


def test_format_summary_checkmark_for_calints() -> None:
    raw = [_make_raw("111")]
    obs = _enrich_with_products(raw, _product_fn_calints)
    table = format_summary(obs)
    assert "✓" in table
