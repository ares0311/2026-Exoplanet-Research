"""Tests for Skills/fetch_jwst_targets.py (Option A1: JWST target discovery)."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
import fetch_jwst_targets
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


# ---------------------------------------------------------------------------
# Run Report (AGENTS.md Rule 7 retrofit)
# ---------------------------------------------------------------------------


def test_run_report_success_status(monkeypatch) -> None:
    captured = {}

    def fake_commit(report, path, **kwargs):
        captured["report"] = report
        captured["path"] = path
        captured["run_fn"] = kwargs.get("run_fn")
        return True

    monkeypatch.setattr(fetch_jwst_targets, "run_and_commit_report", fake_commit)

    fetch_jwst_targets._write_run_report(
        started_at="2026-07-17T00:00:00+00:00",
        elapsed_seconds=5.0,
        status="success",
        n_observations=2,
        output_path="data/jwst_timeseries_targets.json",
        notes="instrument=None min_exptime=1800.0",
        git_run_fn=object(),
    )
    report = captured["report"]
    assert report.script == "fetch_jwst_targets"
    assert report.status == "success"
    assert report.items_processed == 2
    assert report.items_written == 2
    assert report.items_failed == 0
    assert captured["path"].name == "fetch_jwst_targets.jsonl"
    assert captured["run_fn"] is not None


def test_run_report_failed_status_on_query_error(monkeypatch) -> None:
    captured = {}

    def fake_commit(report, path, **kwargs):
        captured["report"] = report
        return True

    monkeypatch.setattr(fetch_jwst_targets, "run_and_commit_report", fake_commit)

    fetch_jwst_targets._write_run_report(
        started_at="2026-07-17T00:00:00+00:00",
        elapsed_seconds=1.0,
        status="failed",
        n_observations=0,
        output_path=None,
        notes="error=connection refused",
        git_run_fn=None,
    )
    report = captured["report"]
    assert report.status == "failed"
    assert report.items_written == 0
    assert report.items_failed == 1
    assert "connection refused" in report.notes


def test_commit_failure_warns_but_does_not_raise(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        fetch_jwst_targets, "run_and_commit_report", lambda *a, **k: False
    )
    fetch_jwst_targets._write_run_report(
        started_at="2026-07-17T00:00:00+00:00",
        elapsed_seconds=1.0,
        status="success",
        n_observations=0,
        output_path=None,
        notes="",
        git_run_fn=None,
    )
    assert "Warning" in capsys.readouterr().out


def test_cli_writes_run_report_on_success(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        fetch_jwst_targets, "query_jwst_timeseries",
        lambda **kwargs: query_jwst_timeseries(
            search_fn=_mock_search, product_fn=_product_fn_calints
        ),
    )
    captured = {}

    def fake_commit(report, path, **kwargs):
        captured["report"] = report
        captured["run_fn"] = kwargs.get("run_fn")
        return True

    monkeypatch.setattr(fetch_jwst_targets, "run_and_commit_report", fake_commit)

    fake_runner = object()
    out = tmp_path / "jwst_targets.json"
    code = fetch_jwst_targets._cli(["--output", str(out)], git_run_fn=fake_runner)

    assert code == 0
    assert captured["run_fn"] is fake_runner
    assert captured["report"].status == "success"
    assert captured["report"].items_written > 0
    assert out.exists()


def test_cli_writes_run_report_on_query_failure(monkeypatch) -> None:
    def _raise_search(instrument, min_exptime):
        raise ConnectionError("no network")

    monkeypatch.setattr(
        fetch_jwst_targets, "query_jwst_timeseries",
        lambda **kwargs: query_jwst_timeseries(
            search_fn=_raise_search, product_fn=_product_fn_calints
        ),
    )
    captured = {}

    def fake_commit(report, path, **kwargs):
        captured["report"] = report
        return True

    monkeypatch.setattr(fetch_jwst_targets, "run_and_commit_report", fake_commit)

    code = fetch_jwst_targets._cli([])

    assert code == 1
    assert captured["report"].status == "failed"
    assert "no network" in captured["report"].notes


def test_default_search_raises_on_missing_column(monkeypatch):
    # Regression: astropy Table Row.get(key, default) silently returns the
    # default for a column that doesn't exist (no KeyError/AttributeError),
    # so a renamed/dropped "t_exptime" column previously made every row's
    # exptime read as 0, which the min_exptime filter then silently dropped
    # -- indistinguishable from "zero JWST time-series observations exist."
    from astropy.table import Table

    fake_table = Table({
        "obsid": ["1"],
        "target_name": ["WASP-1"],
        "s_ra": [10.0],
        "s_dec": [20.0],
        "instrument_name": ["NIRISS/SOSS"],
        "proposal_id": ["123"],
        "t_min": [100.0],
        "t_max": [101.0],
        "filters": ["CLEAR"],
        # t_exptime intentionally omitted.
    })
    monkeypatch.setattr(
        "astroquery.mast.Observations.query_criteria",
        lambda **kwargs: fake_table,
    )
    with pytest.raises(RuntimeError, match="missing expected column"):
        fetch_jwst_targets._default_search(None, 1800.0)


def test_default_search_returns_rows_when_all_columns_present(monkeypatch):
    from astropy.table import Table

    fake_table = Table({
        "obsid": ["1", "2"],
        "target_name": ["WASP-1", "WASP-2"],
        "s_ra": [10.0, 11.0],
        "s_dec": [20.0, 21.0],
        "instrument_name": ["NIRISS/SOSS", "NIRISS/SOSS"],
        "proposal_id": ["123", "124"],
        "t_min": [100.0, 200.0],
        "t_max": [101.0, 201.0],
        "t_exptime": [7200.0, 100.0],
        "filters": ["CLEAR", "CLEAR"],
    })
    monkeypatch.setattr(
        "astroquery.mast.Observations.query_criteria",
        lambda **kwargs: fake_table,
    )
    rows = fetch_jwst_targets._default_search(None, 1800.0)
    assert len(rows) == 1
    assert rows[0]["obsid"] == "1"
