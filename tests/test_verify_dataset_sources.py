"""Tests for Skills/verify_dataset_sources.py."""
from __future__ import annotations

import sys
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Skills"))

from verify_dataset_sources import (
    REQUIRED_KOI_COLUMNS,
    REQUIRED_TOI_COLUMNS,
    SourceSmokeTestResult,
    TableSchemaCheck,
    format_smoke_test_result,
    run_source_smoke_test,
    verify_table_schema,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_KOI_SCHEMA_CSV = "column_name\n" + "\n".join(sorted(REQUIRED_KOI_COLUMNS)) + "\nextra_col\n"
_TOI_SCHEMA_CSV = "column_name\n" + "\n".join(sorted(REQUIRED_TOI_COLUMNS)) + "\n"
_KOI_SCHEMA_MISSING_CSV = "column_name\nkepid\nkepoi_name\n"

_KOI_ROWS_CSV = (
    ",".join(sorted(REQUIRED_KOI_COLUMNS))
    + "\n"
    + ",".join(
        {
            "kepid": "10000001",
            "kepoi_name": "K00001.01",
            "koi_disposition": "CONFIRMED",
            "koi_period": "2.204735",
            "koi_time0bk": "134.452987",
            "koi_duration": "2.123",
        }[c]
        for c in sorted(REQUIRED_KOI_COLUMNS)
    )
    + "\n"
)

_TOI_ROWS_CSV = (
    ",".join(sorted(REQUIRED_TOI_COLUMNS))
    + "\n"
    + ",".join(
        {
            "tid": "150428135",
            "toi": "700.01",
            "tfopwg_disp": "CP",
            "pl_orbper": "9.977",
            "pl_tranmid": "2458326.0",
            "pl_trandurh": "1.34",
        }[c]
        for c in sorted(REQUIRED_TOI_COLUMNS)
    )
    + "\n"
)

_EXOFOP_CSV = "TIC ID,TOI,Disposition\n150428135,700.01,CP\n"


def _query_from_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    params = urllib.parse.parse_qs(parsed.query)
    return params.get("query", [""])[0]


def _make_tap(*, koi_schema: str, toi_schema: str, koi_rows: str, toi_rows: str):
    def _fn(url: str) -> str:
        query = _query_from_url(url).lower()
        if "tap_schema.columns" in query and "cumulative" in query:
            return koi_schema
        if "tap_schema.columns" in query and "'toi'" in query:
            return toi_schema
        if "from cumulative" in query:
            return koi_rows
        if "from toi" in query:
            return toi_rows
        raise AssertionError(f"unexpected TAP query: {query!r}")

    return _fn


def _good_tap_fn():
    return _make_tap(
        koi_schema=_KOI_SCHEMA_CSV,
        toi_schema=_TOI_SCHEMA_CSV,
        koi_rows=_KOI_ROWS_CSV,
        toi_rows=_TOI_ROWS_CSV,
    )


def _good_exofop_fn(url: str) -> str:
    return _EXOFOP_CSV


def _good_lightkurve_search_fn(target: str, *, mission: str):
    return ["result"]


def _raise(*_args, **_kwargs):
    raise ConnectionError("no network")


# ---------------------------------------------------------------------------
# verify_table_schema
# ---------------------------------------------------------------------------


class TestVerifyTableSchema:
    def test_ok_when_all_columns_present(self) -> None:
        result = verify_table_schema(
            "cumulative", REQUIRED_KOI_COLUMNS, tap_fn=lambda _url: _KOI_SCHEMA_CSV
        )
        assert isinstance(result, TableSchemaCheck)
        assert result.ok is True
        assert result.missing_columns == frozenset()

    def test_reports_missing_columns(self) -> None:
        result = verify_table_schema(
            "cumulative", REQUIRED_KOI_COLUMNS, tap_fn=lambda _url: _KOI_SCHEMA_MISSING_CSV
        )
        assert result.ok is False
        assert "koi_period" in result.missing_columns

    def test_never_substitutes_a_renamed_column(self) -> None:
        # A near-miss column name ("koi_periodd") must not satisfy "koi_period".
        near_miss_csv = (
            "column_name\nkepid\nkepoi_name\nkoi_disposition\n"
            "koi_periodd\nkoi_time0bk\nkoi_duration\n"
        )
        result = verify_table_schema(
            "cumulative", REQUIRED_KOI_COLUMNS, tap_fn=lambda _url: near_miss_csv
        )
        assert result.ok is False
        assert result.missing_columns == frozenset({"koi_period"})

    def test_network_failure_reports_error_not_ok(self) -> None:
        result = verify_table_schema("cumulative", REQUIRED_KOI_COLUMNS, tap_fn=_raise)
        assert result.ok is False
        assert result.error is not None
        assert result.missing_columns == REQUIRED_KOI_COLUMNS

    def test_empty_schema_response_is_not_ok(self) -> None:
        result = verify_table_schema("cumulative", REQUIRED_KOI_COLUMNS, tap_fn=lambda _url: "")
        assert result.ok is False


# ---------------------------------------------------------------------------
# run_source_smoke_test
# ---------------------------------------------------------------------------


class TestRunSourceSmokeTest:
    def test_full_success(self) -> None:
        result = run_source_smoke_test(
            tap_fn=_good_tap_fn(),
            exofop_fn=_good_exofop_fn,
            lightkurve_search_fn=_good_lightkurve_search_fn,
        )
        assert isinstance(result, SourceSmokeTestResult)
        assert result.ok is True
        assert result.failure_reason is None
        assert result.sample_kepid == 10000001
        assert result.sample_tid == 150428135
        assert result.kepler_lightcurves_found == 1
        assert result.tess_lightcurves_found == 1

    def test_koi_schema_failure_stops_early(self) -> None:
        tap = _make_tap(
            koi_schema=_KOI_SCHEMA_MISSING_CSV,
            toi_schema=_TOI_SCHEMA_CSV,
            koi_rows=_KOI_ROWS_CSV,
            toi_rows=_TOI_ROWS_CSV,
        )
        result = run_source_smoke_test(
            tap_fn=tap, exofop_fn=_good_exofop_fn, lightkurve_search_fn=_good_lightkurve_search_fn
        )
        assert result.ok is False
        assert "cumulative" in (result.failure_reason or "")
        assert result.koi_row_count == 0

    def test_toi_schema_failure_stops_early(self) -> None:
        tap = _make_tap(
            koi_schema=_KOI_SCHEMA_CSV,
            toi_schema="column_name\ntid\n",
            koi_rows=_KOI_ROWS_CSV,
            toi_rows=_TOI_ROWS_CSV,
        )
        result = run_source_smoke_test(
            tap_fn=tap, exofop_fn=_good_exofop_fn, lightkurve_search_fn=_good_lightkurve_search_fn
        )
        assert result.ok is False
        assert "toi" in (result.failure_reason or "")

    def test_empty_koi_rows_fails(self) -> None:
        tap = _make_tap(
            koi_schema=_KOI_SCHEMA_CSV,
            toi_schema=_TOI_SCHEMA_CSV,
            koi_rows="",
            toi_rows=_TOI_ROWS_CSV,
        )
        result = run_source_smoke_test(
            tap_fn=tap, exofop_fn=_good_exofop_fn, lightkurve_search_fn=_good_lightkurve_search_fn
        )
        assert result.ok is False
        assert "no rows" in (result.failure_reason or "")

    def test_exofop_failure_reported(self) -> None:
        result = run_source_smoke_test(
            tap_fn=_good_tap_fn(), exofop_fn=_raise, lightkurve_search_fn=_good_lightkurve_search_fn
        )
        assert result.ok is False
        assert "ExoFOP" in (result.failure_reason or "")
        assert result.koi_row_count == 1
        assert result.toi_row_count == 1

    def test_kepler_search_zero_results_fails(self) -> None:
        def _lk(target: str, *, mission: str):
            return [] if mission == "Kepler" else ["result"]

        result = run_source_smoke_test(
            tap_fn=_good_tap_fn(), exofop_fn=_good_exofop_fn, lightkurve_search_fn=_lk
        )
        assert result.ok is False
        assert result.kepler_lightcurves_found == 0
        assert "Kepler" in (result.failure_reason or "")

    def test_tess_search_zero_results_fails(self) -> None:
        def _lk(target: str, *, mission: str):
            return ["result"] if mission == "Kepler" else []

        result = run_source_smoke_test(
            tap_fn=_good_tap_fn(), exofop_fn=_good_exofop_fn, lightkurve_search_fn=_lk
        )
        assert result.ok is False
        assert result.kepler_lightcurves_found == 1
        assert result.tess_lightcurves_found == 0

    def test_lightkurve_exception_reported(self) -> None:
        result = run_source_smoke_test(
            tap_fn=_good_tap_fn(), exofop_fn=_good_exofop_fn, lightkurve_search_fn=_raise
        )
        assert result.ok is False
        assert "Lightkurve" in (result.failure_reason or "")

    def test_schema_checks_always_two_entries_on_success(self) -> None:
        result = run_source_smoke_test(
            tap_fn=_good_tap_fn(),
            exofop_fn=_good_exofop_fn,
            lightkurve_search_fn=_good_lightkurve_search_fn,
        )
        assert len(result.schema_checks) == 2
        assert {c.table_name for c in result.schema_checks} == {"cumulative", "toi"}


# ---------------------------------------------------------------------------
# format_smoke_test_result
# ---------------------------------------------------------------------------


class TestFormatSmokeTestResult:
    def test_pass_report_contains_pass(self) -> None:
        result = run_source_smoke_test(
            tap_fn=_good_tap_fn(),
            exofop_fn=_good_exofop_fn,
            lightkurve_search_fn=_good_lightkurve_search_fn,
        )
        report = format_smoke_test_result(result)
        assert "PASS" in report
        assert "150428135" in report

    def test_fail_report_contains_failure_reason(self) -> None:
        result = run_source_smoke_test(tap_fn=_raise, exofop_fn=_good_exofop_fn)
        report = format_smoke_test_result(result)
        assert "FAIL" in report
        assert result.failure_reason in report
