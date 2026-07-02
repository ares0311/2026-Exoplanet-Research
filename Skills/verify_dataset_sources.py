"""Verify T1-1 dataset source access before any bulk download or training run.

Implements the "Resource Access Contract" and "Minimum access smoke test"
from ``docs/exoplanet_exomoon_dataset_handoff.md``: query
``TAP_SCHEMA.columns`` for each required table before trusting any column
name, fetch a few rows to confirm each table is queryable and non-empty, and
confirm both Kepler and TESS light curves are discoverable via Lightkurve for
one real target pulled from the verified tables.

This must be run — and must pass — before writing or running any bulk
downloader against these sources. All network I/O is injectable for offline
tests; the default implementations use the exact TAP/CSV endpoints and
required-column lists specified in the dataset handoff doc.

Public API
----------
TableSchemaCheck(table_name, required_columns, available_columns, missing_columns, ok)
verify_table_schema(table_name, required_columns, *, tap_fn) -> TableSchemaCheck
SourceSmokeTestResult(schema_checks, koi_row_count, toi_row_count, exofop_row_count,
                       sample_kepid, sample_tid, kepler_lightcurves_found,
                       tess_lightcurves_found, ok, failure_reason)
run_source_smoke_test(*, tap_fn, exofop_fn, lightkurve_search_fn) -> SourceSmokeTestResult
format_smoke_test_result(result) -> str
"""
from __future__ import annotations

import csv
import io
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Constants — exact endpoints and required columns from the dataset handoff doc
# ---------------------------------------------------------------------------

_TAP_SYNC = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
_EXOFOP_TOI_CSV = (
    "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
)

REQUIRED_KOI_COLUMNS: frozenset[str] = frozenset(
    {"kepid", "kepoi_name", "koi_disposition", "koi_period", "koi_time0bk", "koi_duration"}
)
REQUIRED_TOI_COLUMNS: frozenset[str] = frozenset(
    {"tid", "toi", "tfopwg_disp", "pl_orbper", "pl_tranmid", "pl_trandurh"}
)

TapFn = Callable[[str], str]
ExofopFn = Callable[[str], str]
LightkurveSearchFn = Callable[..., Any]


def _default_tap_fn(url: str) -> str:
    import ssl

    try:
        import certifi

        ctx: ssl.SSLContext | None = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = None
    with urllib.request.urlopen(url, timeout=120, context=ctx) as resp:  # noqa: S310
        return resp.read().decode("utf-8", errors="replace")


def _default_exofop_fn(url: str) -> str:
    return _default_tap_fn(url)


def _default_lightkurve_search_fn(target: str, *, mission: str) -> Any:
    from lightkurve import search_lightcurve  # noqa: PLC0415

    return search_lightcurve(target, mission=mission)


def _tap_query_url(query: str) -> str:
    params = urllib.parse.urlencode({"query": query, "format": "csv"})
    return f"{_TAP_SYNC}?{params}"


def _parse_csv_rows(raw: str) -> list[dict[str, str]]:
    clean_lines = [ln for ln in raw.splitlines() if not ln.startswith("#")]
    if not clean_lines:
        return []
    reader = csv.DictReader(io.StringIO("\n".join(clean_lines)))
    return list(reader)


# ---------------------------------------------------------------------------
# Schema verification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TableSchemaCheck:
    """Result of verifying one TAP table's columns against a required set.

    Attributes:
        table_name: TAP table name (e.g. "cumulative", "toi").
        required_columns: Columns this project needs from the table.
        available_columns: Columns actually reported by TAP_SCHEMA.columns.
            Empty when the schema query itself failed.
        missing_columns: required_columns not present in available_columns.
        ok: True only when the schema query succeeded and nothing is missing.
        error: Non-None when the schema query failed (network/parse error).
    """

    table_name: str
    required_columns: frozenset[str]
    available_columns: frozenset[str]
    missing_columns: frozenset[str]
    ok: bool
    error: str | None = None


def verify_table_schema(
    table_name: str,
    required_columns: frozenset[str],
    *,
    tap_fn: TapFn | None = None,
) -> TableSchemaCheck:
    """Query TAP_SCHEMA.columns for *table_name* and check required columns.

    Never infers or substitutes renamed columns — a missing column is always
    reported, never silently worked around, per the dataset handoff doc's
    "no guessing" rule.
    """
    _fn = tap_fn if tap_fn is not None else _default_tap_fn
    query = f"select column_name from TAP_SCHEMA.columns where table_name = '{table_name}'"

    try:
        raw = _fn(_tap_query_url(query))
    except Exception as exc:  # noqa: BLE001
        return TableSchemaCheck(
            table_name=table_name,
            required_columns=required_columns,
            available_columns=frozenset(),
            missing_columns=required_columns,
            ok=False,
            error=str(exc),
        )

    rows = _parse_csv_rows(raw)
    available = frozenset(
        str(row.get("column_name", "")).strip() for row in rows if row.get("column_name")
    )
    missing = required_columns - available
    return TableSchemaCheck(
        table_name=table_name,
        required_columns=required_columns,
        available_columns=available,
        missing_columns=missing,
        ok=len(missing) == 0 and len(available) > 0,
    )


# ---------------------------------------------------------------------------
# Full resource-access smoke test
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceSmokeTestResult:
    """Result of the full T1-1 resource-access smoke test.

    Attributes:
        schema_checks: One TableSchemaCheck per required table, in order
            (cumulative, toi).
        koi_row_count: Rows returned by the sample `cumulative` query.
        toi_row_count: Rows returned by the sample `toi` query.
        exofop_row_count: Rows returned by the ExoFOP public TOI CSV.
        sample_kepid: First kepid pulled from the sample KOI rows, if any.
        sample_tid: First tid pulled from the sample TOI rows, if any.
        kepler_lightcurves_found: Number of Lightkurve search results for
            "KIC {sample_kepid}", mission="Kepler". -1 if not attempted.
        tess_lightcurves_found: Number of Lightkurve search results for
            "TIC {sample_tid}", mission="TESS". -1 if not attempted.
        ok: True only when every check above passed.
        failure_reason: Human-readable reason for the first failure, if any.
    """

    schema_checks: tuple[TableSchemaCheck, ...] = field(default_factory=tuple)
    koi_row_count: int = 0
    toi_row_count: int = 0
    exofop_row_count: int = 0
    sample_kepid: int | None = None
    sample_tid: int | None = None
    kepler_lightcurves_found: int = -1
    tess_lightcurves_found: int = -1
    ok: bool = False
    failure_reason: str | None = None


def run_source_smoke_test(
    *,
    tap_fn: TapFn | None = None,
    exofop_fn: ExofopFn | None = None,
    lightkurve_search_fn: LightkurveSearchFn | None = None,
) -> SourceSmokeTestResult:
    """Run the full T1-1 resource-access smoke test.

    Mirrors ``smoke_test_catalogs()`` + ``smoke_test_lightkurve()`` from
    ``docs/exoplanet_exomoon_dataset_handoff.md`` exactly: verify schemas,
    fetch a handful of rows from each catalog, then confirm Lightkurve can
    find light curves for one real target pulled from those rows.

    Every step fails closed: on the first failure, the function stops and
    returns ``ok=False`` with ``failure_reason`` set, rather than guessing
    or substituting a fallback value.
    """
    _tap = tap_fn if tap_fn is not None else _default_tap_fn
    _exofop = exofop_fn if exofop_fn is not None else _default_exofop_fn
    _lk_search = (
        lightkurve_search_fn if lightkurve_search_fn is not None else _default_lightkurve_search_fn
    )

    koi_check = verify_table_schema("cumulative", REQUIRED_KOI_COLUMNS, tap_fn=_tap)
    toi_check = verify_table_schema("toi", REQUIRED_TOI_COLUMNS, tap_fn=_tap)
    schema_checks = (koi_check, toi_check)

    if not koi_check.ok:
        reason = (
            koi_check.error
            or f"cumulative table missing required columns: {sorted(koi_check.missing_columns)}"
        )
        return SourceSmokeTestResult(schema_checks=schema_checks, ok=False, failure_reason=reason)
    if not toi_check.ok:
        reason = (
            toi_check.error
            or f"toi table missing required columns: {sorted(toi_check.missing_columns)}"
        )
        return SourceSmokeTestResult(schema_checks=schema_checks, ok=False, failure_reason=reason)

    koi_cols = ",".join(sorted(REQUIRED_KOI_COLUMNS))
    toi_cols = ",".join(sorted(REQUIRED_TOI_COLUMNS))

    try:
        koi_rows = _parse_csv_rows(
            _tap(_tap_query_url(f"select top 5 {koi_cols} from cumulative"))
        )
        toi_rows = _parse_csv_rows(_tap(_tap_query_url(f"select top 5 {toi_cols} from toi")))
    except Exception as exc:  # noqa: BLE001
        return SourceSmokeTestResult(
            schema_checks=schema_checks, ok=False, failure_reason=f"TAP row fetch failed: {exc}"
        )

    if not koi_rows:
        return SourceSmokeTestResult(
            schema_checks=schema_checks,
            ok=False,
            failure_reason="cumulative smoke-test query returned no rows",
        )
    if not toi_rows:
        return SourceSmokeTestResult(
            schema_checks=schema_checks,
            koi_row_count=len(koi_rows),
            ok=False,
            failure_reason="toi smoke-test query returned no rows",
        )

    try:
        exofop_raw = _exofop(_EXOFOP_TOI_CSV)
        exofop_rows = _parse_csv_rows(exofop_raw)
    except Exception as exc:  # noqa: BLE001
        return SourceSmokeTestResult(
            schema_checks=schema_checks,
            koi_row_count=len(koi_rows),
            toi_row_count=len(toi_rows),
            ok=False,
            failure_reason=f"ExoFOP public TOI CSV fetch failed: {exc}",
        )
    if not exofop_rows:
        return SourceSmokeTestResult(
            schema_checks=schema_checks,
            koi_row_count=len(koi_rows),
            toi_row_count=len(toi_rows),
            ok=False,
            failure_reason="ExoFOP public TOI CSV returned no rows",
        )

    try:
        sample_kepid = int(float(koi_rows[0]["kepid"]))
        sample_tid = int(float(toi_rows[0]["tid"]))
    except (KeyError, ValueError, TypeError) as exc:
        return SourceSmokeTestResult(
            schema_checks=schema_checks,
            koi_row_count=len(koi_rows),
            toi_row_count=len(toi_rows),
            exofop_row_count=len(exofop_rows),
            ok=False,
            failure_reason=f"could not parse sample kepid/tid from smoke-test rows: {exc}",
        )

    try:
        kepler_search = _lk_search(f"KIC {sample_kepid}", mission="Kepler")
        n_kepler = len(kepler_search)
    except Exception as exc:  # noqa: BLE001
        return SourceSmokeTestResult(
            schema_checks=schema_checks,
            koi_row_count=len(koi_rows),
            toi_row_count=len(toi_rows),
            exofop_row_count=len(exofop_rows),
            sample_kepid=sample_kepid,
            sample_tid=sample_tid,
            ok=False,
            failure_reason=f"Lightkurve Kepler search failed: {exc}",
        )
    if n_kepler == 0:
        return SourceSmokeTestResult(
            schema_checks=schema_checks,
            koi_row_count=len(koi_rows),
            toi_row_count=len(toi_rows),
            exofop_row_count=len(exofop_rows),
            sample_kepid=sample_kepid,
            sample_tid=sample_tid,
            kepler_lightcurves_found=0,
            ok=False,
            failure_reason=f"No Kepler light curves found for KIC {sample_kepid}",
        )

    try:
        tess_search = _lk_search(f"TIC {sample_tid}", mission="TESS")
        n_tess = len(tess_search)
    except Exception as exc:  # noqa: BLE001
        return SourceSmokeTestResult(
            schema_checks=schema_checks,
            koi_row_count=len(koi_rows),
            toi_row_count=len(toi_rows),
            exofop_row_count=len(exofop_rows),
            sample_kepid=sample_kepid,
            sample_tid=sample_tid,
            kepler_lightcurves_found=n_kepler,
            ok=False,
            failure_reason=f"Lightkurve TESS search failed: {exc}",
        )
    if n_tess == 0:
        return SourceSmokeTestResult(
            schema_checks=schema_checks,
            koi_row_count=len(koi_rows),
            toi_row_count=len(toi_rows),
            exofop_row_count=len(exofop_rows),
            sample_kepid=sample_kepid,
            sample_tid=sample_tid,
            kepler_lightcurves_found=n_kepler,
            tess_lightcurves_found=0,
            ok=False,
            failure_reason=f"No TESS light curves found for TIC {sample_tid}",
        )

    return SourceSmokeTestResult(
        schema_checks=schema_checks,
        koi_row_count=len(koi_rows),
        toi_row_count=len(toi_rows),
        exofop_row_count=len(exofop_rows),
        sample_kepid=sample_kepid,
        sample_tid=sample_tid,
        kepler_lightcurves_found=n_kepler,
        tess_lightcurves_found=n_tess,
        ok=True,
    )


def format_smoke_test_result(result: SourceSmokeTestResult) -> str:
    """Render a :class:`SourceSmokeTestResult` as a Markdown report."""
    lines = [
        "# T1-1 Dataset Source Access Smoke Test",
        "",
        f"**Overall**: {'PASS' if result.ok else 'FAIL'}",
    ]
    if result.failure_reason:
        lines.append(f"**Failure reason**: {result.failure_reason}")
    lines += [
        "",
        "## Schema checks",
        "| Table | OK | Missing columns | Error |",
        "| --- | --- | --- | --- |",
    ]
    for check in result.schema_checks:
        missing = ", ".join(sorted(check.missing_columns)) or "—"
        ok_str = "yes" if check.ok else "no"
        lines.append(f"| {check.table_name} | {ok_str} | {missing} | {check.error or '—'} |")
    lines += [
        "",
        "## Row/search checks",
        f"- KOI (`cumulative`) sample rows: {result.koi_row_count}",
        f"- TOI (`toi`) sample rows: {result.toi_row_count}",
        f"- ExoFOP public TOI CSV rows: {result.exofop_row_count}",
        f"- Sample kepid: {result.sample_kepid if result.sample_kepid is not None else '—'}",
        f"- Sample tid: {result.sample_tid if result.sample_tid is not None else '—'}",
        f"- Kepler light curves found for sample kepid: {result.kepler_lightcurves_found}",
        f"- TESS light curves found for sample tid: {result.tess_lightcurves_found}",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        prog="verify_dataset_sources",
        description=(
            "Run the T1-1 resource-access smoke test (NASA Exoplanet Archive TAP, "
            "ExoFOP public TOI CSV, Lightkurve Kepler/TESS search) before any bulk "
            "download or training run. Exits non-zero on any failure."
        ),
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Optional path to save the Markdown report"
    )
    args = parser.parse_args(argv)

    result = run_source_smoke_test()
    report = format_smoke_test_result(result)
    print(report)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report)
        print(f"Saved to {out}")

    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
