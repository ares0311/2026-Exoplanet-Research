"""Plan the first T1-1 training-data batch without downloading light curves.

This script is the next step after ``verify_dataset_sources.py`` passes. It
creates small, committed metadata artifacts that satisfy the dataset handoff
brief's source-snapshot and storage-estimate requirements before any bulk MAST
download is proposed.

It does not download FITS files. It queries source catalogs, samples MAST
Lightkurve search metadata, and estimates raw-storage needs from provider
``size`` metadata when present.

Public API
----------
SourceSnapshot(...)
SearchProduct(...)
BatchPlan(...)
build_batch_plan(...)
format_batch_plan(...)
write_batch_plan(...)
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from verify_dataset_sources import (
    REQUIRED_KOI_COLUMNS,
    REQUIRED_TOI_COLUMNS,
    verify_table_schema,
)

TAP_SYNC = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
EXOFOP_TOI_CSV = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
PSCOMPARS_REQUIRED_COLUMNS = frozenset({"pl_name", "hostname", "discoverymethod", "disc_year"})
MAX_WORKING_BYTES = 100 * 1024**3

TapFn = Callable[[str], str]
CsvFetchFn = Callable[[str], str]
LightkurveSearchFn = Callable[..., Any]


@dataclass(frozen=True)
class SourceSnapshot:
    """Small source snapshot for one verified upstream data source."""

    name: str
    access_method: str
    url: str
    table: str | None
    query: str | None
    row_count_at_download: int
    required_columns_verified: bool
    required_columns: tuple[str, ...]
    actual_columns: tuple[str, ...]


@dataclass(frozen=True)
class SearchProduct:
    """One Lightkurve/MAST search result used for storage estimation."""

    mission: str
    target: str
    product_filename: str
    data_uri: str
    author: str
    size_bytes: int | None


@dataclass(frozen=True)
class BatchPlan:
    """T1-1 source-snapshot and storage-estimate plan."""

    created_at_utc: str
    sample_size: int
    working_data_cap_bytes: int
    source_snapshots: tuple[SourceSnapshot, ...]
    koi_label_rows: int
    koi_unique_targets: int
    koi_positive_rows: int
    koi_negative_rows: int
    toi_label_rows: int
    toi_unique_targets: int
    exofop_rows: int
    kepler_sample_targets: int
    kepler_sample_products: int
    kepler_sample_total_bytes: int
    kepler_estimated_raw_bytes: int | None
    tess_sample_targets: int
    tess_sample_products: int
    tess_sample_total_bytes: int
    tess_estimated_raw_bytes: int | None
    download_manifest_rows: tuple[SearchProduct, ...]
    flag: str
    next_action: str


def _default_fetch(url: str) -> str:
    import ssl

    try:
        import certifi

        ctx: ssl.SSLContext | None = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = None
    with urllib.request.urlopen(url, timeout=120, context=ctx) as resp:  # noqa: S310
        return resp.read().decode("utf-8", errors="replace")


def _default_lightkurve_search(target: str, *, mission: str, **criteria: Any) -> Any:
    from lightkurve import search_lightcurve  # noqa: PLC0415

    return search_lightcurve(target, mission=mission, **criteria)


def _tap_url(query: str) -> str:
    params = urllib.parse.urlencode({"query": query, "format": "csv"})
    return f"{TAP_SYNC}?{params}"


def _csv_rows(raw: str) -> list[dict[str, str]]:
    clean_lines = [line for line in raw.splitlines() if not line.startswith("#")]
    if not clean_lines:
        return []
    return list(csv.DictReader(io.StringIO("\n".join(clean_lines))))


def _tap_rows(query: str, tap_fn: TapFn) -> list[dict[str, str]]:
    return _csv_rows(tap_fn(_tap_url(query)))


def _int_value(value: object, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        if isinstance(value, str) and value.strip().lower() in {"nan", "masked"}:
            return default
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return default


def _str_value(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    return "" if text.lower() in {"--", "masked", "nan"} else text


def _row_count(query: str, tap_fn: TapFn) -> int:
    rows = _tap_rows(query, tap_fn)
    if not rows:
        return 0
    first = rows[0]
    return _int_value(first.get("n_rows") or first.get("N_ROWS") or next(iter(first.values()), 0))


def _source_snapshot(
    *,
    name: str,
    table: str,
    required_columns: frozenset[str],
    tap_fn: TapFn,
    query: str | None = None,
) -> SourceSnapshot:
    schema = verify_table_schema(table, required_columns, tap_fn=tap_fn)
    count_query = f"select count(*) as n_rows from {table}"
    return SourceSnapshot(
        name=name,
        access_method="tap",
        url=TAP_SYNC,
        table=table,
        query=query or count_query,
        row_count_at_download=_row_count(count_query, tap_fn),
        required_columns_verified=schema.ok,
        required_columns=tuple(sorted(required_columns)),
        actual_columns=tuple(sorted(schema.available_columns)),
    )


def _koi_rows(tap_fn: TapFn) -> list[dict[str, str]]:
    return _tap_rows(
        "select kepid,kepoi_name,koi_disposition,koi_period,koi_time0bk,koi_duration "
        "from cumulative "
        "where (koi_disposition='CONFIRMED' or koi_disposition='FALSE POSITIVE') "
        "and koi_period > 0.5 and koi_period < 500 and koi_time0bk > 0",
        tap_fn,
    )


def _toi_rows(tap_fn: TapFn) -> list[dict[str, str]]:
    return _tap_rows(
        "select tid,toi,tfopwg_disp,pl_orbper,pl_tranmid,pl_trandurh "
        "from toi "
        "where pl_orbper > 0 and pl_tranmid > 0 and pl_trandurh > 0",
        tap_fn,
    )


def _lightkurve_products(
    *,
    mission: str,
    targets: Sequence[str],
    lightkurve_search_fn: LightkurveSearchFn,
    request_delay: float,
    search_criteria: dict[str, Any] | None = None,
) -> list[SearchProduct]:
    products: list[SearchProduct] = []
    criteria = search_criteria or {}
    start = time.monotonic()
    total = len(targets)
    for index, target in enumerate(targets, 1):
        result = lightkurve_search_fn(target, mission=mission, **criteria)
        table = getattr(result, "table", None)
        colnames = set(getattr(table, "colnames", ()))
        n_rows = len(result)
        if table is not None:
            for row in table:
                products.append(
                    SearchProduct(
                        mission=mission,
                        target=target,
                        product_filename=_str_value(row["productFilename"])
                        if "productFilename" in colnames
                        else "",
                        data_uri=_str_value(row["dataURI"]) if "dataURI" in colnames else "",
                        author=_str_value(row["author"]) if "author" in colnames else "",
                        size_bytes=_int_value(row["size"], default=-1)
                        if "size" in colnames and _int_value(row["size"], default=-1) >= 0
                        else None,
                    )
                )
        elapsed = time.monotonic() - start
        rate = index / elapsed if elapsed > 0 else 0.0
        remaining = (total - index) / rate if rate > 0 else 0.0
        print(
            f"  [{index}/{total}] {mission} {target} products={n_rows} "
            f"elapsed={elapsed:.0f}s ETA={remaining:.0f}s",
            flush=True,
        )
        if request_delay > 0 and index < total:
            time.sleep(request_delay)
    return products


def _unique_ids(rows: Sequence[dict[str, str]], key: str) -> list[int]:
    values = sorted({_int_value(row.get(key)) for row in rows if _int_value(row.get(key)) > 0})
    return values


def _estimate_total_bytes(
    *,
    sample_products: Sequence[SearchProduct],
    sample_targets: int,
    total_targets: int,
) -> int | None:
    known_sizes = [
        product.size_bytes for product in sample_products if product.size_bytes is not None
    ]
    if sample_targets <= 0 or not known_sizes:
        return None
    sample_bytes = sum(known_sizes)
    return int(math.ceil((sample_bytes / sample_targets) * total_targets))


def build_batch_plan(
    *,
    sample_size: int = 5,
    request_delay: float = 0.5,
    tap_fn: TapFn | None = None,
    exofop_fn: CsvFetchFn | None = None,
    lightkurve_search_fn: LightkurveSearchFn | None = None,
    created_at_utc: str | None = None,
    working_data_cap_bytes: int = MAX_WORKING_BYTES,
) -> BatchPlan:
    """Build a source snapshot and storage estimate without downloading FITS."""
    _tap = tap_fn or _default_fetch
    _exofop = exofop_fn or _default_fetch
    _search = lightkurve_search_fn or _default_lightkurve_search
    created_at_utc = created_at_utc or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    snapshots = (
        _source_snapshot(
            name="nasa_exoplanet_archive_cumulative",
            table="cumulative",
            required_columns=REQUIRED_KOI_COLUMNS,
            tap_fn=_tap,
            query=(
                "select kepid,kepoi_name,koi_disposition,koi_period,"
                "koi_time0bk,koi_duration from cumulative"
            ),
        ),
        _source_snapshot(
            name="nasa_exoplanet_archive_toi",
            table="toi",
            required_columns=REQUIRED_TOI_COLUMNS,
            tap_fn=_tap,
            query="select tid,toi,tfopwg_disp,pl_orbper,pl_tranmid,pl_trandurh from toi",
        ),
        _source_snapshot(
            name="nasa_exoplanet_archive_pscomppars",
            table="pscomppars",
            required_columns=PSCOMPARS_REQUIRED_COLUMNS,
            tap_fn=_tap,
        ),
    )
    if any(not snapshot.required_columns_verified for snapshot in snapshots):
        return BatchPlan(
            created_at_utc=created_at_utc,
            sample_size=sample_size,
            working_data_cap_bytes=working_data_cap_bytes,
            source_snapshots=snapshots,
            koi_label_rows=0,
            koi_unique_targets=0,
            koi_positive_rows=0,
            koi_negative_rows=0,
            toi_label_rows=0,
            toi_unique_targets=0,
            exofop_rows=0,
            kepler_sample_targets=0,
            kepler_sample_products=0,
            kepler_sample_total_bytes=0,
            kepler_estimated_raw_bytes=None,
            tess_sample_targets=0,
            tess_sample_products=0,
            tess_sample_total_bytes=0,
            tess_estimated_raw_bytes=None,
            download_manifest_rows=(),
            flag="SCHEMA_FAIL",
            next_action="Stop and inspect source_snapshots: at least one required schema failed.",
        )

    koi_rows = _koi_rows(_tap)
    toi_rows = _toi_rows(_tap)
    exofop_rows = _csv_rows(_exofop(EXOFOP_TOI_CSV))

    koi_ids = _unique_ids(koi_rows, "kepid")
    toi_ids = _unique_ids(toi_rows, "tid")
    sample_koi_targets = [f"KIC {kepid}" for kepid in koi_ids[:sample_size]]
    sample_toi_targets = [f"TIC {tid}" for tid in toi_ids[:sample_size]]

    print(
        f"Planning T1-1 batch: sample_size={sample_size} "
        f"Kepler targets={len(sample_koi_targets)} TESS targets={len(sample_toi_targets)}",
        flush=True,
    )
    kepler_products = _lightkurve_products(
        mission="Kepler",
        targets=sample_koi_targets,
        lightkurve_search_fn=_search,
        request_delay=request_delay,
        search_criteria={"author": "Kepler", "exptime": 1800},
    )
    tess_products = _lightkurve_products(
        mission="TESS",
        targets=sample_toi_targets,
        lightkurve_search_fn=_search,
        request_delay=request_delay,
    )

    koi_positive = sum(1 for row in koi_rows if row.get("koi_disposition") == "CONFIRMED")
    koi_negative = sum(1 for row in koi_rows if row.get("koi_disposition") == "FALSE POSITIVE")
    kepler_estimate = _estimate_total_bytes(
        sample_products=kepler_products,
        sample_targets=len(sample_koi_targets),
        total_targets=len(koi_ids),
    )
    tess_estimate = _estimate_total_bytes(
        sample_products=tess_products,
        sample_targets=len(sample_toi_targets),
        total_targets=len(toi_ids),
    )
    total_known_estimate = sum(
        estimate for estimate in (kepler_estimate, tess_estimate) if estimate is not None
    )
    flag = "OK" if total_known_estimate <= working_data_cap_bytes else "OVER_CAP"
    next_action = (
        "Proceed to manifest design and a bounded first Kepler-only batch."
        if flag == "OK"
        else "Reduce batch size or ask human approval before any download."
    )

    return BatchPlan(
        created_at_utc=created_at_utc,
        sample_size=sample_size,
        working_data_cap_bytes=working_data_cap_bytes,
        source_snapshots=snapshots,
        koi_label_rows=len(koi_rows),
        koi_unique_targets=len(koi_ids),
        koi_positive_rows=koi_positive,
        koi_negative_rows=koi_negative,
        toi_label_rows=len(toi_rows),
        toi_unique_targets=len(toi_ids),
        exofop_rows=len(exofop_rows),
        kepler_sample_targets=len(sample_koi_targets),
        kepler_sample_products=len(kepler_products),
        kepler_sample_total_bytes=sum(p.size_bytes or 0 for p in kepler_products),
        kepler_estimated_raw_bytes=kepler_estimate,
        tess_sample_targets=len(sample_toi_targets),
        tess_sample_products=len(tess_products),
        tess_sample_total_bytes=sum(p.size_bytes or 0 for p in tess_products),
        tess_estimated_raw_bytes=tess_estimate,
        download_manifest_rows=tuple(kepler_products + tess_products),
        flag=flag,
        next_action=next_action,
    )


def format_batch_plan(plan: BatchPlan) -> str:
    """Render a compact Markdown report for operator review."""
    def _gb(value: int | None) -> str:
        return "unknown" if value is None else f"{value / 1024**3:.2f} GB"

    lines = [
        "# T1-1 Training Data Batch Plan",
        "",
        f"**Flag**: {plan.flag}",
        f"**Created at UTC**: {plan.created_at_utc}",
        f"**Sample size**: {plan.sample_size}",
        f"**Working data cap**: {_gb(plan.working_data_cap_bytes)}",
        "",
        "## Source Snapshots",
        "| Source | Rows | Columns verified |",
        "| --- | ---: | --- |",
    ]
    for snapshot in plan.source_snapshots:
        lines.append(
            f"| {snapshot.name} | {snapshot.row_count_at_download} | "
            f"{'yes' if snapshot.required_columns_verified else 'no'} |"
        )
    lines += [
        "",
        "## Label Inventory",
        f"- KOI label rows: {plan.koi_label_rows}",
        f"- KOI unique targets: {plan.koi_unique_targets}",
        f"- KOI positives/negatives: {plan.koi_positive_rows} / {plan.koi_negative_rows}",
        f"- TOI ephemeris rows: {plan.toi_label_rows}",
        f"- TOI unique targets: {plan.toi_unique_targets}",
        f"- ExoFOP public TOI CSV rows: {plan.exofop_rows}",
        "",
        "## Storage Estimate From MAST Search Metadata",
        "- Kepler estimate filter: `mission=Kepler`, `author=Kepler`, `exptime=1800`.",
        "- TESS estimate filter: `mission=TESS`.",
        (
            f"- Kepler sample targets/products: {plan.kepler_sample_targets} / "
            f"{plan.kepler_sample_products}"
        ),
        f"- Kepler sample bytes: {_gb(plan.kepler_sample_total_bytes)}",
        f"- Kepler estimated raw bytes for all KOI targets: {_gb(plan.kepler_estimated_raw_bytes)}",
        f"- TESS sample targets/products: {plan.tess_sample_targets} / {plan.tess_sample_products}",
        f"- TESS sample bytes: {_gb(plan.tess_sample_total_bytes)}",
        f"- TESS estimated raw bytes for all TOI targets: {_gb(plan.tess_estimated_raw_bytes)}",
        "",
        "## Next Action",
        plan.next_action,
    ]
    return "\n".join(lines) + "\n"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_batch_plan(plan: BatchPlan, *, metadata_dir: Path, report_path: Path) -> None:
    """Write source snapshots, sample manifest, JSON plan, and Markdown report."""
    metadata_dir.mkdir(parents=True, exist_ok=True)
    source_payload = {
        "created_at_utc": plan.created_at_utc,
        "sources": [asdict(snapshot) for snapshot in plan.source_snapshots],
    }
    _write_json(metadata_dir / "source_snapshots.json", source_payload)
    _write_jsonl(
        metadata_dir / "download_manifest_sample.jsonl",
        [asdict(row) for row in plan.download_manifest_rows],
    )
    _write_json(
        metadata_dir / "t1_1_training_batch_plan.json",
        {
            **asdict(plan),
            "source_snapshots": [asdict(snapshot) for snapshot in plan.source_snapshots],
            "download_manifest_rows": [asdict(row) for row in plan.download_manifest_rows],
        },
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(format_batch_plan(plan), encoding="utf-8")


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build T1-1 source snapshots and storage estimates without downloading "
            "light-curve FITS files."
        )
    )
    parser.add_argument("--sample-size", type=int, default=5)
    parser.add_argument("--request-delay", type=float, default=0.5)
    parser.add_argument("--metadata-dir", type=Path, default=Path("metadata"))
    parser.add_argument("--report", type=Path, default=Path("reports/t1-1_training_batch_plan.md"))
    args = parser.parse_args(argv)

    if args.sample_size < 1:
        parser.error("--sample-size must be >= 1")
    if args.request_delay < 0:
        parser.error("--request-delay must be >= 0")

    plan = build_batch_plan(sample_size=args.sample_size, request_delay=args.request_delay)
    write_batch_plan(plan, metadata_dir=args.metadata_dir, report_path=args.report)
    print(format_batch_plan(plan))
    print(f"Saved metadata to {args.metadata_dir}", flush=True)
    print(f"Saved report to {args.report}", flush=True)
    return 0 if plan.flag == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
