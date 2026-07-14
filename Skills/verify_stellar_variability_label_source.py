"""Verify the pinned Catalina stellar-variability label source without downloading it.

The verifier checks the publication-backed VizieR table schema, row and class
counts, a tiny positive-evidence sample, and the compressed table's delivery
headers.  It never downloads the catalog payload, crossmatches mission targets,
or authorizes representation training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import ssl
import sys
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from Skills.run_report import RunReport, report_path_for, run_and_commit_report  # noqa: E402

DEFAULT_CONTRACT_PATH = REPO_ROOT / "metadata/stellar_variability_label_source_contract_v1.json"
DEFAULT_OUTPUT_PATH = (
    REPO_ROOT / "artifacts/manifests/stellar_variability_label_source_verification_v1.json"
)

TapFn = Callable[[str, str], dict[str, Any]]
HeadFn = Callable[[str], Mapping[str, str]]
ReportFn = Callable[..., bool]


def _ssl_context() -> ssl.SSLContext | None:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return None


def _default_tap(endpoint: str, query: str) -> dict[str, Any]:
    params = urllib.parse.urlencode(
        {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "json", "QUERY": query}
    )
    url = f"{endpoint}?{params}"
    with urllib.request.urlopen(url, timeout=60, context=_ssl_context()) as response:  # noqa: S310
        value = json.load(response)
    if not isinstance(value, dict):
        raise ValueError("VizieR TAP response must be a JSON object")
    return value


def _default_head(url: str) -> Mapping[str, str]:
    request = urllib.request.Request(url, method="HEAD")  # noqa: S310
    with urllib.request.urlopen(request, timeout=60, context=_ssl_context()) as response:  # noqa: S310
        return {key.lower(): value for key, value in response.headers.items()}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _require_positive_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def load_contract(path: Path) -> dict[str, Any]:
    """Load and validate the immutable expected-source contract."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("stellar-variability label contract schema_version must be 1")
    catalog = raw.get("catalog")
    if not isinstance(catalog, dict):
        raise ValueError("stellar-variability label contract requires catalog")
    for field in ("name", "paper_doi", "catalog_doi", "tap_endpoint", "table", "gzip_url"):
        _require_text(catalog.get(field), f"catalog.{field}")
    delivery = catalog.get("delivery")
    if not isinstance(delivery, dict):
        raise ValueError("catalog.delivery must be an object")
    _require_positive_int(
        delivery.get("content_length_bytes"), "catalog.delivery.content_length_bytes"
    )
    for field in ("etag", "last_modified", "content_type"):
        _require_text(delivery.get(field), f"catalog.delivery.{field}")
    row_count = _require_positive_int(catalog.get("row_count"), "catalog.row_count")
    columns = catalog.get("required_columns")
    if not isinstance(columns, dict) or not columns:
        raise ValueError("catalog.required_columns must be a non-empty object")
    for name, datatype in columns.items():
        _require_text(name, "catalog.required_columns key")
        _require_text(datatype, f"catalog.required_columns.{name}")
    classes = catalog.get("classes")
    counts = catalog.get("class_counts")
    if not isinstance(classes, dict) or not isinstance(counts, dict):
        raise ValueError("catalog classes and class_counts must be objects")
    expected_codes = {str(index) for index in range(1, 18)}
    if set(classes) != expected_codes or set(counts) != expected_codes:
        raise ValueError("catalog classes and class_counts must contain codes 1 through 17")
    for code in expected_codes:
        _require_text(classes[code], f"catalog.classes.{code}")
        _require_positive_int(counts[code], f"catalog.class_counts.{code}")
    if sum(int(value) for value in counts.values()) != row_count:
        raise ValueError("catalog.class_counts must sum to catalog.row_count")
    rejections = raw.get("rejected_alternatives")
    if not isinstance(rejections, list) or len(rejections) < 2:
        raise ValueError("contract must record at least two rejected alternatives")
    for index, rejection in enumerate(rejections):
        if not isinstance(rejection, dict):
            raise ValueError(f"rejected_alternatives[{index}] must be an object")
        _require_text(rejection.get("source"), f"rejected_alternatives[{index}].source")
        _require_text(rejection.get("reason"), f"rejected_alternatives[{index}].reason")
    return raw


def _tap_rows(payload: Mapping[str, Any], label: str) -> list[list[Any]]:
    metadata = payload.get("metadata")
    rows = payload.get("data")
    if not isinstance(metadata, list) or not isinstance(rows, list):
        raise ValueError(f"{label} TAP response is missing metadata or data")
    if not all(isinstance(row, list) for row in rows):
        raise ValueError(f"{label} TAP response contains a non-row value")
    return rows


def _unquote_identifier(value: Any) -> str:
    return str(value).strip('"')


def _verify_headers(catalog: Mapping[str, Any], headers: Mapping[str, str]) -> dict[str, Any]:
    normalized = {key.lower(): value for key, value in headers.items()}
    expected = catalog["delivery"]
    actual = {
        "content_length_bytes": int(normalized.get("content-length", "0")),
        "etag": normalized.get("etag", ""),
        "last_modified": normalized.get("last-modified", ""),
        "content_type": normalized.get("content-type", "").split(";", maxsplit=1)[0],
    }
    if actual != expected:
        raise ValueError(f"catalog delivery metadata mismatch: expected={expected} actual={actual}")
    return {**actual, "status": "verified"}


def _verify_schema(catalog: Mapping[str, Any], payload: Mapping[str, Any]) -> list[dict[str, str]]:
    rows = _tap_rows(payload, "schema")
    table = str(catalog["table"])
    actual = {
        _unquote_identifier(row[1]): str(row[2])
        for row in rows
        if len(row) >= 3 and _unquote_identifier(row[0]) == table
    }
    expected = catalog["required_columns"]
    missing_or_changed = {
        name: {"expected": datatype, "actual": actual.get(name)}
        for name, datatype in expected.items()
        if actual.get(name) != datatype
    }
    if missing_or_changed:
        raise ValueError(f"catalog schema mismatch: {missing_or_changed}")
    return [
        {"name": name, "datatype": datatype, "status": "verified"}
        for name, datatype in expected.items()
    ]


def _verify_count(catalog: Mapping[str, Any], payload: Mapping[str, Any]) -> int:
    rows = _tap_rows(payload, "row-count")
    actual = int(rows[0][0]) if len(rows) == 1 and len(rows[0]) == 1 else -1
    if actual != catalog["row_count"]:
        raise ValueError(
            f"catalog row-count mismatch: expected={catalog['row_count']} actual={actual}"
        )
    return actual


def _verify_class_counts(catalog: Mapping[str, Any], payload: Mapping[str, Any]) -> dict[str, int]:
    rows = _tap_rows(payload, "class-count")
    actual = {str(int(row[0])): int(row[1]) for row in rows if len(row) == 2}
    if actual != catalog["class_counts"]:
        raise ValueError(
            f"catalog class-count mismatch: expected={catalog['class_counts']} actual={actual}"
        )
    return actual


def _verify_sample(catalog: Mapping[str, Any], payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = _tap_rows(payload, "sample")
    if len(rows) != 3:
        raise ValueError(f"catalog sample must contain 3 rows, got {len(rows)}")
    valid_classes = set(catalog["classes"])
    result: list[dict[str, Any]] = []
    for row in rows:
        if len(row) != 6 or not row[0] or str(int(row[4])) not in valid_classes:
            raise ValueError(f"catalog sample row is malformed: {row}")
        ra = float(row[1])
        dec = float(row[2])
        period = float(row[3])
        if not (0.0 <= ra < 360.0 and -90.0 <= dec <= 90.0 and period > 0.0):
            raise ValueError(f"catalog sample row has invalid coordinates or period: {row}")
        result.append(
            {
                "source_id": str(row[0]),
                "ra_deg": ra,
                "dec_deg": dec,
                "period_days": period,
                "class_code": int(row[4]),
                "class_flag": row[5],
            }
        )
    return result


def _progress(index: int, total: int, label: str, started: float) -> None:
    elapsed = time.monotonic() - started
    rate = index / elapsed if elapsed else 0.0
    eta = (total - index) / rate if rate else 0.0
    print(f"  [{index}/{total}] {label} elapsed={elapsed:.1f}s ETA={eta:.1f}s", flush=True)


def _artifact_contract_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return path.name


def verify_source(
    contract_path: Path,
    output_path: Path,
    *,
    tap_fn: TapFn = _default_tap,
    head_fn: HeadFn = _default_head,
    report_fn: ReportFn = run_and_commit_report,
) -> dict[str, Any]:
    """Verify the source contract and write one metadata-only evidence artifact."""
    contract = load_contract(contract_path)
    catalog = contract["catalog"]
    endpoint = str(catalog["tap_endpoint"])
    table = str(catalog["table"])
    started_at = datetime.now(UTC)
    started = time.monotonic()
    total_steps = 5
    print(
        "Stellar-variability label source verification startup: "
        f"catalogs=1 metadata_requests={total_steps} full_catalog_downloads=0 "
        "training_authorized=false",
        flush=True,
    )

    delivery = _verify_headers(catalog, head_fn(str(catalog["gzip_url"])))
    _progress(1, total_steps, "verified compressed-table headers", started)
    schema_query = (
        "SELECT table_name,column_name,datatype,description FROM TAP_SCHEMA.columns "
        "WHERE table_name LIKE '%ApJS/213/9/table3%'"
    )
    schema = _verify_schema(catalog, tap_fn(endpoint, schema_query))
    _progress(2, total_steps, "verified VizieR schema", started)
    row_count = _verify_count(
        catalog, tap_fn(endpoint, f'SELECT COUNT(*) AS n FROM "{table}"')
    )
    _progress(3, total_steps, "verified row count", started)
    class_counts = _verify_class_counts(
        catalog,
        tap_fn(endpoint, f'SELECT Cl, COUNT(*) AS n FROM "{table}" GROUP BY Cl ORDER BY Cl'),
    )
    _progress(4, total_steps, "verified class distribution", started)
    sample = _verify_sample(
        catalog,
        tap_fn(
            endpoint,
            f'SELECT TOP 3 CRTS,RAJ2000,DEJ2000,Per,Cl,f_Cl FROM "{table}"',
        ),
    )
    _progress(5, total_steps, "verified labeled row sample", started)

    artifact: dict[str, Any] = {
        "schema_version": 1,
        "artifact_id": "stellar_variability_label_source_verification_v1",
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "contract_path": _artifact_contract_path(contract_path),
        "contract_sha256": _sha256(contract_path),
        "runtime_python": sys.version.split()[0],
        "status": "success",
        "verification_mode": "primary_metadata_and_three_row_sample",
        "metadata_requests": total_steps,
        "full_catalog_payload_bytes_downloaded": 0,
        "catalog": {
            "name": catalog["name"],
            "table": table,
            "row_count": row_count,
            "class_counts": class_counts,
            "required_columns": schema,
            "delivery": delivery,
            "sample_rows": sample,
            "ground_truth_basis": catalog["ground_truth_basis"],
        },
        "source_identity_authorized": True,
        "crossmatch_authorized": False,
        "training_authorized": False,
        "rejected_alternatives": contract["rejected_alternatives"],
        "limitations": contract["limitations"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    elapsed = time.monotonic() - started
    report = RunReport(
        script="verify_stellar_variability_label_source",
        status="success",
        started_at=started_at.isoformat(),
        completed_at=datetime.now(UTC).isoformat(),
        elapsed_seconds=elapsed,
        items_processed=total_steps,
        items_written=1,
        output_paths=(str(output_path),),
        notes="primary metadata plus three TAP rows; zero full-catalog payload bytes",
    )
    report_path = report_path_for("verify_stellar_variability_label_source")
    if not report_fn(report, report_path):
        print(f"WARNING: Run Report push failed for {report_path}", flush=True)
    print(
        "Stellar-variability label source verification COMPLETE: "
        f"rows={row_count} classes={len(class_counts)} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return artifact


def format_status(artifact: Mapping[str, Any]) -> str:
    """Format a durable verification artifact for operator inspection."""
    catalog = artifact.get("catalog", {})
    return (
        "Stellar-variability label source status: "
        f"status={artifact.get('status')} rows={catalog.get('row_count')} "
        f"classes={len(catalog.get('class_counts', {}))} "
        f"full_catalog_payload_bytes_downloaded="
        f"{artifact.get('full_catalog_payload_bytes_downloaded')} "
        f"training_authorized={artifact.get('training_authorized')}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--status-only", action="store_true")
    args = parser.parse_args(argv)
    if args.status_only:
        if not args.output.is_file():
            print(f"ERROR: verification artifact is missing: {args.output}", file=sys.stderr)
            return 1
        artifact = json.loads(args.output.read_text(encoding="utf-8"))
        print(format_status(artifact), flush=True)
        return 0
    try:
        artifact = verify_source(args.contract, args.output)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(
            f"ERROR: stellar-variability label source verification failed: {exc}",
            file=sys.stderr,
        )
        return 1
    print(format_status(artifact), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
