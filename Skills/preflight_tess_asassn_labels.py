"""Reproduce exact-TIC overlap between the frozen TESS inventory and ASAS-SN.

Normal mode owns one modulo shard and writes one row for every selected TIC.
Aggregate mode reconciles all six disjoint outputs.  Both modes remain
metadata-only and keep training unauthorized.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import ssl
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from Skills.run_report import RunReport, report_path_for, run_and_commit_report  # noqa: E402

DEFAULT_CONTRACT_PATH = REPO_ROOT / "metadata/asassn_variability_label_source_contract_v1.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "artifacts/manifests/tess_asassn_preflight_v1.jsonl"
DEFAULT_SUMMARY_PATH = REPO_ROOT / "artifacts/manifests/tess_asassn_preflight_summary_v1.json"
DEFAULT_AGGREGATE_PATH = REPO_ROOT / "artifacts/manifests/tess_asassn_preflight_aggregate_v1.json"
_TIC_PATTERN = re.compile(r"TIC\s+(\d+)")

TapFn = Callable[[str, str], dict[str, Any]]
HeadFn = Callable[[str], Mapping[str, str]]
ReportFn = Callable[..., bool]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


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
    with urllib.request.urlopen(  # noqa: S310
        f"{endpoint}?{params}", timeout=60, context=_ssl_context()
    ) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise ValueError("VizieR TAP response must be a JSON object")
    return value


def _default_head(url: str) -> Mapping[str, str]:
    request = urllib.request.Request(url, method="HEAD")  # noqa: S310
    with urllib.request.urlopen(  # noqa: S310
        request, timeout=60, context=_ssl_context()
    ) as response:
        return {key.lower(): value for key, value in response.headers.items()}


def _require_positive_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def load_contract(path: Path) -> dict[str, Any]:
    """Load the immutable source and overlap contract fail-closed."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("ASAS-SN contract schema_version must be 1")
    for section in ("catalog", "inventory", "query", "parallel_shape", "evidence_gate"):
        if not isinstance(raw.get(section), dict):
            raise ValueError(f"ASAS-SN contract requires {section}")
    catalog = raw["catalog"]
    for field in ("tap_endpoint", "table", "gzip_url"):
        if not isinstance(catalog.get(field), str) or not catalog[field]:
            raise ValueError(f"catalog.{field} must be non-empty")
    _require_positive_int(catalog.get("row_count"), "catalog.row_count")
    _require_positive_int(catalog.get("tic_row_count"), "catalog.tic_row_count")
    _require_positive_int(raw["inventory"].get("target_count"), "inventory.target_count")
    max_batch = _require_positive_int(raw["query"].get("max_batch_size"), "query.max_batch_size")
    if _require_positive_int(raw["query"].get("batch_size"), "query.batch_size") > max_batch:
        raise ValueError("query.batch_size exceeds max_batch_size")
    if raw.get("training_authorized") is not False:
        raise ValueError("ASAS-SN preflight must keep training_authorized=false")
    if raw["evidence_gate"].get("training_authorized_on_pass") is not False:
        raise ValueError("ASAS-SN evidence pass must not authorize training")
    if not isinstance(raw.get("source_verification_required"), bool):
        raise ValueError("source_verification_required must be boolean")
    return raw


def validate_inventory(contract: Mapping[str, Any]) -> Path:
    path = REPO_ROOT / str(contract["inventory"]["path"])
    if not path.is_file():
        raise ValueError(f"inventory path is missing: {path}")
    actual = _sha256(path)
    expected = str(contract["inventory"]["sha256"])
    if actual != expected:
        raise ValueError(f"inventory SHA-256 mismatch: expected={expected} actual={actual}")
    return path


def load_inventory_tics(path: Path, expected_count: int) -> tuple[int, ...]:
    values: set[int] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            match = _TIC_PATTERN.fullmatch(str(row.get("target_id", "")))
            if match is None:
                raise ValueError(f"{path}:{line_number}: invalid target_id")
            values.add(int(match.group(1)))
    if len(values) != expected_count:
        raise ValueError(
            f"inventory TIC count mismatch: expected={expected_count} actual={len(values)}"
        )
    return tuple(sorted(values))


def select_shard(tic_ids: Sequence[int], shard_index: int, shard_count: int) -> tuple[int, ...]:
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("shard index/count are invalid")
    return tuple(tic_id for tic_id in tic_ids if tic_id % shard_count == shard_index)


def _shard_path(path: Path, shard_index: int, shard_count: int) -> Path:
    if shard_count == 1:
        return path
    return path.with_name(
        f"{path.stem}.shard{shard_index + 1}of{shard_count}{path.suffix}"
    )


def build_query(contract: Mapping[str, Any], tic_ids: Sequence[int]) -> str:
    if not tic_ids:
        raise ValueError("cannot build an empty TIC query")
    selected = ",".join(f'"{column}"' for column in contract["query"]["selected_columns"])
    values = ",".join(repr(f"TIC {int(tic_id)}") for tic_id in tic_ids)
    return f'SELECT {selected} FROM "{contract["catalog"]["table"]}" WHERE "TIC" IN ({values})'


def _response_rows(response: Mapping[str, Any]) -> tuple[list[str], list[list[Any]]]:
    metadata = response.get("metadata")
    data = response.get("data")
    if not isinstance(metadata, list) or not isinstance(data, list):
        raise ValueError("VizieR response requires metadata and data arrays")
    names = [str(item["name"]) for item in metadata]
    if not all(isinstance(row, list) for row in data):
        raise ValueError("VizieR data rows must be arrays")
    return names, data


def verify_source_metadata(
    contract: Mapping[str, Any],
    *,
    tap_fn: TapFn = _default_tap,
    head_fn: HeadFn = _default_head,
) -> dict[str, Any]:
    """Verify delivery, schema, counts, and label distributions without payload bytes."""
    catalog = contract["catalog"]
    endpoint = str(catalog["tap_endpoint"])
    table = str(catalog["table"])
    headers = {key.lower(): value for key, value in head_fn(str(catalog["gzip_url"])).items()}
    header_map = {
        "content_length_bytes": "content-length",
        "etag": "etag",
        "last_modified": "last-modified",
        "content_type": "content-type",
    }
    delivery: dict[str, Any] = {}
    for contract_key, header_key in header_map.items():
        actual: Any = headers.get(header_key)
        if contract_key == "content_length_bytes" and actual is not None:
            actual = int(actual)
        expected = catalog["delivery"][contract_key]
        if actual != expected:
            raise ValueError(
                f"ASAS-SN delivery {header_key} changed: expected={expected!r} actual={actual!r}"
            )
        delivery[contract_key] = actual

    schema_query = (
        "SELECT table_name,column_name,datatype,description FROM TAP_SCHEMA.columns "
        "WHERE table_name LIKE '%MNRAS/519/5271%'"
    )
    _, schema_rows = _response_rows(tap_fn(endpoint, schema_query))
    actual_schema = {str(row[1]).strip('"'): str(row[2]) for row in schema_rows}
    expected_schema = dict(catalog["required_columns"])
    if any(actual_schema.get(name) != datatype for name, datatype in expected_schema.items()):
        raise ValueError("ASAS-SN required VizieR schema changed")

    def scalar(query: str) -> int:
        _, rows = _response_rows(tap_fn(endpoint, query))
        if len(rows) != 1 or len(rows[0]) != 1:
            raise ValueError("VizieR scalar query returned an unexpected shape")
        return int(rows[0][0])

    row_count = scalar(f'SELECT COUNT(*) AS n FROM "{table}"')
    tic_row_count = scalar(
        f'SELECT COUNT(*) AS n FROM "{table}" WHERE "TIC" LIKE \'TIC %\''
    )
    if row_count != int(catalog["row_count"]) or tic_row_count != int(catalog["tic_row_count"]):
        raise ValueError("ASAS-SN row or TIC count changed")

    _, class_rows = _response_rows(
        tap_fn(
            endpoint,
            f'SELECT "Class",COUNT(*) AS n FROM "{table}" GROUP BY "Class" ORDER BY "Class"',
        )
    )
    class_counts = {str(row[0]).strip(): int(row[1]) for row in class_rows}
    if class_counts != catalog["class_counts"]:
        raise ValueError("ASAS-SN class distribution changed")
    _, discovery_rows = _response_rows(
        tap_fn(
            endpoint,
            f'SELECT "Discovery",COUNT(*) AS n FROM "{table}" '
            'GROUP BY "Discovery" ORDER BY "Discovery"',
        )
    )
    discovery_counts = {str(row[0]).strip(): int(row[1]) for row in discovery_rows}
    if discovery_counts != catalog["discovery_counts"]:
        raise ValueError("ASAS-SN discovery distribution changed")
    return {
        "status": "pass",
        "metadata_requests": 6,
        "full_catalog_payload_bytes_downloaded": 0,
        "delivery": delivery,
        "required_columns": expected_schema,
        "row_count": row_count,
        "tic_row_count": tic_row_count,
        "class_counts": class_counts,
        "discovery_counts": discovery_counts,
    }


def _query_batch(
    contract: Mapping[str, Any],
    tic_ids: tuple[int, ...],
    tap_fn: TapFn,
) -> tuple[dict[str, Any], ...]:
    attempts = int(contract["query"]["retry_attempts"])
    endpoint = str(contract["catalog"]["tap_endpoint"])
    query = build_query(contract, tic_ids)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = tap_fn(endpoint, query)
            names, data = _response_rows(response)
            required = list(contract["query"]["selected_columns"])
            if names != required:
                raise ValueError(f"VizieR response columns changed: {names!r}")
            allowed = set(tic_ids)
            parsed: list[dict[str, Any]] = []
            for values in data:
                row = dict(zip(names, values, strict=True))
                match = _TIC_PATTERN.fullmatch(str(row["TIC"]).strip())
                if match is None:
                    raise ValueError(f"invalid VizieR TIC value: {row['TIC']!r}")
                tic_id = int(match.group(1))
                if tic_id not in allowed:
                    raise ValueError(f"VizieR returned unrequested TIC {tic_id}")
                probability = float(row["Prob"])
                if not 0.0 <= probability <= 1.0:
                    raise ValueError(f"invalid ASAS-SN probability for TIC {tic_id}")
                parsed.append(
                    {
                        "tic_id": tic_id,
                        "asassn_id": str(row["ASAS-SN"]).strip(),
                        "class_code": str(row["Class"]).strip(),
                        "probability": probability,
                        "discovery": str(row["Discovery"]).strip() == "True",
                        "ra_deg": float(row["RAJ2000"]),
                        "dec_deg": float(row["DEJ2000"]),
                    }
                )
            ids = [int(row["tic_id"]) for row in parsed]
            if len(ids) != len(set(ids)):
                raise ValueError("VizieR returned duplicate rows for one TIC")
            return tuple(parsed)
        except Exception as exc:  # network and fail-closed validation share retry policy
            last_error = exc
            if attempt < attempts:
                time.sleep(0.5 * attempt)
    raise RuntimeError(f"VizieR batch failed after {attempts} attempts: {last_error}")


def _format_eta(seconds: float) -> str:
    if seconds == float("inf"):
        return "unknown"
    return f"{seconds / 60:.0f}m{seconds % 60:.0f}s" if seconds > 90 else f"{seconds:.0f}s"


def _summary_from_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    matched = [row for row in rows if row["status"] == "matched"]
    probabilities = sorted(float(row["probability"]) for row in matched)
    return {
        "selected_tics": len(rows),
        "matched_rows": len(matched),
        "matched_unique_tics": len({int(row["tic_id"]) for row in matched}),
        "known_variables": sum(not bool(row["discovery"]) for row in matched),
        "discoveries": sum(bool(row["discovery"]) for row in matched),
        "class_counts": dict(sorted(Counter(str(row["class_code"]) for row in matched).items())),
        "minimum_probability": probabilities[0] if probabilities else None,
        "median_probability": probabilities[len(probabilities) // 2] if probabilities else None,
    }


def run_shard(
    contract_path: Path,
    output_path: Path,
    summary_path: Path,
    *,
    workers: int,
    batch_size: int,
    shard_index: int,
    shard_count: int,
    tap_fn: TapFn = _default_tap,
    head_fn: HeadFn = _default_head,
    report_fn: ReportFn = run_and_commit_report,
) -> dict[str, Any]:
    contract = load_contract(contract_path)
    if contract["source_verification_required"]:
        required_shards = int(contract["parallel_shape"]["process_shards"])
        required_workers = int(contract["parallel_shape"]["workers_per_shard"])
        if shard_count != required_shards or workers != required_workers:
            raise ValueError(
                f"merged ASAS-SN gate requires {required_shards} shards x "
                f"{required_workers} workers"
            )
    if workers <= 0:
        raise ValueError("workers must be positive")
    if not 0 < batch_size <= int(contract["query"]["max_batch_size"]):
        raise ValueError("batch size is outside the contract bound")
    inventory_path = validate_inventory(contract)
    all_tics = load_inventory_tics(inventory_path, int(contract["inventory"]["target_count"]))
    selected = select_shard(all_tics, shard_index, shard_count)
    if not selected:
        raise ValueError(f"shard {shard_index}/{shard_count} has no TICs")
    batches = [
        tuple(selected[index : index + batch_size])
        for index in range(0, len(selected), batch_size)
    ]
    started_at = datetime.now(UTC)
    started = time.monotonic()
    print(
        "TESS-ASAS-SN preflight startup: "
        f"selected_tics={len(selected)} batches={len(batches)} workers={workers} "
        f"shard={shard_index}/{shard_count} catalog_payload_bytes=0 "
        "training_authorized=false",
        flush=True,
    )
    source_verification: dict[str, Any]
    if shard_index == 0 and contract["source_verification_required"]:
        source_verification = verify_source_metadata(
            contract, tap_fn=tap_fn, head_fn=head_fn
        )
        print("  source metadata verification PASS (6/6 requests)", flush=True)
    else:
        source_verification = {"status": "not_repeated_on_this_shard"}
    matches: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(_query_batch, contract, batch, tap_fn): batch for batch in batches}
        for completed, future in enumerate(as_completed(pending), 1):
            for row in future.result():
                tic_id = int(row["tic_id"])
                if tic_id in matches:
                    raise ValueError(f"duplicate matched TIC across batches: {tic_id}")
                matches[tic_id] = row
            elapsed = time.monotonic() - started
            rate = completed / elapsed if elapsed else 0.0
            remaining = (len(batches) - completed) / rate if rate else float("inf")
            print(
                f"  [{completed}/{len(batches)}] batches elapsed={elapsed:.1f}s "
                f"ETA={_format_eta(remaining)} matches={len(matches)}",
                flush=True,
            )
    rows: list[dict[str, Any]] = []
    for tic_id in selected:
        match = matches.get(tic_id)
        rows.append(
            {
                "tic_id": tic_id,
                "status": "matched" if match else "no_exact_tic_match",
                **(match or {}),
                "training_authorized": False,
            }
        )
    output_path = _shard_path(output_path, shard_index, shard_count)
    summary_path = _shard_path(summary_path, shard_index, shard_count)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    elapsed = time.monotonic() - started
    summary = {
        "schema_version": 1,
        "artifact_id": "tess_asassn_preflight_shard_v1",
        "status": "success",
        "contract_path": _display_path(contract_path),
        "contract_sha256": _sha256(contract_path),
        "inventory_sha256": _sha256(inventory_path),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "workers": workers,
        "batch_size": batch_size,
        "batches": len(batches),
        "elapsed_seconds": elapsed,
        "output_path": _display_path(output_path),
        "output_sha256": _sha256(output_path),
        "catalog_payload_bytes_downloaded": 0,
        "source_verification": source_verification,
        **_summary_from_rows(rows),
        "training_authorized": False,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = RunReport(
        script="preflight_tess_asassn_labels",
        status="success",
        started_at=started_at.isoformat(),
        completed_at=datetime.now(UTC).isoformat(),
        elapsed_seconds=elapsed,
        items_processed=len(selected),
        items_written=len(rows),
        items_failed=0,
        output_paths=(_display_path(output_path), _display_path(summary_path)),
        shard_index=shard_index,
        shard_count=shard_count,
    )
    report_path = report_path_for(
        "preflight_tess_asassn_labels", shard_index=shard_index, shard_count=shard_count
    )
    if not report_fn(report, report_path):
        print(f"WARNING: Run Report was not committed: {report_path}", flush=True)
    print(
        f"TESS-ASAS-SN shard COMPLETE: matches={summary['matched_rows']}/{len(rows)} "
        f"elapsed={elapsed:.1f}s output={output_path}",
        flush=True,
    )
    return summary


def aggregate_shards(
    contract_path: Path,
    output_path: Path,
    summary_path: Path,
    aggregate_path: Path,
    *,
    shard_count: int,
    report_fn: ReportFn = run_and_commit_report,
) -> dict[str, Any]:
    contract = load_contract(contract_path)
    if contract["source_verification_required"] and shard_count != int(
        contract["parallel_shape"]["process_shards"]
    ):
        raise ValueError("merged ASAS-SN aggregate requires the contracted shard count")
    started_at = datetime.now(UTC)
    started = time.monotonic()
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for shard_index in range(shard_count):
        shard_output = _shard_path(output_path, shard_index, shard_count)
        shard_summary = _shard_path(summary_path, shard_index, shard_count)
        if not shard_output.is_file() or not shard_summary.is_file():
            raise ValueError(f"missing shard artifacts for index {shard_index}")
        summary = json.loads(shard_summary.read_text(encoding="utf-8"))
        if summary.get("shard_index") != shard_index or summary.get("shard_count") != shard_count:
            raise ValueError(f"shard summary identity mismatch: {shard_summary}")
        if summary.get("output_sha256") != _sha256(shard_output):
            raise ValueError(f"shard output hash mismatch: {shard_output}")
        summaries.append(summary)
        all_rows.extend(
            json.loads(line)
            for line in shard_output.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if contract["source_verification_required"]:
        source_status = summaries[0].get("source_verification", {}).get("status")
        if source_status != "pass":
            raise ValueError("shard 0 did not verify ASAS-SN source metadata")
    tic_ids = [int(row["tic_id"]) for row in all_rows]
    duplicate_tics = len(tic_ids) - len(set(tic_ids))
    expected = int(contract["inventory"]["target_count"])
    if len(all_rows) != expected or duplicate_tics:
        raise ValueError(
            f"global TIC reconciliation failed: rows={len(all_rows)} expected={expected} "
            f"duplicates={duplicate_tics}"
        )
    matched = [row for row in all_rows if row["status"] == "matched"]
    sources = [str(row["asassn_id"]) for row in matched]
    duplicate_sources = len(sources) - len(set(sources))
    metrics = _summary_from_rows(all_rows)
    gate = contract["evidence_gate"]
    checks = {
        "minimum_unique_tic_matches": metrics["matched_unique_tics"]
        >= int(gate["minimum_unique_tic_matches"]),
        "minimum_known_variable_matches": metrics["known_variables"]
        >= int(gate["minimum_known_variable_matches"]),
        "minimum_probability": metrics["minimum_probability"] is not None
        and float(metrics["minimum_probability"]) >= float(gate["minimum_probability"]),
        "maximum_duplicate_tic_rows": duplicate_tics
        <= int(gate["maximum_duplicate_tic_rows"]),
        "maximum_duplicate_source_ids": duplicate_sources
        <= int(gate["maximum_duplicate_source_ids"]),
    }
    status = "pass" if all(checks.values()) else "fail"
    artifact = {
        "schema_version": 1,
        "artifact_id": "tess_asassn_preflight_aggregate_v1",
        "status": status,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "contract_path": _display_path(contract_path),
        "contract_sha256": _sha256(contract_path),
        "shard_count": shard_count,
        "workers_per_shard": int(contract["parallel_shape"]["workers_per_shard"]),
        "catalog_payload_bytes_downloaded": 0,
        "source_verification": summaries[0].get("source_verification"),
        "duplicate_tic_rows": duplicate_tics,
        "duplicate_source_ids": duplicate_sources,
        "match_fraction": metrics["matched_unique_tics"] / expected,
        **metrics,
        "checks": checks,
        "shard_summaries": summaries,
        "followup_benchmark_design_authorized": status == "pass",
        "training_authorized": False,
        "limitations": contract["limitations"],
    }
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    elapsed = time.monotonic() - started
    report = RunReport(
        script="preflight_tess_asassn_labels_aggregate",
        status="success" if status == "pass" else "failed",
        started_at=started_at.isoformat(),
        completed_at=datetime.now(UTC).isoformat(),
        elapsed_seconds=elapsed,
        items_processed=len(all_rows),
        items_written=1,
        items_failed=0 if status == "pass" else sum(not value for value in checks.values()),
        output_paths=(_display_path(aggregate_path),),
    )
    report_path = report_path_for("preflight_tess_asassn_labels_aggregate")
    if not report_fn(report, report_path):
        print(f"WARNING: Run Report was not committed: {report_path}", flush=True)
    print(
        f"TESS-ASAS-SN aggregate COMPLETE: status={status.upper()} "
        f"matches={metrics['matched_unique_tics']}/{expected} "
        f"duplicate_sources={duplicate_sources} "
        f"training_authorized=false output={aggregate_path}",
        flush=True,
    )
    return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--aggregate-output", type=Path, default=DEFAULT_AGGREGATE_PATH)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.aggregate_only:
        artifact = aggregate_shards(
            args.contract,
            args.output,
            args.summary,
            args.aggregate_output,
            shard_count=args.shard_count,
        )
        return 0 if artifact["status"] == "pass" else 1
    run_shard(
        args.contract,
        args.output,
        args.summary,
        workers=args.workers,
        batch_size=args.batch_size,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
