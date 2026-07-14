"""Crossmatch cached TESS inventory TICs with pinned Catalina variable labels.

The first authorized run is a deterministic 216-target pilot.  TIC metadata
lookups are batched and threaded within each shard; ``run_six_shards.py``
provides six process shards.  The Catalina table is a single tiny shared,
hash-pinned input and is downloaded once under a filesystem lock.
"""
from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import json
import math
import os
import re
import ssl
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from Skills.run_report import RunReport, report_path_for, run_and_commit_report  # noqa: E402

DEFAULT_CONTRACT = REPO_ROOT / "metadata/tess_catalina_crossmatch_contract_v2.json"
DEFAULT_CACHE = REPO_ROOT / ".cache/stellar_variability_labels/table3.dat.gz"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts/manifests/tess_catalina_crossmatch_pilot_v1.jsonl"
DEFAULT_SUMMARY = REPO_ROOT / "artifacts/manifests/tess_catalina_crossmatch_pilot_v1.json"

DownloadFn = Callable[[str, Path], None]
QueryFn = Callable[[Sequence[int]], Iterable[Mapping[str, Any]]]
ReportFn = Callable[..., bool]


@dataclass(frozen=True)
class CatalinaRecord:
    """One publication-backed Catalina catalog row."""

    source_id: str
    ra_deg: float
    dec_deg: float
    vmag: float
    period_days: float | None
    amplitude_mag: float
    class_code: int
    class_flag: str | None


@dataclass(frozen=True)
class TicRecord:
    """The exact TIC metadata needed for crossmatching and safeguards."""

    tic_id: int
    ra_deg: float
    dec_deg: float
    tmag: float | None
    vmag: float | None
    gaia_id: str | None
    pmra_mas_per_year: float | None
    pmdec_mas_per_year: float | None
    duplicate_tic_id: int | None
    object_type: str | None


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
    """Load and validate the immutable pilot contract."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("TESS-Catalina crossmatch contract schema_version must be 1")
    sections = (
        "inventory",
        "source_evidence",
        "catalog",
        "tic_query",
        "match_policy",
        "pilot",
    )
    for section in sections:
        if not isinstance(raw.get(section), dict):
            raise ValueError(f"contract requires object section {section}")
    for section in ("inventory", "source_evidence"):
        _require_text(raw[section].get("path"), f"{section}.path")
        sha = _require_text(raw[section].get("sha256"), f"{section}.sha256")
        if not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise ValueError(f"{section}.sha256 must be lowercase SHA-256")
    catalog = raw["catalog"]
    for field in ("url", "sha256"):
        _require_text(catalog.get(field), f"catalog.{field}")
    _require_positive_int(catalog.get("size_bytes"), "catalog.size_bytes")
    _require_positive_int(catalog.get("row_count"), "catalog.row_count")
    counts = catalog.get("class_counts")
    count_sum = sum(int(value) for value in counts.values()) if isinstance(counts, dict) else -1
    if not isinstance(counts, dict) or count_sum != catalog["row_count"]:
        raise ValueError("catalog.class_counts must sum to catalog.row_count")
    query = raw["tic_query"]
    _require_positive_int(query.get("batch_size"), "tic_query.batch_size")
    columns = query.get("columns")
    if not isinstance(columns, list) or not columns:
        raise ValueError("tic_query.columns must be a non-empty list")
    policy = raw["match_policy"]
    policy_fields = (
        "candidate_radius_arcsec",
        "accept_radius_arcsec",
        "max_vmag_delta",
        "max_tmag_delta",
    )
    for field in policy_fields:
        value = policy.get(field)
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"match_policy.{field} must be positive")
    if policy["accept_radius_arcsec"] > policy["candidate_radius_arcsec"]:
        raise ValueError("accept radius cannot exceed candidate radius")
    pilot = raw["pilot"]
    _require_positive_int(pilot.get("max_targets"), "pilot.max_targets")
    _require_text(pilot.get("selection_seed"), "pilot.selection_seed")
    if raw.get("training_authorized") is not False:
        raise ValueError("crossmatch contract must keep training_authorized=false")
    return raw


def _contract_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def validate_inputs(contract: Mapping[str, Any]) -> tuple[Path, Path]:
    """Validate inventory and merged source evidence identities."""
    inventory_path = _contract_path(str(contract["inventory"]["path"]))
    evidence_path = _contract_path(str(contract["source_evidence"]["path"]))
    for section, path in (("inventory", inventory_path), ("source_evidence", evidence_path)):
        if not path.is_file():
            raise ValueError(f"{section} path is missing: {path}")
        actual = _sha256(path)
        if actual != contract[section]["sha256"]:
            expected = contract[section]["sha256"]
            raise ValueError(f"{section} SHA-256 mismatch: expected={expected} actual={actual}")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if (
        evidence.get("status") != "success"
        or evidence.get("source_identity_authorized") is not True
    ):
        raise ValueError("merged source evidence does not authorize source identity")
    if evidence.get("training_authorized") is not False:
        raise ValueError("source evidence unexpectedly authorizes training")
    return inventory_path, evidence_path


def load_inventory_tics(path: Path, expected_count: int) -> tuple[int, ...]:
    """Load unique TIC identifiers from the committed representation inventory."""
    values: set[int] = set()
    pattern = re.compile(r"TIC\s+(\d+)")
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            match = pattern.fullmatch(str(row.get("target_id", "")))
            if match is None:
                raise ValueError(f"{path}:{line_number}: invalid target_id")
            values.add(int(match.group(1)))
    if len(values) != expected_count:
        raise ValueError(
            f"inventory TIC count mismatch: expected={expected_count} actual={len(values)}"
        )
    return tuple(sorted(values))


def select_tics(
    tic_ids: Sequence[int],
    *,
    seed: str,
    max_targets: int | None,
    shard_index: int,
    shard_count: int,
) -> tuple[int, ...]:
    """Deterministically sample globally, then partition by TIC modulo."""
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("shard index/count are invalid")
    ordered = sorted(
        tic_ids,
        key=lambda tic_id: (hashlib.sha256(f"{seed}:{tic_id}".encode()).hexdigest(), tic_id),
    )
    selected = ordered if max_targets is None else ordered[:max_targets]
    return tuple(sorted(tic_id for tic_id in selected if tic_id % shard_count == shard_index))


def _ssl_context() -> ssl.SSLContext | None:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return None


def _download(url: str, path: Path) -> None:
    with (
        urllib.request.urlopen(url, timeout=60, context=_ssl_context()) as response,  # noqa: S310
        path.open("wb") as handle,
    ):
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)


def _validate_catalog_file(path: Path, catalog: Mapping[str, Any]) -> None:
    actual_size = path.stat().st_size
    actual_sha = _sha256(path)
    if actual_size != catalog["size_bytes"] or actual_sha != catalog["sha256"]:
        raise ValueError(
            f"Catalina payload mismatch: expected={catalog['size_bytes']}/{catalog['sha256']} "
            f"actual={actual_size}/{actual_sha}"
        )


def ensure_catalog_cache(
    contract: Mapping[str, Any],
    cache_path: Path,
    *,
    download_fn: DownloadFn = _download,
) -> Path:
    """Download the single catalog once under a cross-process lock and verify it."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if cache_path.is_file():
            _validate_catalog_file(cache_path, contract["catalog"])
            return cache_path
        temporary = cache_path.with_name(f"{cache_path.name}.part.{os.getpid()}")
        try:
            download_fn(str(contract["catalog"]["url"]), temporary)
            _validate_catalog_file(temporary, contract["catalog"])
            os.replace(temporary, cache_path)
        finally:
            if temporary.exists():
                temporary.unlink()
    return cache_path


def _optional_float(value: Any) -> float | None:
    if value is None or str(value).strip() in {"", "--", "None"}:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def parse_catalog(path: Path, contract: Mapping[str, Any]) -> tuple[CatalinaRecord, ...]:
    """Parse and validate the pinned 71- to 73-byte fixed-width catalog rows."""
    records: list[CatalinaRecord] = []
    with gzip.open(path, "rt", encoding="ascii") as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.rstrip("\n")
            if not 71 <= len(line) <= 73:
                raise ValueError(
                    f"Catalina row {line_number} is outside the supported 71-73 byte range"
                )
            # CDS omits both trailing blanks when the optional class flag is absent.
            line = line.ljust(73)
            ra = 15.0 * (int(line[21:23]) + int(line[24:26]) / 60 + float(line[27:32]) / 3600)
            dec_abs = int(line[34:36]) + int(line[37:39]) / 60 + float(line[40:44]) / 3600
            dec = -dec_abs if line[33] == "-" else dec_abs
            period_text = line[51:62].strip()
            records.append(
                CatalinaRecord(
                    source_id=line[4:20].strip(),
                    ra_deg=ra,
                    dec_deg=dec,
                    vmag=float(line[45:50]),
                    period_days=float(period_text) if period_text else None,
                    amplitude_mag=float(line[63:68]),
                    class_code=int(line[69:71]),
                    class_flag=line[72:73].strip() or None,
                )
            )
    expected_count = int(contract["catalog"]["row_count"])
    if len(records) != expected_count:
        raise ValueError(
            f"Catalina row count mismatch: expected={expected_count} actual={len(records)}"
        )
    actual_counts = Counter(str(record.class_code) for record in records)
    if dict(actual_counts) != contract["catalog"]["class_counts"]:
        raise ValueError("Catalina parsed class counts do not match the contract")
    return tuple(records)


def _mast_query(
    tic_ids: Sequence[int], columns: Sequence[str]
) -> Iterable[Mapping[str, Any]]:
    from astroquery.mast import Mast

    table = Mast.mast_query(
        "Mast.Catalogs.Filtered.Tic.Rows",
        columns=",".join(columns),
        ID=[str(value) for value in tic_ids],
        pagesize=max(100, len(tic_ids) * 2),
    )
    return ({name: row[name] for name in table.colnames} for row in table)


def _optional_text(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text if text and text not in {"--", "None"} else None


def _tic_record(row: Mapping[str, Any]) -> TicRecord:
    tic_id = int(row["ID"])
    ra = _optional_float(row.get("ra"))
    dec = _optional_float(row.get("dec"))
    if ra is None or dec is None or not (0 <= ra < 360 and -90 <= dec <= 90):
        raise ValueError(f"TIC {tic_id} has invalid coordinates")
    duplicate = _optional_float(row.get("duplicate_id"))
    return TicRecord(
        tic_id=tic_id,
        ra_deg=ra,
        dec_deg=dec,
        tmag=_optional_float(row.get("Tmag")),
        vmag=_optional_float(row.get("Vmag")),
        gaia_id=_optional_text(row.get("GAIA")),
        pmra_mas_per_year=_optional_float(row.get("pmRA")),
        pmdec_mas_per_year=_optional_float(row.get("pmDEC")),
        duplicate_tic_id=int(duplicate) if duplicate and duplicate > 0 else None,
        object_type=_optional_text(row.get("objType")),
    )


def query_tic_metadata(
    tic_ids: Sequence[int],
    *,
    workers: int,
    batch_size: int,
    query_fn: QueryFn,
    retry_attempts: int = 3,
) -> tuple[TicRecord, ...]:
    """Query exact TIC batches concurrently, failing closed on missing/extra rows."""
    if workers <= 0 or batch_size <= 0 or retry_attempts <= 0:
        raise ValueError("workers, batch size, and retry attempts must be positive")
    batches = [
        tuple(tic_ids[index : index + batch_size])
        for index in range(0, len(tic_ids), batch_size)
    ]
    started = time.monotonic()

    def fetch(batch: tuple[int, ...]) -> tuple[TicRecord, ...]:
        last_error: Exception | None = None
        for attempt in range(1, retry_attempts + 1):
            try:
                rows = tuple(_tic_record(row) for row in query_fn(batch))
                actual = [row.tic_id for row in rows]
                if len(actual) != len(set(actual)):
                    raise ValueError(f"MAST returned duplicate TIC rows for batch {batch}")
                if set(actual) != set(batch):
                    requested_text = sorted(batch)
                    returned_text = sorted(actual)
                    raise ValueError(
                        f"MAST TIC batch mismatch: requested={requested_text} "
                        f"returned={returned_text}"
                    )
                return rows
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < retry_attempts:
                    time.sleep(2 ** (attempt - 1))
        raise RuntimeError(f"MAST TIC batch failed after {retry_attempts} attempts: {last_error}")

    results: list[TicRecord] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch, batch): batch for batch in batches}
        for index, future in enumerate(as_completed(futures), 1):
            results.extend(future.result())
            elapsed = time.monotonic() - started
            rate = index / elapsed if elapsed else 0.0
            eta = (len(batches) - index) / rate if rate else 0.0
            print(
                f"  [{index}/{len(batches)}] TIC batches elapsed={elapsed:.1f}s ETA={eta:.1f}s",
                flush=True,
            )
    return tuple(sorted(results, key=lambda row: row.tic_id))


def _family(class_code: int) -> str:
    binary_codes = {1, 2, 3, 14, 16, 17}
    return "binary_false_positive" if class_code in binary_codes else "stellar_variability"


def crossmatch_records(
    tics: Sequence[TicRecord],
    catalog: Sequence[CatalinaRecord],
    policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Apply positional, magnitude, object, duplicate, and blend safeguards."""
    from astropy import units as u
    from astropy.coordinates import SkyCoord, search_around_sky

    tic_coords = SkyCoord(
        [row.ra_deg for row in tics] * u.deg,
        [row.dec_deg for row in tics] * u.deg,
    )
    cat_coords = SkyCoord(
        [row.ra_deg for row in catalog] * u.deg,
        [row.dec_deg for row in catalog] * u.deg,
    )
    indices_tic, indices_cat, separations, _ = search_around_sky(
        tic_coords,
        cat_coords,
        float(policy["candidate_radius_arcsec"]) * u.arcsec,
    )
    candidates: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for tic_index, cat_index, separation in zip(
        indices_tic, indices_cat, separations, strict=True
    ):
        candidates[int(tic_index)].append((int(cat_index), float(separation.arcsec)))

    rows: list[dict[str, Any]] = []
    for tic_index, tic in enumerate(tics):
        nearby = sorted(candidates.get(tic_index, []), key=lambda value: value[1])
        base: dict[str, Any] = {
            "tic_id": tic.tic_id,
            "group_key": f"tess:tic:{tic.tic_id}",
            "tic_ra_deg": tic.ra_deg,
            "tic_dec_deg": tic.dec_deg,
            "tic_tmag": tic.tmag,
            "tic_vmag": tic.vmag,
            "gaia_id": tic.gaia_id,
            "pmra_mas_per_year": tic.pmra_mas_per_year,
            "pmdec_mas_per_year": tic.pmdec_mas_per_year,
            "role": "training",
            "training_authorized": False,
        }
        if not nearby:
            rows.append({**base, "match_status": "no_candidate_within_radius"})
            continue
        cat_index, separation = nearby[0]
        source = catalog[cat_index]
        magnitude_value = tic.vmag if tic.vmag is not None else tic.tmag
        magnitude_band = "Vmag" if tic.vmag is not None else "Tmag"
        magnitude_limit = (
            float(policy["max_vmag_delta"])
            if tic.vmag is not None
            else float(policy["max_tmag_delta"])
        )
        magnitude_delta = (
            abs(source.vmag - magnitude_value) if magnitude_value is not None else None
        )
        reasons: list[str] = []
        if separation > float(policy["accept_radius_arcsec"]):
            reasons.append("separation_exceeds_accept_radius")
        if len(nearby) != 1:
            reasons.append("multiple_catalina_candidates")
        if magnitude_delta is None or magnitude_delta > magnitude_limit:
            reasons.append("magnitude_safeguard_failed")
        if tic.duplicate_tic_id is not None:
            reasons.append("tic_duplicate_entry")
        if tic.object_type is not None and tic.object_type.upper() != "STAR":
            reasons.append("tic_object_not_star")
        if source.class_flag == "f":
            reasons.append("catalina_blend_flag")
        rows.append(
            {
                **base,
                "match_status": "accepted_label" if not reasons else "rejected_match",
                "rejection_reasons": reasons,
                "catalina_source_id": source.source_id,
                "catalina_ra_deg": source.ra_deg,
                "catalina_dec_deg": source.dec_deg,
                "separation_arcsec": separation,
                "catalina_vmag": source.vmag,
                "magnitude_comparison_band": magnitude_band,
                "magnitude_delta": magnitude_delta,
                "period_days": source.period_days,
                "amplitude_mag": source.amplitude_mag,
                "class_code": source.class_code,
                "class_flag": source.class_flag,
                "benchmark_family": _family(source.class_code),
                "label_eligible": not reasons,
            }
        )

    accepted_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["match_status"] == "accepted_label":
            accepted_by_source[str(row["catalina_source_id"])].append(row)
    for duplicates in accepted_by_source.values():
        if len(duplicates) > 1:
            for row in duplicates:
                row["match_status"] = "rejected_match"
                row["label_eligible"] = False
                row["rejection_reasons"].append("catalina_source_matches_multiple_tics")
    return sorted(rows, key=lambda row: int(row["tic_id"]))


def _shard_path(path: Path, shard_index: int, shard_count: int) -> Path:
    if shard_count <= 1:
        return path
    return path.with_name(f"{path.stem}.shard{shard_index}of{shard_count}{path.suffix}")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")


def run_crossmatch(
    contract_path: Path,
    cache_path: Path,
    output_path: Path,
    summary_path: Path,
    *,
    workers: int,
    batch_size: int,
    max_targets: int | None,
    shard_index: int,
    shard_count: int,
    dry_run: bool = False,
    download_fn: DownloadFn = _download,
    query_fn: QueryFn | None = None,
    report_fn: ReportFn = run_and_commit_report,
) -> dict[str, Any]:
    """Run one deterministic shard or validate it without network/file output."""
    contract = load_contract(contract_path)
    inventory_path, evidence_path = validate_inputs(contract)
    inventory_count = int(contract["inventory"]["target_count"])
    all_tics = load_inventory_tics(inventory_path, inventory_count)
    pilot_limit = int(contract["pilot"]["max_targets"])
    outside_pilot = max_targets is None or max_targets > pilot_limit
    if outside_pilot and contract["pilot"].get("full_run_authorized") is not True:
        raise ValueError(
            f"only a bounded pilot of at most {pilot_limit} TICs is authorized; "
            "full crossmatch remains gated"
        )
    if batch_size > int(contract["tic_query"]["max_batch_size"]):
        raise ValueError("batch size exceeds the contract maximum")
    selected = select_tics(
        all_tics,
        seed=str(contract["pilot"]["selection_seed"]),
        max_targets=max_targets,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    if not selected:
        raise ValueError(f"shard {shard_index}/{shard_count} has no selected TICs")
    output_path = _shard_path(output_path, shard_index, shard_count)
    summary_path = _shard_path(summary_path, shard_index, shard_count)
    batches = math.ceil(len(selected) / batch_size)
    print(
        "TESS-Catalina crossmatch startup: "
        f"selected_tics={len(selected)} batches={batches} workers={workers} "
        f"shard={shard_index}/{shard_count} pilot_limit={pilot_limit} "
        f"training_authorized=false dry_run={str(dry_run).lower()}",
        flush=True,
    )
    if dry_run:
        return {
            "status": "dry_run",
            "selected_tics": len(selected),
            "batches": batches,
            "shard_index": shard_index,
            "shard_count": shard_count,
            "output_path": str(output_path),
            "summary_path": str(summary_path),
        }

    started_at = datetime.now(UTC)
    started = time.monotonic()
    catalog_path = ensure_catalog_cache(contract, cache_path, download_fn=download_fn)
    catalog = parse_catalog(catalog_path, contract)
    columns = tuple(str(value) for value in contract["tic_query"]["columns"])
    effective_query_fn = query_fn or (lambda values: _mast_query(values, columns))
    tics = query_tic_metadata(
        selected,
        workers=workers,
        batch_size=batch_size,
        query_fn=effective_query_fn,
        retry_attempts=int(contract["tic_query"]["retry_attempts"]),
    )
    rows = crossmatch_records(tics, catalog, contract["match_policy"])
    _write_jsonl(output_path, rows)
    counts = Counter(str(row["match_status"]) for row in rows)
    elapsed = time.monotonic() - started
    summary: dict[str, Any] = {
        "schema_version": 1,
        "artifact_id": "tess_catalina_crossmatch_pilot_v1",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "status": "success",
        "contract_path": str(contract_path),
        "contract_sha256": _sha256(contract_path),
        "inventory_path": str(inventory_path),
        "source_evidence_path": str(evidence_path),
        "catalog_sha256": _sha256(catalog_path),
        "selected_tics": len(selected),
        "queried_tics": len(tics),
        "rows_written": len(rows),
        "match_status_counts": dict(sorted(counts.items())),
        "elapsed_seconds": elapsed,
        "tic_rate_per_second": len(tics) / elapsed if elapsed else 0.0,
        "workers": workers,
        "batch_size": batch_size,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "output_path": str(output_path),
        "output_sha256": _sha256(output_path),
        "training_authorized": False,
        "limitations": contract["limitations"],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = RunReport(
        script="crossmatch_tess_catalina_labels",
        status="success",
        started_at=started_at.isoformat(),
        completed_at=datetime.now(UTC).isoformat(),
        elapsed_seconds=elapsed,
        items_processed=len(tics),
        items_written=int(counts.get("accepted_label", 0)),
        output_paths=(str(output_path), str(summary_path)),
        shard_index=shard_index,
        shard_count=shard_count,
        notes="bounded label-source pilot; metadata plus 1.17 MB shared catalog; training disabled",
    )
    report_path = report_path_for(
        "crossmatch_tess_catalina_labels",
        shard_index=shard_index,
        shard_count=shard_count,
    )
    if not report_fn(report, report_path):
        print(f"WARNING: Run Report push failed for {report_path}", flush=True)
    print(
        "TESS-Catalina crossmatch COMPLETE: "
        f"tics={len(tics)} accepted={counts.get('accepted_label', 0)} "
        f"rejected={counts.get('rejected_match', 0)} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--catalog-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--max-targets", type=int, default=216)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        run_crossmatch(
            args.contract,
            args.catalog_cache,
            args.output,
            args.summary,
            workers=args.workers,
            batch_size=args.batch_size,
            max_targets=args.max_targets,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            dry_run=args.dry_run,
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"ERROR: TESS-Catalina crossmatch failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
