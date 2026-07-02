"""Build leakage-safe T1-1 training manifests from verified catalog rows.

This script is intentionally catalog-only: it queries public source rows and
writes compact, committed metadata before any raw light-curve download. It
does not download FITS files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from verify_dataset_sources import REQUIRED_KOI_COLUMNS, verify_table_schema

TAP_SYNC = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
MANIFEST_VERSION = "t1-1-kepler-manifest-v1"
DEFAULT_SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

TapFn = Callable[[str], str]


@dataclass(frozen=True)
class TrainingManifestRow:
    """One supervised training example planned before raw download."""

    manifest_version: str
    source: str
    source_table: str
    mission: str
    target_id: int
    target_name: str
    source_row_id: str
    group_key: str
    split: str
    label: int
    label_name: str
    period_days: float
    epoch_bkjd: float
    duration_hours: float
    lightcurve_search: dict[str, Any]


@dataclass(frozen=True)
class CleanupPolicy:
    """Raw/processed retention policy for the manifest's first processing batch."""

    raw_lightcurve_dir: str
    processed_artifact_dir: str
    runtime_log_db: str
    keep_after_verified_processing: tuple[str, ...]
    delete_after_verified_processing: tuple[str, ...]
    delete_only_after: tuple[str, ...]


@dataclass(frozen=True)
class ManifestSummary:
    """Summary and validation status for a training manifest."""

    created_at_utc: str
    manifest_version: str
    source_snapshot: str
    row_count: int
    target_count: int
    split_counts: dict[str, int]
    split_label_counts: dict[str, dict[str, int]]
    label_counts: dict[str, int]
    leakage_errors: tuple[str, ...]
    cleanup_policy: CleanupPolicy
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


def _tap_url(query: str) -> str:
    return f"{TAP_SYNC}?{urllib.parse.urlencode({'query': query, 'format': 'csv'})}"


def _csv_rows(raw: str) -> list[dict[str, str]]:
    clean_lines = [line for line in raw.splitlines() if not line.startswith("#")]
    if not clean_lines:
        return []
    return list(csv.DictReader(io.StringIO("\n".join(clean_lines))))


def _tap_rows(query: str, tap_fn: TapFn) -> list[dict[str, str]]:
    return _csv_rows(tap_fn(_tap_url(query)))


def _int_value(value: object) -> int:
    if value is None or value == "":
        return 0
    return int(float(str(value)))


def _float_value(value: object) -> float:
    if value is None or value == "":
        return 0.0
    return float(str(value))


def _stable_unit_interval(text: str) -> float:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16)


def _split_for_group(group_key: str, *, seed: int, ratios: dict[str, float]) -> str:
    value = _stable_unit_interval(f"{seed}:{group_key}")
    train_cut = ratios["train"]
    val_cut = train_cut + ratios["val"]
    if value < train_cut:
        return "train"
    if value < val_cut:
        return "val"
    return "test"


def _koi_training_rows(tap_fn: TapFn) -> list[dict[str, str]]:
    return _tap_rows(
        "select kepid,kepoi_name,koi_disposition,koi_period,koi_time0bk,koi_duration "
        "from cumulative "
        "where (koi_disposition='CONFIRMED' or koi_disposition='FALSE POSITIVE') "
        "and koi_period > 0.5 and koi_period < 500 "
        "and koi_time0bk > 0 and koi_duration > 0",
        tap_fn,
    )


def _source_snapshot_id(path: Path) -> str:
    if not path.exists():
        return "unavailable"
    payload = json.loads(path.read_text(encoding="utf-8"))
    created = str(payload.get("created_at_utc", "unknown"))
    sources = payload.get("sources", [])
    names = ",".join(str(source.get("name", "")) for source in sources)
    return f"{created}:{names}"


def _progress(index: int, total: int, start: float) -> None:
    if total == 0:
        return
    if index != total and index % 500 != 0:
        return
    elapsed = time.monotonic() - start
    rate = index / elapsed if elapsed > 0 else 0.0
    remaining = (total - index) / rate if rate > 0 else 0.0
    print(
        f"  [{index}/{total}] manifest rows elapsed={elapsed:.0f}s ETA={remaining:.0f}s",
        flush=True,
    )


def build_kepler_manifest(
    *,
    tap_fn: TapFn | None = None,
    source_snapshot_path: Path = Path("metadata/source_snapshots.json"),
    seed: int = 42,
    split_ratios: dict[str, float] | None = None,
    created_at_utc: str | None = None,
) -> tuple[list[TrainingManifestRow], ManifestSummary]:
    """Build a target-grouped Kepler KOI manifest and validate leakage."""
    _tap = tap_fn or _default_fetch
    ratios = split_ratios or DEFAULT_SPLIT_RATIOS
    if not {"train", "val", "test"} <= set(ratios):
        raise ValueError("split_ratios must include train, val, and test")
    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise ValueError("split_ratios must sum to 1.0")

    schema = verify_table_schema("cumulative", REQUIRED_KOI_COLUMNS, tap_fn=_tap)
    if not schema.ok:
        policy = default_cleanup_policy()
        schema_errors = list(schema.missing_columns)
        if schema.error:
            schema_errors.append(schema.error)
        summary = ManifestSummary(
            created_at_utc=created_at_utc
            or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            manifest_version=MANIFEST_VERSION,
            source_snapshot=_source_snapshot_id(source_snapshot_path),
            row_count=0,
            target_count=0,
            split_counts={},
            split_label_counts={},
            label_counts={},
            leakage_errors=tuple(sorted(schema_errors)),
            cleanup_policy=policy,
            flag="SCHEMA_FAIL",
            next_action="Stop: cumulative schema verification failed before manifest creation.",
        )
        return [], summary

    source_rows = _koi_training_rows(_tap)
    created = created_at_utc or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest_rows: list[TrainingManifestRow] = []
    start = time.monotonic()
    total = len(source_rows)
    for index, row in enumerate(source_rows, 1):
        target_id = _int_value(row["kepid"])
        disposition = str(row["koi_disposition"])
        label = 1 if disposition == "CONFIRMED" else 0
        group_key = f"kepler:kic:{target_id}"
        split = _split_for_group(group_key, seed=seed, ratios=ratios)
        manifest_rows.append(
            TrainingManifestRow(
                manifest_version=MANIFEST_VERSION,
                source="nasa_exoplanet_archive_dr25_koi",
                source_table="cumulative",
                mission="Kepler",
                target_id=target_id,
                target_name=f"KIC {target_id}",
                source_row_id=str(row["kepoi_name"]),
                group_key=group_key,
                split=split,
                label=label,
                label_name=disposition,
                period_days=_float_value(row["koi_period"]),
                epoch_bkjd=_float_value(row["koi_time0bk"]),
                duration_hours=_float_value(row["koi_duration"]),
                lightcurve_search={
                    "target": f"KIC {target_id}",
                    "mission": "Kepler",
                    "author": "Kepler",
                    "exptime": 1800,
                },
            )
        )
        _progress(index, total, start)

    summary = summarize_manifest(
        manifest_rows,
        created_at_utc=created,
        source_snapshot=_source_snapshot_id(source_snapshot_path),
        cleanup_policy=default_cleanup_policy(),
    )
    return manifest_rows, summary


def default_cleanup_policy() -> CleanupPolicy:
    """Return the committed cleanup policy for the first Kepler processing batch."""
    return CleanupPolicy(
        raw_lightcurve_dir="data/raw/t1_1_kepler_lc",
        processed_artifact_dir="data/processed/t1_1_kepler_snippets",
        runtime_log_db="logs/t1_1_kepler_processing.sqlite3",
        keep_after_verified_processing=(
            "metadata/t1_1_kepler_training_manifest.jsonl",
            "metadata/t1_1_kepler_manifest_summary.json",
            "data/processed/t1_1_kepler_snippets",
            "logs/t1_1_kepler_processing.sqlite3",
        ),
        delete_after_verified_processing=("data/raw/t1_1_kepler_lc",),
        delete_only_after=(
            "processed snippets validate",
            "manifest summary flag is OK",
            "runtime SQLite log has no incomplete active targets",
            "operator confirms no failed raw FITS are needed for debugging",
        ),
    )


def summarize_manifest(
    rows: Sequence[TrainingManifestRow],
    *,
    created_at_utc: str,
    source_snapshot: str,
    cleanup_policy: CleanupPolicy,
) -> ManifestSummary:
    """Summarize and validate a manifest for target leakage."""
    split_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    split_label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    group_to_split: dict[str, str] = {}
    leakage_errors: list[str] = []

    for row in rows:
        split_counts[row.split] += 1
        label_key = str(row.label)
        label_counts[label_key] += 1
        split_label_counts[row.split][label_key] += 1
        previous = group_to_split.setdefault(row.group_key, row.split)
        if previous != row.split:
            leakage_errors.append(
                f"group {row.group_key} appears in both {previous} and {row.split}"
            )

    for split in ("train", "val", "test"):
        if split_counts[split] == 0:
            leakage_errors.append(f"split {split} has zero rows")
        for label in ("0", "1"):
            if split_label_counts[split][label] == 0:
                leakage_errors.append(f"split {split} has zero label={label} rows")

    flag = "OK" if not leakage_errors and rows else "LEAKAGE_FAIL"
    return ManifestSummary(
        created_at_utc=created_at_utc,
        manifest_version=MANIFEST_VERSION,
        source_snapshot=source_snapshot,
        row_count=len(rows),
        target_count=len(group_to_split),
        split_counts=dict(sorted(split_counts.items())),
        split_label_counts={
            split: dict(sorted(counts.items()))
            for split, counts in sorted(split_label_counts.items())
        },
        label_counts=dict(sorted(label_counts.items())),
        leakage_errors=tuple(leakage_errors),
        cleanup_policy=cleanup_policy,
        flag=flag,
        next_action=(
            "Proceed to a bounded Kepler-first processing batch using this manifest."
            if flag == "OK"
            else "Stop and fix manifest leakage or label coverage before any download."
        ),
    )


def format_summary(summary: ManifestSummary) -> str:
    """Render a concise Markdown summary."""
    lines = [
        "# T1-1 Kepler Training Manifest",
        "",
        f"**Flag**: {summary.flag}",
        f"**Created at UTC**: {summary.created_at_utc}",
        f"**Rows**: {summary.row_count}",
        f"**Target groups**: {summary.target_count}",
        "",
        "## Split Counts",
        "| Split | Rows | Label 0 | Label 1 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for split in ("train", "val", "test"):
        labels = summary.split_label_counts.get(split, {})
        lines.append(
            f"| {split} | {summary.split_counts.get(split, 0)} | "
            f"{labels.get('0', 0)} | {labels.get('1', 0)} |"
        )
    lines += [
        "",
        "## Cleanup Policy",
        f"- Raw light-curve directory: `{summary.cleanup_policy.raw_lightcurve_dir}`",
        f"- Processed artifact directory: `{summary.cleanup_policy.processed_artifact_dir}`",
        f"- Runtime SQLite log: `{summary.cleanup_policy.runtime_log_db}`",
        "- Delete raw FITS only after: "
        + "; ".join(summary.cleanup_policy.delete_only_after),
        "",
        "## Next Action",
        summary.next_action,
    ]
    if summary.leakage_errors:
        lines += ["", "## Leakage Errors", *[f"- {error}" for error in summary.leakage_errors]]
    return "\n".join(lines) + "\n"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_manifest_outputs(
    rows: Sequence[TrainingManifestRow],
    summary: ManifestSummary,
    *,
    manifest_path: Path,
    summary_path: Path,
    report_path: Path,
) -> None:
    """Write JSONL manifest, JSON summary, and operator report."""
    _write_jsonl(manifest_path, (asdict(row) for row in rows))
    _write_json(summary_path, asdict(summary))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(format_summary(summary), encoding="utf-8")


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a leakage-safe T1-1 Kepler training manifest from verified "
            "NASA Exoplanet Archive KOI labels. Does not download FITS files."
        )
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--source-snapshot",
        type=Path,
        default=Path("metadata/source_snapshots.json"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("metadata/t1_1_kepler_training_manifest.jsonl"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("metadata/t1_1_kepler_manifest_summary.json"),
    )
    parser.add_argument("--report", type=Path, default=Path("reports/t1-1_kepler_manifest.md"))
    args = parser.parse_args(argv)

    rows, summary = build_kepler_manifest(
        source_snapshot_path=args.source_snapshot,
        seed=args.seed,
    )
    write_manifest_outputs(
        rows,
        summary,
        manifest_path=args.manifest,
        summary_path=args.summary,
        report_path=args.report,
    )
    print(format_summary(summary), flush=True)
    print(f"Saved manifest to {args.manifest}", flush=True)
    print(f"Saved summary to {args.summary}", flush=True)
    print(f"Saved report to {args.report}", flush=True)
    return 0 if summary.flag == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
