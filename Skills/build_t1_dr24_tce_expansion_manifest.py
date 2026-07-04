"""Build a leakage-safe T1-1 expansion manifest from the Kepler DR24 TCE table.

Adds genuinely new KIC targets that are not already present in the committed
T1-1 KOI manifest (``metadata/t1_1_kepler_training_manifest.jsonl``), using
the ``Q1_Q17_DR24_TCE`` table's ``av_training_set`` labels: ``PC`` (planet
candidate) and ``AFP``/``NTP`` (astrophysical false positive / non-transiting
phenomenon) map to label 1/0/0 respectively. ``UNK`` rows are excluded as an
ambiguous label, matching this project's existing KOI ``CANDIDATE``-exclusion
policy. Does not download FITS files.

Public API
----------
REQUIRED_DR24_TCE_COLUMNS
load_existing_target_ids(manifest_path) -> frozenset[int]
build_dr24_expansion_manifest(*, tap_fn, existing_manifest_path,
    source_snapshot_path, seed, split_ratios, created_at_utc)
    -> tuple[list[TrainingManifestRow], ManifestSummary]
"""
from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from build_t1_training_manifest import (
    DEFAULT_SPLIT_RATIOS,
    ManifestSummary,
    TrainingManifestRow,
    _default_fetch,
    _float_value,
    _int_value,
    _progress,
    _source_snapshot_id,
    _split_for_group,
    _tap_rows,
    default_cleanup_policy,
    format_summary,
    summarize_manifest,
    write_manifest_outputs,
)
from verify_dataset_sources import verify_table_schema

MANIFEST_VERSION = "t1-1-kepler-dr24-tce-expansion-v1"
SOURCE_TABLE = "Q1_Q17_DR24_TCE"
REQUIRED_DR24_TCE_COLUMNS: frozenset[str] = frozenset(
    {"kepid", "tce_plnt_num", "av_training_set", "tce_period", "tce_time0bk", "tce_duration"}
)
_LABEL_MAP: dict[str, int] = {"PC": 1, "AFP": 0, "NTP": 0}

TapFn = Callable[[str], str]


def load_existing_target_ids(manifest_path: Path) -> frozenset[int]:
    """Return the ``target_id`` values already present in a committed manifest JSONL file.

    Returns an empty set if the file does not exist, so this script can be
    exercised (and tested) without requiring the KOI manifest to be present.
    """
    if not manifest_path.exists():
        return frozenset()
    ids: set[int] = set()
    with manifest_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids.add(int(json.loads(line)["target_id"]))
    return frozenset(ids)


def _dr24_tce_training_rows(tap_fn: TapFn) -> list[dict[str, str]]:
    return _tap_rows(
        "select kepid,tce_plnt_num,av_training_set,tce_period,tce_time0bk,tce_duration "
        f"from {SOURCE_TABLE} "
        "where av_training_set='PC' or av_training_set='AFP' or av_training_set='NTP'",
        tap_fn,
    )


def build_dr24_expansion_manifest(
    *,
    tap_fn: TapFn | None = None,
    existing_manifest_path: Path = Path("metadata/t1_1_kepler_training_manifest.jsonl"),
    source_snapshot_path: Path = Path("metadata/source_snapshots.json"),
    seed: int = 42,
    split_ratios: dict[str, float] | None = None,
    created_at_utc: str | None = None,
) -> tuple[list[TrainingManifestRow], ManifestSummary]:
    """Build a target-grouped DR24 TCE expansion manifest, excluding known targets."""
    _tap = tap_fn or _default_fetch
    ratios = split_ratios or DEFAULT_SPLIT_RATIOS
    if not {"train", "val", "test"} <= set(ratios):
        raise ValueError("split_ratios must include train, val, and test")
    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise ValueError("split_ratios must sum to 1.0")

    created = created_at_utc or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    schema = verify_table_schema(SOURCE_TABLE, REQUIRED_DR24_TCE_COLUMNS, tap_fn=_tap)
    if not schema.ok:
        schema_errors = list(schema.missing_columns)
        if schema.error:
            schema_errors.append(schema.error)
        summary = ManifestSummary(
            created_at_utc=created,
            manifest_version=MANIFEST_VERSION,
            source_snapshot=_source_snapshot_id(source_snapshot_path),
            row_count=0,
            target_count=0,
            split_counts={},
            split_label_counts={},
            label_counts={},
            leakage_errors=tuple(sorted(schema_errors)),
            cleanup_policy=default_cleanup_policy(),
            flag="SCHEMA_FAIL",
            next_action=(
                f"Stop: {SOURCE_TABLE} schema verification failed before manifest creation."
            ),
        )
        return [], summary

    existing_ids = load_existing_target_ids(existing_manifest_path)
    source_rows = _dr24_tce_training_rows(_tap)
    manifest_rows: list[TrainingManifestRow] = []
    start = time.monotonic()
    total = len(source_rows)
    for index, row in enumerate(source_rows, 1):
        target_id = _int_value(row["kepid"])
        disposition = str(row["av_training_set"])
        label = _LABEL_MAP.get(disposition)
        period = _float_value(row["tce_period"])
        epoch = _float_value(row["tce_time0bk"])
        duration = _float_value(row["tce_duration"])
        if (
            target_id not in existing_ids
            and label is not None
            and 0.5 < period < 500
            and epoch > 0
            and duration > 0
        ):
            group_key = f"kepler:kic:{target_id}"
            split = _split_for_group(group_key, seed=seed, ratios=ratios)
            manifest_rows.append(
                TrainingManifestRow(
                    manifest_version=MANIFEST_VERSION,
                    source="nasa_exoplanet_archive_dr24_tce",
                    source_table=SOURCE_TABLE,
                    mission="Kepler",
                    target_id=target_id,
                    target_name=f"KIC {target_id}",
                    source_row_id=f"KIC{target_id}-TCE{row['tce_plnt_num']}",
                    group_key=group_key,
                    split=split,
                    label=label,
                    label_name=disposition,
                    period_days=period,
                    epoch_bkjd=epoch,
                    duration_hours=duration,
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


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a leakage-safe T1-1 Kepler DR24 TCE expansion manifest, adding only "
            "targets not already present in the committed KOI manifest. Does not "
            "download FITS files."
        )
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--existing-manifest",
        type=Path,
        default=Path("metadata/t1_1_kepler_training_manifest.jsonl"),
    )
    parser.add_argument(
        "--source-snapshot",
        type=Path,
        default=Path("metadata/source_snapshots.json"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("metadata/t1_1_kepler_dr24_expansion_manifest.jsonl"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("metadata/t1_1_kepler_dr24_expansion_manifest_summary.json"),
    )
    parser.add_argument(
        "--report", type=Path, default=Path("reports/t1-1_kepler_dr24_expansion_manifest.md")
    )
    args = parser.parse_args(argv)

    rows, summary = build_dr24_expansion_manifest(
        existing_manifest_path=args.existing_manifest,
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
