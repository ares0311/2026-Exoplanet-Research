"""Build a leakage-safe T1-2 held-out calibration manifest from K2 pandc.

T1-2 (stacking Tier-3 weight calibration) requires a held-out set that was
never used to train the XGBoost KOI model or the promoted ``benchmark_cnn_v1``
Kepler CNN checkpoint. Both of those were trained/split entirely from Kepler
prime-mission KIC targets (NASA Exoplanet Archive ``cumulative`` KOI table
plus the ``Q1_Q17_DR24_TCE`` expansion). The K2 mission's own planet catalog
(``k2pandc``) uses a disjoint EPIC target catalog observed during different
campaigns/sky fields than the original Kepler mission, so its rows carry zero
group-level leakage risk against the existing Kepler training/val/frozen-eval
splits by construction (``k2:epic:<id>`` group keys can never collide with the
``kepler:kic:<id>`` group keys used by ``build_t1_training_manifest.py`` and
``build_t1_dr24_tce_expansion_manifest.py``).

This script is catalog-only: it queries public K2 pandc rows and writes
compact, committed metadata before any native K2 light-curve download.  It
does not download FITS files.

Column units (verified live against ``tap_schema.columns`` on 2026-07-09 —
do not assume KOI-style units without checking, NASA archive tables use
different units across catalogs):

- ``pl_trandep`` is Transit Depth in **percent** (KOI's ``koi_depth`` is ppm) —
  confirmed correct against K2-18 b (pl_trandep ~0.29-0.35, matching its
  well-published ~0.3% transit depth)
- ``pl_trandur`` is empirically already in **hours**, NOT days as
  ``tap_schema.columns`` claims ("Transit Duration [day]", unit="day") — this
  is a live-confirmed NASA archive schema metadata inconsistency, found by
  cross-checking real values against literature: K2-18 b (period 32.94 d) has
  ``pl_trandur`` values of 2.663, 2.682, 2.97, matching its well-published
  ~2.3-2.7 HOUR transit duration; a 2.7-DAY transit on a 32.9-day period
  would be astronomically implausible. K2-3 b (period 10.05 d) shows the same
  pattern (``pl_trandur`` ~2.5-3.05, matching its published ~2.5-3 hour
  duration). Do not multiply this column by 24 — use it as hours directly.
- ``pl_orbper`` is Orbital Period in days (matches KOI's ``koi_period``)
- ``pl_rade`` is Planet Radius in Earth radii (matches KOI's ``koi_prad``)
- ``sy_pnum`` is Number of Planets in the system (proxy for KOI's
  ``koi_count``, which is "Number of KOIs Identified in the System")
- ``pl_tranmid`` is already full BJD_TDB — verified live via both
  ``tap_schema.columns`` and the raw value's magnitude (~2457000+, matching
  the K2 mission era). It is **not** BKJD (BJD − 2454833) despite superficial
  resemblance to Kepler's ``koi_time0bk`` convention, and despite an earlier,
  now-fixed assumption to the contrary in
  ``Skills/fetch_tess_k2_overlap_snippets.py`` (that bug double-added the
  BKJD offset before this manifest builder was written; both are now
  corrected). Do not add a BKJD offset to ``pl_tranmid``.

Public API
----------
CalibrationManifestRow(...)
CalibrationManifestSummary(...)
verify_k2_schema(tap_fn) -> tuple[bool, list[str]]
build_k2_calibration_manifest(*, tap_fn, seed, sample_size,
                              created_at_utc) -> tuple[list[Row], Summary]
write_manifest_outputs(rows, summary, *, manifest_path, summary_path, report_path)
format_summary(summary) -> str
"""
from __future__ import annotations

import argparse
import csv
import io
import random
import time
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TAP_SYNC = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
MANIFEST_VERSION = "t1-2-k2-calibration-manifest-v1"

REQUIRED_K2_COLUMNS: tuple[str, ...] = (
    "epic_candname",
    "disposition",
    "pl_orbper",
    "pl_tranmid",
    "pl_trandep",
    "pl_trandur",
    "pl_rade",
    "sy_pnum",
)

TapFn = Callable[[str], str]


@dataclass(frozen=True)
class CalibrationManifestRow:
    """One held-out calibration example for T1-2 stacking weight calibration."""

    manifest_version: str
    source: str
    source_table: str
    mission: str
    epic_id: int
    group_key: str
    label: int
    label_name: str
    period_days: float
    epoch_bjd: float    # Already full BJD_TDB; do not add a BKJD offset
    duration_hours: float | None
    depth_ppm: float | None
    planet_radius_rearth: float | None
    system_planet_count: int | None
    lightcurve_search: dict[str, Any]


@dataclass(frozen=True)
class CalibrationManifestSummary:
    """Summary and validation status for the T1-2 calibration manifest."""

    created_at_utc: str
    manifest_version: str
    row_count: int
    label_counts: dict[str, int]
    seed: int
    sample_size_requested: int
    n_available_confirmed: int
    n_available_false_positive: int
    n_duplicate_epic_source_rows: int
    leakage_errors: tuple[str, ...]
    flag: str
    next_action: str


# ---------------------------------------------------------------------------
# TAP plumbing (mirrors build_t1_training_manifest.py exactly for consistency)
# ---------------------------------------------------------------------------


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


def verify_k2_schema(tap_fn: TapFn) -> tuple[bool, list[str]]:
    """Verify k2pandc has every column this manifest builder depends on.

    Returns:
        ``(ok, missing_columns)``. Never guesses column names.
    """
    rows = _tap_rows(
        "select column_name from tap_schema.columns where table_name='k2pandc'",
        tap_fn,
    )
    available = {str(row.get("column_name", "")).lower() for row in rows}
    missing = [c for c in REQUIRED_K2_COLUMNS if c not in available]
    return (not missing), missing


def _float_or_none(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        f = float(str(value))
    except (TypeError, ValueError):
        return None
    return f if f == f else None  # filter NaN


def _int_or_none(value: object) -> int | None:
    f = _float_or_none(value)
    return None if f is None else int(f)


def _parse_epic_id(raw: object) -> int | None:
    text = str(raw).strip()
    if text.upper().startswith("EPIC"):
        text = text.upper().replace("EPIC", "").strip().split(".")[0]
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _fetch_k2_calibration_rows(tap_fn: TapFn) -> list[dict[str, str]]:
    return _tap_rows(
        "select epic_candname,disposition,pl_orbper,pl_tranmid,"
        "pl_trandep,pl_trandur,pl_rade,sy_pnum "
        "from k2pandc "
        "where pl_orbper is not null and pl_tranmid is not null",
        tap_fn,
    )


def _progress(index: int, total: int, start: float) -> None:
    if total == 0:
        return
    if index != total and index % 500 != 0:
        return
    elapsed = time.monotonic() - start
    rate = index / elapsed if elapsed > 0 else 0.0
    remaining = (total - index) / rate if rate > 0 else 0.0
    print(
        f"  [{index}/{total}] calibration rows elapsed={elapsed:.0f}s "
        f"ETA={remaining:.0f}s",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Manifest construction
# ---------------------------------------------------------------------------


def build_k2_calibration_manifest(
    *,
    tap_fn: TapFn | None = None,
    seed: int = 42,
    sample_size: int = 600,
    created_at_utc: str | None = None,
) -> tuple[list[CalibrationManifestRow], CalibrationManifestSummary]:
    """Build a deterministic, leakage-safe K2 held-out calibration manifest.

    Uses every available FALSE POSITIVE row (k2pandc has few) plus a seeded
    sample of CONFIRMED rows up to ``sample_size`` total, so the calibration
    set is realistically class-balanced rather than mirroring k2pandc's
    natural ~87%/13% confirmed/false-positive prevalence.

    Args:
        tap_fn: Injectable TAP fetch function (for tests). Defaults to a live
            HTTPS TAP sync query.
        seed: Deterministic sampling seed.
        sample_size: Target total row count (all available false positives
            plus a sampled subset of confirmed rows, capped at this total).
        created_at_utc: Override for the manifest timestamp (for tests).

    Returns:
        ``(rows, summary)``. ``rows`` is empty and ``summary.flag`` is
        ``"SCHEMA_FAIL"`` or ``"INSUFFICIENT"`` when the source cannot be
        used safely.
    """
    _tap = tap_fn or _default_fetch
    created = created_at_utc or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    schema_ok, missing = verify_k2_schema(_tap)
    if not schema_ok:
        return [], CalibrationManifestSummary(
            created_at_utc=created,
            manifest_version=MANIFEST_VERSION,
            row_count=0,
            label_counts={},
            seed=seed,
            sample_size_requested=sample_size,
            n_available_confirmed=0,
            n_available_false_positive=0,
            n_duplicate_epic_source_rows=0,
            leakage_errors=tuple(sorted(missing)),
            flag="SCHEMA_FAIL",
            next_action="Stop: k2pandc schema verification failed before manifest creation.",
        )

    raw_rows = _fetch_k2_calibration_rows(_tap)

    by_epic: dict[int, dict[str, Any]] = {}
    n_duplicate_source_rows = 0
    start = time.monotonic()
    total = len(raw_rows)
    for index, row in enumerate(raw_rows, 1):
        epic_id = _parse_epic_id(row.get("epic_candname"))
        disposition = str(row.get("disposition", "")).strip().upper()
        period = _float_or_none(row.get("pl_orbper"))
        # pl_tranmid is already full BJD_TDB — see module docstring. Do not
        # add a BKJD offset (this bug was found and fixed live on 2026-07-10).
        epoch_bjd = _float_or_none(row.get("pl_tranmid"))
        if (
            epic_id is None
            or disposition not in {"CONFIRMED", "FALSE POSITIVE"}
            or period is None
            or period <= 0
            or epoch_bjd is None
        ):
            _progress(index, total, start)
            continue

        if epic_id in by_epic:
            # A star can have multiple K2 planet-candidate entries; keep the
            # first-seen row per EPIC target so calibration examples remain
            # one-per-star and independent.
            n_duplicate_source_rows += 1
            _progress(index, total, start)
            continue

        depth_pct = _float_or_none(row.get("pl_trandep"))
        # pl_trandur is empirically already in hours despite the archive's own
        # schema metadata claiming "day" — see module docstring for the
        # literature cross-check (K2-18 b, K2-3 b). Do not multiply by 24.
        dur_hours = _float_or_none(row.get("pl_trandur"))
        by_epic[epic_id] = {
            "epic_id": epic_id,
            "label": 1 if disposition == "CONFIRMED" else 0,
            "label_name": disposition,
            "period_days": period,
            "epoch_bjd": epoch_bjd,
            # Unit conversions verified live via tap_schema.columns and, for
            # pl_trandur, cross-checked against literature values — see
            # module docstring. Do not change without re-verifying units.
            "duration_hours": dur_hours,
            "depth_ppm": depth_pct * 10_000.0 if depth_pct is not None else None,
            "planet_radius_rearth": _float_or_none(row.get("pl_rade")),
            "system_planet_count": _int_or_none(row.get("sy_pnum")),
        }
        _progress(index, total, start)

    confirmed = sorted(
        (v for v in by_epic.values() if v["label"] == 1), key=lambda v: v["epic_id"]
    )
    false_positive = sorted(
        (v for v in by_epic.values() if v["label"] == 0), key=lambda v: v["epic_id"]
    )

    n_fp = len(false_positive)
    n_pos_target = max(0, min(len(confirmed), sample_size - n_fp))
    rng = random.Random(seed)
    selected_confirmed = rng.sample(confirmed, n_pos_target) if n_pos_target else []
    selected = false_positive + selected_confirmed

    rows: list[CalibrationManifestRow] = []
    for entry in sorted(selected, key=lambda v: v["epic_id"]):
        epic_id = entry["epic_id"]
        rows.append(
            CalibrationManifestRow(
                manifest_version=MANIFEST_VERSION,
                source="nasa_exoplanet_archive_k2pandc",
                source_table="k2pandc",
                mission="K2",
                epic_id=epic_id,
                group_key=f"k2:epic:{epic_id}",
                label=entry["label"],
                label_name=entry["label_name"],
                period_days=entry["period_days"],
                epoch_bjd=entry["epoch_bjd"],
                duration_hours=entry["duration_hours"],
                depth_ppm=entry["depth_ppm"],
                planet_radius_rearth=entry["planet_radius_rearth"],
                system_planet_count=entry["system_planet_count"],
                lightcurve_search={
                    "target": f"EPIC {epic_id}",
                    "mission": "K2",
                },
            )
        )

    leakage_errors: list[str] = []
    seen_groups: set[str] = set()
    for out_row in rows:
        if out_row.group_key in seen_groups:
            leakage_errors.append(f"duplicate group_key {out_row.group_key} in manifest")
        seen_groups.add(out_row.group_key)

    label_counts = dict(sorted(Counter(str(r.label) for r in rows).items()))
    flag = "OK"
    if not rows or len(confirmed) + len(false_positive) < 20:
        flag = "INSUFFICIENT"
    elif leakage_errors:
        flag = "LEAKAGE_FAIL"

    summary = CalibrationManifestSummary(
        created_at_utc=created,
        manifest_version=MANIFEST_VERSION,
        row_count=len(rows),
        label_counts=label_counts,
        seed=seed,
        sample_size_requested=sample_size,
        n_available_confirmed=len(confirmed),
        n_available_false_positive=len(false_positive),
        n_duplicate_epic_source_rows=n_duplicate_source_rows,
        leakage_errors=tuple(leakage_errors),
        flag=flag,
        next_action=(
            "Build native K2 light-curve snippets for these EPIC targets "
            "(fetch_t1_2_k2_calibration_snippets.py), then score with "
            "score_t1_2_k2_calibration.py and feed the result into "
            "calibrate_stacking_weights.py."
            if flag == "OK"
            else "Stop and fix schema/label coverage before any snippet download."
        ),
    )
    return rows, summary


# ---------------------------------------------------------------------------
# Formatting and I/O
# ---------------------------------------------------------------------------


def format_summary(summary: CalibrationManifestSummary) -> str:
    """Render a concise Markdown summary."""
    lines = [
        "# T1-2 K2 Calibration Manifest",
        "",
        f"**Flag**: {summary.flag}",
        f"**Created at UTC**: {summary.created_at_utc}",
        f"**Rows**: {summary.row_count}",
        f"**Label counts**: {summary.label_counts}",
        f"**Seed**: {summary.seed}",
        f"**Sample size requested**: {summary.sample_size_requested}",
        f"**Available CONFIRMED**: {summary.n_available_confirmed}",
        f"**Available FALSE POSITIVE**: {summary.n_available_false_positive}",
        f"**Duplicate EPIC source rows collapsed**: {summary.n_duplicate_epic_source_rows}",
        "",
        "## Next Action",
        summary.next_action,
    ]
    if summary.leakage_errors:
        lines += ["", "## Leakage Errors", *[f"- {e}" for e in summary.leakage_errors]]
    return "\n".join(lines) + "\n"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_manifest_outputs(
    rows: Sequence[CalibrationManifestRow],
    summary: CalibrationManifestSummary,
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a leakage-safe T1-2 held-out calibration manifest from "
            "the NASA Exoplanet Archive K2 planets-and-candidates table "
            "(k2pandc). Does not download light curves."
        )
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=600)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("metadata/t1_2_k2_calibration_manifest.jsonl"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("metadata/t1_2_k2_calibration_manifest_summary.json"),
    )
    parser.add_argument(
        "--report", type=Path, default=Path("reports/t1-2_k2_calibration_manifest.md")
    )
    args = parser.parse_args(argv)

    rows, summary = build_k2_calibration_manifest(
        seed=args.seed, sample_size=args.sample_size
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
