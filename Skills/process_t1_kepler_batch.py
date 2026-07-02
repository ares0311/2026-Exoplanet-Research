"""Bounded, resumable Kepler-first processing batch for T1-1.

Consumes the leakage-safe manifest committed by
``Skills/build_t1_training_manifest.py``
(``metadata/t1_1_kepler_training_manifest.jsonl``). For each unique KIC
target, fetches its Kepler light curve once, then phase-folds every KOI row
sharing that target at its own period/epoch -- reusing the proven phase-fold
and normalisation math from ``Skills/fetch_kepler_lc_snippets.py`` rather than
reimplementing it. Processed snippets are written to
``data/processed/t1_1_kepler_snippets/kepler_snippets.jsonl``.

Progress and resume state are tracked in a SQLite database
(``logs/t1_1_kepler_processing.sqlite3`` by default) so an interrupted run
restarts without reprocessing completed targets. Raw downloaded FITS files
are scoped to a dedicated directory and deleted after every target finishes
(success or failure), so local raw storage never exceeds roughly one target's
data at a time. This satisfies the dataset handoff doc's storage cap and its
"delete raw FITS after verified processing" rule by construction, rather than
by monitoring a large threshold.

Public API
----------
ManifestRow -- TypedDict-like alias for one manifest JSONL row
load_manifest_rows(path) -> list[dict]
group_rows_by_target(rows) -> dict[int, list[dict]]
T1KeplerProcessingStore(db_path)
    .mark_active/.mark_done/.done_target_ids/.summary
process_target(target_id, rows, *, lc_fetcher, n_bins) -> TargetResult
run_batch(*, manifest_path, output_path, db_path, raw_dir, max_targets,
          lc_fetcher, ...) -> BatchSummary
format_batch_summary(summary) -> str
"""
from __future__ import annotations

import json
import math
import sqlite3
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Literal

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fetch_kepler_lc_snippets import _normalise, _phase_fold_bin  # noqa: E402

_KEPLER_BJD_OFFSET = 2454833.0  # Kepler time is BJD - 2454833

DEFAULT_MANIFEST_PATH = Path("metadata/t1_1_kepler_training_manifest.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/processed/t1_1_kepler_snippets")
DEFAULT_OUTPUT_FILENAME = "kepler_snippets.jsonl"
DEFAULT_DB_PATH = Path("logs/t1_1_kepler_processing.sqlite3")
DEFAULT_RAW_DIR = Path("data/raw/t1_1_kepler_lc")

LcFetcher = Callable[[int], "tuple[list[float], list[float]] | None"]

TargetStatus = Literal["active", "done"]


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------


def load_manifest_rows(path: Path) -> list[dict[str, Any]]:
    """Load and parse the T1-1 Kepler training manifest JSONL file."""
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def group_rows_by_target(rows: Iterable[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    """Group manifest rows by ``target_id`` (KIC) so each light curve is fetched once."""
    groups: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        target_id = int(row["target_id"])
        groups.setdefault(target_id, []).append(row)
    return dict(sorted(groups.items()))


# ---------------------------------------------------------------------------
# SQLite progress/resume store
# ---------------------------------------------------------------------------


class _ClosingConnection(sqlite3.Connection):
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        super().__exit__(exc_type, exc_value, traceback)
        self.close()
        return False


class T1KeplerProcessingStore:
    """Durable SQLite progress/resume store for the T1-1 Kepler processing batch."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30.0, factory=_ClosingConnection)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS targets (
                    target_id INTEGER PRIMARY KEY,
                    status TEXT NOT NULL,
                    n_manifest_rows INTEGER NOT NULL,
                    n_written INTEGER NOT NULL DEFAULT 0,
                    n_failed INTEGER NOT NULL DEFAULT 0,
                    flag TEXT,
                    started_at TEXT NOT NULL,
                    completed_at TEXT
                );
                """
            )

    def mark_active(self, target_id: int, n_manifest_rows: int) -> None:
        """Record that *target_id* has started processing (idempotent)."""
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO targets (target_id, status, n_manifest_rows, started_at)
                VALUES (?, 'active', ?, ?)
                ON CONFLICT(target_id) DO UPDATE SET
                    status='active', n_manifest_rows=excluded.n_manifest_rows,
                    started_at=excluded.started_at, completed_at=NULL
                """,
                (target_id, n_manifest_rows, now),
            )

    def mark_done(self, target_id: int, *, n_written: int, n_failed: int, flag: str) -> None:
        """Record that *target_id* finished processing (success or terminal failure)."""
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE targets
                SET status='done', n_written=?, n_failed=?, flag=?, completed_at=?
                WHERE target_id=?
                """,
                (n_written, n_failed, flag, now, target_id),
            )

    def done_target_ids(self) -> set[int]:
        """Return target IDs whose processing has completed (skip on resume)."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT target_id FROM targets WHERE status='done'"
            ).fetchall()
        return {int(row["target_id"]) for row in rows}

    def active_target_ids(self) -> set[int]:
        """Return target IDs currently marked active (interrupted mid-run, if any)."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT target_id FROM targets WHERE status='active'"
            ).fetchall()
        return {int(row["target_id"]) for row in rows}

    def summary(self) -> dict[str, Any]:
        """Return aggregate progress counts."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    COUNT(*) AS n_targets,
                    COALESCE(SUM(CASE WHEN status='done' THEN 1 ELSE 0 END), 0) AS n_done,
                    COALESCE(SUM(CASE WHEN status='active' THEN 1 ELSE 0 END), 0) AS n_active,
                    COALESCE(SUM(n_written), 0) AS n_written,
                    COALESCE(SUM(n_failed), 0) AS n_failed
                FROM targets
                """
            ).fetchone()
        return {
            "n_targets": int(row["n_targets"]),
            "n_done": int(row["n_done"]),
            "n_active": int(row["n_active"]),
            "n_written": int(row["n_written"]),
            "n_failed": int(row["n_failed"]),
        }


# ---------------------------------------------------------------------------
# Per-target processing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TargetResult:
    """Outcome of processing every manifest row for one KIC target."""

    target_id: int
    records: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    n_failed: int = 0
    flag: str = "OK"  # "OK" | "NO_DATA" | "NO_LIGHTKURVE" | "ERROR:<message>"


def _epoch_bjd(row: dict[str, Any]) -> float:
    return float(row["epoch_bkjd"]) + _KEPLER_BJD_OFFSET


def _snippet_record(row: dict[str, Any], flux: tuple[float, ...], n_bins: int) -> dict[str, Any]:
    return {
        "target_id": int(row["target_id"]),
        "target_name": row.get("target_name"),
        "source_row_id": row.get("source_row_id"),
        "group_key": row.get("group_key"),
        "split": row.get("split"),
        "label": int(row["label"]),
        "label_name": row.get("label_name"),
        "period_days": float(row["period_days"]),
        "epoch_bjd": _epoch_bjd(row),
        "duration_hours": row.get("duration_hours"),
        "n_bins": n_bins,
        "flux": list(flux),
        "manifest_version": row.get("manifest_version"),
        "source": row.get("source"),
    }


def process_target(
    target_id: int,
    rows: list[dict[str, Any]],
    *,
    lc_fetcher: LcFetcher,
    n_bins: int = 201,
) -> TargetResult:
    """Fetch one target's light curve once and phase-fold every manifest row for it."""
    try:
        fetched = lc_fetcher(target_id)
    except Exception as exc:  # noqa: BLE001
        return TargetResult(target_id=target_id, n_failed=len(rows), flag=f"ERROR:{exc}")

    if fetched is None:
        try:
            import lightkurve  # noqa: F401, PLC0415
        except ImportError:
            return TargetResult(target_id=target_id, n_failed=len(rows), flag="NO_LIGHTKURVE")
        return TargetResult(target_id=target_id, n_failed=len(rows), flag="NO_DATA")

    time_bjd, flux = fetched
    finite_pairs = [
        (t, f)
        for t, f in zip(time_bjd, flux, strict=False)
        if math.isfinite(t) and math.isfinite(f)
    ]
    records: list[dict[str, Any]] = []
    n_failed = 0
    for row in rows:
        period_days = float(row["period_days"])
        epoch_bjd = _epoch_bjd(row)
        if period_days <= 0.0 or len(finite_pairs) < n_bins:
            n_failed += 1
            continue
        curve_time = [pair[0] for pair in finite_pairs]
        curve_flux = [pair[1] for pair in finite_pairs]
        bins = _phase_fold_bin(curve_time, curve_flux, period_days, epoch_bjd, n_bins)
        normalised = _normalise(bins)
        if len(normalised) != n_bins or any(not math.isfinite(value) for value in normalised):
            n_failed += 1
            continue
        records.append(_snippet_record(row, tuple(normalised), n_bins))

    flag = "OK" if records else "NONFINITE"
    return TargetResult(target_id=target_id, records=tuple(records), n_failed=n_failed, flag=flag)


# ---------------------------------------------------------------------------
# Default light-curve fetcher: download to a scoped dir, delete when done
# ---------------------------------------------------------------------------


def _clear_directory(path: Path) -> None:
    """Delete every file/subdirectory under *path*, leaving the directory itself."""
    import shutil

    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def _directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def make_default_lc_fetcher(raw_dir: Path) -> LcFetcher:
    """Build the default Kepler light-curve fetcher, scoped to *raw_dir*.

    Downloads via Lightkurve into *raw_dir*, extracts (time_bjd, flux), then
    deletes everything under *raw_dir* before returning -- regardless of
    success or failure -- so raw storage never accumulates across targets.
    """

    def _fetch(target_id: int) -> tuple[list[float], list[float]] | None:
        import contextlib

        import lightkurve as lk  # noqa: PLC0415

        raw_dir.mkdir(parents=True, exist_ok=True)
        try:
            search = lk.search_lightcurve(
                f"KIC {target_id}", mission="Kepler", exptime=1800, author="Kepler"
            )
            if len(search) == 0:
                return None
            collection = search.download_all(download_dir=str(raw_dir))
            if collection is None or len(collection) == 0:
                return None
            lc = collection.stitch()
            with contextlib.suppress(Exception):
                lc = lc.normalize()
            time_bjd = [float(t) + _KEPLER_BJD_OFFSET for t in lc.time.value]
            flux = [float(f) for f in lc.flux.value]
            return time_bjd, flux
        finally:
            _clear_directory(raw_dir)

    return _fetch


# ---------------------------------------------------------------------------
# Batch orchestration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BatchSummary:
    """Result of one ``run_batch`` invocation."""

    n_targets_total: int
    n_targets_processed_this_run: int
    n_targets_skipped_done: int
    n_snippets_written: int
    n_rows_failed: int
    elapsed_seconds: float
    output_path: str
    db_path: str


def _write_records(fh: Any, records: Iterable[dict[str, Any]]) -> int:
    n = 0
    for record in records:
        fh.write(json.dumps(record) + "\n")
        n += 1
    if n:
        fh.flush()
    return n


def run_batch(
    *,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    output_path: Path = DEFAULT_OUTPUT_DIR / DEFAULT_OUTPUT_FILENAME,
    db_path: Path = DEFAULT_DB_PATH,
    raw_dir: Path = DEFAULT_RAW_DIR,
    max_targets: int | None = 25,
    n_bins: int = 201,
    lc_fetcher: LcFetcher | None = None,
    progress_fn: Callable[[str], None] | None = None,
) -> BatchSummary:
    """Run one bounded pass of the T1-1 Kepler processing batch.

    Args:
        manifest_path: Leakage-safe manifest JSONL from ``build_t1_training_manifest.py``.
        output_path: Destination JSONL for processed snippets (appended; never
            duplicates a target already marked ``done`` in the SQLite store).
        db_path: SQLite progress/resume database.
        raw_dir: Scratch directory for raw FITS downloads; wiped after every target.
        max_targets: Maximum number of *not-yet-done* targets to process in this
            call. ``None`` means all remaining targets -- use with care given the
            dataset handoff doc's storage/runtime bounding rules.
        n_bins: Phase-fold bin count per snippet.
        lc_fetcher: Injectable light-curve fetcher (for tests); defaults to a
            real Lightkurve fetch scoped to *raw_dir*.
        progress_fn: Optional callable invoked with a one-line status message
            before/after each target, so a long batch never looks hung.
    """
    manifest_path = Path(manifest_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _progress = progress_fn if progress_fn is not None else (lambda _msg: None)

    store = T1KeplerProcessingStore(db_path)
    _fetch = lc_fetcher if lc_fetcher is not None else make_default_lc_fetcher(Path(raw_dir))

    rows = load_manifest_rows(manifest_path)
    groups = group_rows_by_target(rows)
    n_targets_total = len(groups)

    done_ids = store.done_target_ids()
    pending_target_ids = [tid for tid in groups if tid not in done_ids]
    if max_targets is not None:
        pending_target_ids = pending_target_ids[:max_targets]

    _progress(
        f"T1-1 Kepler batch: {n_targets_total} total targets, "
        f"{len(done_ids)} already done, {len(pending_target_ids)} to process this run"
    )

    start = time.monotonic()
    n_written = 0
    n_failed = 0
    with output_path.open("a", encoding="utf-8") as fh:
        for index, target_id in enumerate(pending_target_ids, 1):
            target_rows = groups[target_id]
            store.mark_active(target_id, len(target_rows))
            elapsed = time.monotonic() - start
            _progress(
                f"[{index}/{len(pending_target_ids)}] KIC {target_id} "
                f"({len(target_rows)} manifest rows)  elapsed={elapsed:.0f}s"
            )
            result = process_target(target_id, target_rows, lc_fetcher=_fetch, n_bins=n_bins)
            wrote = _write_records(fh, result.records)
            store.mark_done(target_id, n_written=wrote, n_failed=result.n_failed, flag=result.flag)
            n_written += wrote
            n_failed += result.n_failed
            _progress(
                f"  -> KIC {target_id} flag={result.flag} "
                f"written={wrote} failed={result.n_failed}  "
                f"total_written={n_written} total_failed={n_failed}"
            )

    elapsed_total = time.monotonic() - start
    _progress(
        f"Done in {elapsed_total:.0f}s: {len(pending_target_ids)} targets processed, "
        f"{n_written} snippets written, {n_failed} rows failed"
    )

    return BatchSummary(
        n_targets_total=n_targets_total,
        n_targets_processed_this_run=len(pending_target_ids),
        n_targets_skipped_done=len(done_ids),
        n_snippets_written=n_written,
        n_rows_failed=n_failed,
        elapsed_seconds=elapsed_total,
        output_path=str(output_path),
        db_path=str(db_path),
    )


def format_batch_summary(summary: BatchSummary) -> str:
    """Render a :class:`BatchSummary` as a short Markdown report."""
    return "\n".join(
        [
            "# T1-1 Kepler Processing Batch",
            "",
            f"- Total targets in manifest: {summary.n_targets_total}",
            f"- Already done before this run: {summary.n_targets_skipped_done}",
            f"- Processed this run: {summary.n_targets_processed_this_run}",
            f"- Snippets written: {summary.n_snippets_written}",
            f"- Manifest rows failed: {summary.n_rows_failed}",
            f"- Elapsed: {summary.elapsed_seconds:.0f}s",
            f"- Output: `{summary.output_path}`",
            f"- Progress DB: `{summary.db_path}`",
        ]
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="process_t1_kepler_batch",
        description=(
            "Run one bounded pass of the T1-1 Kepler-first processing batch: "
            "consumes the leakage-safe manifest, fetches each target's light "
            "curve once, phase-folds every KOI row, and writes processed "
            "snippets. Resumable via SQLite; raw FITS are deleted per target."
        ),
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR / DEFAULT_OUTPUT_FILENAME)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument(
        "--max-targets",
        type=int,
        default=25,
        help="Maximum not-yet-done targets to process this run (default: 25; bounded first batch)",
    )
    parser.add_argument("--n-bins", type=int, default=201)
    args = parser.parse_args(argv)

    if not args.manifest.exists():
        print(f"Manifest not found: {args.manifest}", file=sys.stderr)
        print(
            "Run Skills/build_t1_training_manifest.py first to create it.",
            file=sys.stderr,
        )
        return 1

    summary = run_batch(
        manifest_path=args.manifest,
        output_path=args.output,
        db_path=args.db_path,
        raw_dir=args.raw_dir,
        max_targets=args.max_targets,
        n_bins=args.n_bins,
        progress_fn=lambda msg: print(msg, flush=True),
    )
    print(format_batch_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
