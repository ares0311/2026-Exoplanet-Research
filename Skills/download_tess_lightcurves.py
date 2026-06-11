"""Download TESS light curves from MAST and build a JSONL snippet corpus.

Reads a TOI CSV produced by Skills/fetch_tess_toi.py, downloads each TIC ID
from MAST via lightkurve, phase-folds and normalises the light curve, and
appends the result as one JSON object per line to an output JSONL file.

Resume safety
-------------
A checkpoint JSON (same schema as lc_snippet_batch_builder.py) tracks
completed and failed TIC IDs.  An interrupted run can be restarted with the
same command and will skip already-processed targets automatically.

Concurrency
-----------
MAST downloads are I/O-bound.  ThreadPoolExecutor is used for parallel HTTP
requests.  File writes and checkpoint saves happen in the main thread so no
lock is needed on disk operations.  Use --workers to tune (default 4).

DECISION-015 compliance
-----------------------
Rows with epoch_bjd=0.0 or missing epoch_bjd are rejected before any
download attempt.  If all rows are rejected, re-run fetch_tess_toi.py.

Usage
-----
    python Skills/download_tess_lightcurves.py \\
        --toi-csv data/tess_toi.csv \\
        --output data/tess_snippets.jsonl \\
        [--checkpoint checkpoints/download_tess.json] \\
        [--workers 4] \\
        [--n-bins 201] \\
        [--no-resume] \\
        [--limit N]
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Checkpoint helpers (compatible schema with lc_snippet_batch_builder.py)
# ---------------------------------------------------------------------------


def _load_checkpoint(path: Path) -> dict[str, set[str]]:
    """Load checkpoint; returns empty sets if file absent or corrupt."""
    with contextlib.suppress(OSError, json.JSONDecodeError):
        data = json.loads(path.read_text())
        return {
            "completed": {str(t) for t in data.get("completed_tic_ids", [])},
            "failed": {str(t) for t in data.get("failed_tic_ids", [])},
        }
    return {"completed": set(), "failed": set()}


def _save_checkpoint(path: Path, completed: set[str], failed: set[str]) -> None:
    """Atomically write checkpoint JSON."""
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "completed_tic_ids": sorted(completed),
        "failed_tic_ids": sorted(failed),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp, str(path))
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


# ---------------------------------------------------------------------------
# TOI CSV loading
# ---------------------------------------------------------------------------


def load_toi_csv(csv_path: Path) -> list[dict]:
    """Load TIC IDs, ephemerides, and labels from a TOI CSV.

    Expected columns (output of Skills/fetch_tess_toi.py):
        tic_id, period_days, epoch_bjd, tfopwg_disposition

    DECISION-015: rows with epoch_bjd=0.0 or missing are silently rejected
    and counted.  A warning is emitted if any rows are rejected.

    Returns:
        List of dicts with keys: tic_id (int), period_days (float),
        epoch_bjd (float), label (0|1), source (str).
    """
    import csv

    rows: list[dict] = []
    rejected_epoch = 0

    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                tic_id = int(float(row.get("tic_id") or 0))
                period = float(row.get("period_days") or 0)
                raw_epoch = (row.get("epoch_bjd") or "").strip()
                epoch = float(raw_epoch) if raw_epoch else 0.0
                disp = str(row.get("tfopwg_disposition", "")).strip().upper()
            except (ValueError, TypeError):
                continue

            if tic_id <= 0 or period <= 0:
                continue

            if epoch <= 0:
                rejected_epoch += 1
                continue

            rows.append({
                "tic_id": tic_id,
                "period_days": period,
                "epoch_bjd": epoch,
                "label": 1 if disp == "CP" else 0,
                "source": "tess",
            })

    if rejected_epoch:
        logger.warning(
            "Rejected %d row(s) with missing/zero epoch_bjd (DECISION-015). "
            "Re-run Skills/fetch_tess_toi.py to refresh epoch data.",
            rejected_epoch,
        )

    return rows


# ---------------------------------------------------------------------------
# Per-TIC download worker (injectable for tests)
# ---------------------------------------------------------------------------


def _download_one(row: dict, *, n_bins: int) -> dict | None:
    """Download one TIC ID from MAST and return a snippet dict or None.

    The light curve is stitched across all available TESS sectors, phase-folded
    using epoch_bjd, and normalised by out-of-transit median flux.

    Time axis: lc.time.jd (standard Julian Date, sufficient precision for
    phase-folding; difference from full BJD is < 8 min = negligible vs transit
    durations of hours).

    Returns None on any download or extraction failure.
    """
    try:
        import lightkurve as lk  # noqa: PLC0415
    except ImportError:
        logger.error("lightkurve not installed; run: pip install lightkurve")
        return None

    from Skills.labelled_lc_collector import extract_snippet  # noqa: PLC0415

    tic_id = row["tic_id"]
    period = row["period_days"]
    epoch = row["epoch_bjd"]

    try:
        # Prefer 2-min cadence; fall back to any available
        results = lk.search_lightcurve(
            f"TIC {tic_id}", mission="TESS", exptime="short"
        )
        if len(results) == 0:
            results = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS")
        if len(results) == 0:
            return None

        lc_col = results.download_all(quality_bitmask="default")
        if lc_col is None or len(lc_col) == 0:
            return None

        lc = lc_col.stitch().remove_nans().remove_outliers()
        time = list(map(float, lc.time.jd))
        flux = list(map(float, lc.flux.value))

        snippet = extract_snippet(
            time, flux, period, epoch,
            n_bins=n_bins,
            label=row["label"],
            tic_id=tic_id,
            source=row["source"],
        )
        if snippet is None:
            return None

        return {
            "tic_id": tic_id,
            "label": row["label"],
            "period_days": period,
            "epoch_bjd": epoch,
            "phase": list(snippet.phase),
            "flux": list(snippet.flux),
            "source": row["source"],
            "normalization": "local_median_mad",
        }

    except Exception as exc:
        logger.debug("TIC %s: %s", tic_id, exc)
        return None


# ---------------------------------------------------------------------------
# Main download function
# ---------------------------------------------------------------------------


def download_tess_lightcurves(
    toi_csv: Path,
    output_path: Path,
    *,
    checkpoint_path: Path,
    workers: int = 4,
    n_bins: int = 201,
    resume: bool = True,
    limit: int | None = None,
    _download_fn: object = None,
) -> dict:
    """Download TESS light curves and build a JSONL snippet corpus.

    Args:
        toi_csv:         Path to TOI CSV from Skills/fetch_tess_toi.py.
        output_path:     Destination JSONL file (appended to when resuming).
        checkpoint_path: Checkpoint JSON tracking completed/failed TIC IDs.
        workers:         Concurrent download threads (I/O-bound; default 4).
        n_bins:          Phase bins per snippet (must match CNN input size).
        resume:          If False, ignore checkpoint and overwrite output.
        limit:           Process at most this many TIC IDs (testing/dry runs).
        _download_fn:    Injection point for tests; replaces _download_one.

    Returns:
        Dict with: flag, n_attempted, n_succeeded, n_failed, n_skipped,
        output_path, checkpoint_path.
    """
    rows = load_toi_csv(toi_csv)
    if not rows:
        print(
            "No valid rows found (all rejected due to missing epoch_bjd).\n"
            "Run: python Skills/fetch_tess_toi.py --output data/tess_toi.csv"
        )
        return {
            "flag": "NO_DATA",
            "n_attempted": 0,
            "n_succeeded": 0,
            "n_failed": 0,
            "n_skipped": 0,
        }

    checkpoint = (
        _load_checkpoint(checkpoint_path)
        if resume
        else {"completed": set(), "failed": set()}
    )
    completed: set[str] = checkpoint["completed"]
    failed: set[str] = checkpoint["failed"]

    # Deduplicate by tic_id (keep first) and filter already-done
    seen: set[int] = set()
    pending: list[dict] = []
    n_skipped = 0
    for row in rows:
        tid = row["tic_id"]
        if tid in seen:
            continue
        seen.add(tid)
        if resume and str(tid) in completed:
            n_skipped += 1
            continue
        pending.append(row)

    if limit is not None:
        pending = pending[:limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    fn = _download_fn if _download_fn is not None else (
        lambda row: _download_one(row, n_bins=n_bins)
    )

    n_attempted = 0
    n_succeeded = 0
    n_failed = 0

    print(
        f"Starting: {len(pending)} pending, {n_skipped} already done, "
        f"{workers} worker(s) → {output_path}"
    )

    open_mode = "a" if resume else "w"

    # Use explicit pool management so KeyboardInterrupt can cancel pending
    # futures immediately instead of blocking in ThreadPoolExecutor.__exit__
    # (which calls shutdown(wait=True) and holds until all threads finish).
    pool = ThreadPoolExecutor(max_workers=workers)
    interrupted = False
    try:
        with open(output_path, open_mode) as out_fh:
            futures = {pool.submit(fn, row): row for row in pending}
            try:
                for i, future in enumerate(as_completed(futures), 1):
                    row = futures[future]
                    tic_key = str(row["tic_id"])
                    n_attempted += 1

                    try:
                        result = future.result()
                    except Exception as exc:
                        logger.warning("TIC %s raised: %s", row["tic_id"], exc)
                        result = None

                    if result is not None:
                        out_fh.write(json.dumps(result) + "\n")
                        out_fh.flush()
                        completed.add(tic_key)
                        failed.discard(tic_key)
                        n_succeeded += 1
                    else:
                        failed.add(tic_key)
                        n_failed += 1

                    # Save checkpoint after every record (main thread only)
                    _save_checkpoint(checkpoint_path, completed, failed)

                    if i % 10 == 0 or i == len(pending):
                        print(
                            f"  [{i}/{len(pending)}] "
                            f"+{n_succeeded} ok  -{n_failed} failed"
                        )

            except KeyboardInterrupt:
                interrupted = True
                pool.shutdown(wait=False, cancel_futures=True)
                _save_checkpoint(checkpoint_path, completed, failed)
                print(
                    f"\nInterrupted at {n_attempted}/{len(pending)} targets. "
                    f"Checkpoint saved ({n_succeeded} done). "
                    f"Resume with the same command."
                )
    finally:
        pool.shutdown(wait=False)

    if interrupted:
        return {
            "flag": "INTERRUPTED",
            "n_attempted": n_attempted,
            "n_succeeded": n_succeeded,
            "n_failed": n_failed,
            "n_skipped": n_skipped,
            "output_path": str(output_path),
            "checkpoint_path": str(checkpoint_path),
        }

    print(
        f"\nDone: {n_succeeded} succeeded, {n_failed} failed, "
        f"{n_skipped} skipped (already in checkpoint)."
    )
    return {
        "flag": "OK",
        "n_attempted": n_attempted,
        "n_succeeded": n_succeeded,
        "n_failed": n_failed,
        "n_skipped": n_skipped,
        "output_path": str(output_path),
        "checkpoint_path": str(checkpoint_path),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    p = argparse.ArgumentParser(
        prog="download_tess_lightcurves",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--toi-csv", default="data/tess_toi.csv", metavar="PATH",
        help="TOI CSV from fetch_tess_toi.py (default: data/tess_toi.csv)",
    )
    p.add_argument(
        "--output", default="data/tess_snippets.jsonl", metavar="PATH",
        help="Output JSONL file (default: data/tess_snippets.jsonl)",
    )
    p.add_argument(
        "--checkpoint", default="checkpoints/download_tess.json", metavar="PATH",
        help="Checkpoint JSON (default: checkpoints/download_tess.json)",
    )
    p.add_argument(
        "--workers", type=int, default=4, metavar="N",
        help="Concurrent download threads (default: 4)",
    )
    p.add_argument(
        "--n-bins", type=int, default=201, metavar="N",
        help="Phase bins per snippet (default: 201)",
    )
    p.add_argument(
        "--no-resume", action="store_true",
        help="Ignore checkpoint and overwrite output file",
    )
    p.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Process at most N TIC IDs (for testing / dry runs)",
    )
    args = p.parse_args(argv)

    result = download_tess_lightcurves(
        Path(args.toi_csv),
        Path(args.output),
        checkpoint_path=Path(args.checkpoint),
        workers=args.workers,
        n_bins=args.n_bins,
        resume=not args.no_resume,
        limit=args.limit,
    )
    for key, val in result.items():
        print(f"  {key}: {val}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
