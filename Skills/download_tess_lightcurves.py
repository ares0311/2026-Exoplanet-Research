"""Download TESS light curves and extract phase-folded CNN training snippets.

Reads data/tess_toi.csv (produced by fetch_tess_toi.py), fetches PDCSAP light
curves from MAST via lightkurve, phase-folds each one into a 201-point array,
and saves compact snippet records to a JSONL checkpoint file.

No FITS files are kept — raw light curves are discarded after extraction.
Lightkurve's own cache (~/.lightkurve/cache/) holds temporary FITS files.
Run `python -c "import lightkurve; lightkurve.conf.cache_dir"` to see it.

Usage
-----
    # Always use caffeinate -i on macOS to prevent sleep from killing the run:
    caffeinate -i python Skills/download_tess_lightcurves.py \\
        --toi-csv  data/tess_toi.csv \\
        --output   data/tess_snippets.jsonl \\
        [--resume]             # skip TIC IDs already in output
        [--max-targets N]      # stop after N targets (default: all)
        [--n-bins 201]         # phase bins per snippet (default: 201)
        [--sleep 0.5]          # seconds between MAST requests (default: 0.5)

    # To run with lid closed add -dims:
    caffeinate -dims python Skills/download_tess_lightcurves.py --resume \\
        --toi-csv data/tess_toi.csv --output data/tess_snippets.jsonl

Output JSONL format (one JSON object per line)
----------------------------------------------
Success:
    {"tic_id": 150428135, "label": 1, "disposition": "CP",
     "period_days": 37.4, "epoch_bjd": 2458325.0,
     "phase": [...201 floats...], "flux": [...201 floats...],
     "n_points": 8732, "status": "ok"}

Failure:
    {"tic_id": 999, "status": "error", "reason": "No TESS LC found"}

Progress report
---------------
After the run, check progress with:
    python -c "
import json, collections
rows = [json.loads(l) for l in open('data/tess_snippets.jsonl')]
st = collections.Counter(r['status'] for r in rows)
print(st)
"
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

_DISPOSITION_LABEL: dict[str, int] = {
    "CP": 1, "KP": 1,   # positive class — confirmed/known planets
    "FP": 0, "FA": 0,   # negative class — false positives / false alarms
}


# ---------------------------------------------------------------------------
# Light-curve fetcher
# ---------------------------------------------------------------------------


def _fetch_lc_lightkurve(tic_id: int) -> tuple[list[float], list[float]]:
    """Fetch TESS PDCSAP light curve via lightkurve; returns (time_jd, flux)."""
    import lightkurve as lk

    # Prefer 2-min SPOC cadence; fall back to QLP
    results = lk.search_lightcurve(
        f"TIC {tic_id}", mission="TESS", author="SPOC", exptime=120
    )
    if len(results) == 0:
        results = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", author="QLP")
    if len(results) == 0:
        raise ValueError(f"No TESS light curve found for TIC {tic_id}")

    lc = results.download_all().stitch()

    # Remove NaN cadences before returning
    import numpy as np
    mask = np.isfinite(lc.flux.value)
    time_jd = lc.time.jd[mask].tolist()
    flux = lc.flux.value[mask].tolist()

    if len(time_jd) == 0:
        raise ValueError(f"All cadences are NaN for TIC {tic_id}")

    return time_jd, flux


# ---------------------------------------------------------------------------
# CSV reader
# ---------------------------------------------------------------------------


def _load_toi_rows(csv_path: Path) -> list[dict]:
    """Read TOI CSV and return rows filtered to labelled dispositions."""
    rows: list[dict] = []
    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        required = {
            "tic_id",
            "tfopwg_disposition",
            "period_days",
            "epoch_bjd",
        }
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(
                "TOI CSV is missing required CNN columns: " + ", ".join(missing)
            )
        for row in reader:
            disp = row.get("tfopwg_disposition", "").strip()
            if disp not in _DISPOSITION_LABEL:
                continue
            try:
                tic_id = int(float(row["tic_id"]))
                period = float(row["period_days"])
                epoch = float(row["epoch_bjd"])
            except (ValueError, KeyError):
                continue
            if (
                period <= 0
                or not math.isfinite(period)
                or epoch < 2_000_000.0
                or not math.isfinite(epoch)
            ):
                continue
            rows.append({
                "tic_id": tic_id,
                "disposition": disp,
                "label": _DISPOSITION_LABEL[disp],
                "period_days": period,
                "epoch_bjd": epoch,
            })
    return rows


def audit_snippet_corpus(path: Path) -> dict[str, int | bool]:
    """Check that a downloaded corpus has usable, BJD-centered snippets."""
    n_rows = n_ok = n_error = invalid_epoch = invalid_bins = 0
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            n_rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                n_error += 1
                continue
            if row.get("status") != "ok":
                n_error += 1
                continue
            n_ok += 1
            epoch = row.get("epoch_bjd")
            if (
                not isinstance(epoch, (int, float))
                or isinstance(epoch, bool)
                or not math.isfinite(float(epoch))
                or float(epoch) < 2_000_000.0
            ):
                invalid_epoch += 1
            phase = row.get("phase")
            flux = row.get("flux")
            if (
                not isinstance(phase, list)
                or not isinstance(flux, list)
                or len(phase) != 201
                or len(flux) != 201
            ):
                invalid_bins += 1
    return {
        "n_rows": n_rows,
        "n_ok": n_ok,
        "n_error": n_error,
        "invalid_epoch": invalid_epoch,
        "invalid_bins": invalid_bins,
        "valid": n_ok > 0 and invalid_epoch == 0 and invalid_bins == 0,
    }


# ---------------------------------------------------------------------------
# Checkpoint reader
# ---------------------------------------------------------------------------


def _load_done_tic_ids(output_path: Path) -> set[int]:
    """Return set of TIC IDs already written to *output_path*."""
    done: set[int] = set()
    if not output_path.exists():
        return done
    with output_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                done.add(int(obj["tic_id"]))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return done


# ---------------------------------------------------------------------------
# Phase-fold and bin (local copy to avoid cross-import issues)
# ---------------------------------------------------------------------------


def _phase_fold_bin(
    time: list[float],
    flux: list[float],
    period: float,
    epoch: float,
    n_bins: int,
) -> tuple[list[float], list[float]]:
    phases = [((t - epoch) % period) / period for t in time]
    phases = [p - 1.0 if p >= 0.5 else p for p in phases]

    bin_flux: list[list[float]] = [[] for _ in range(n_bins)]
    for ph, f in zip(phases, flux, strict=False):
        b = int((ph + 0.5) * n_bins)
        b = max(0, min(n_bins - 1, b))
        bin_flux[b].append(f)

    bin_centers = [(-0.5 + (i + 0.5) / n_bins) for i in range(n_bins)]
    bin_means = [
        sum(vals) / len(vals) if vals else 1.0
        for vals in bin_flux
    ]
    return bin_centers, bin_means


def _extract_snippet(
    time_jd: list[float],
    flux: list[float],
    period: float,
    epoch: float,
    n_bins: int,
) -> tuple[list[float], list[float]] | None:
    """Phase-fold, normalise, and bin. Returns (phase, flux_binned) or None."""
    if len(time_jd) < n_bins // 2 or period <= 0:
        return None

    # Normalise by median (OOT) flux
    sorted_f = sorted(flux)
    median_f = sorted_f[len(sorted_f) // 2]
    if median_f == 0.0:
        median_f = 1.0
    flux_norm = [f / median_f for f in flux]

    phase_centers, bin_means = _phase_fold_bin(time_jd, flux_norm, period, epoch, n_bins)
    return phase_centers, bin_means


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def download_and_extract(
    toi_csv: Path,
    output_path: Path,
    *,
    resume: bool = False,
    max_targets: int | None = None,
    n_bins: int = 201,
    lc_fetch_fn: Callable[[int], tuple[list[float], list[float]]] | None = None,
    sleep_between: float = 0.5,
) -> dict[str, int]:
    """Download TESS light curves and extract phase-folded CNN snippets.

    Args:
        toi_csv: Path to data/tess_toi.csv.
        output_path: JSONL output path (appended to when resume=True).
        resume: Skip TIC IDs already present in output_path.
        max_targets: Cap total targets processed (default: all).
        n_bins: Phase bins per snippet — 201 matches Shallue & Vanderburg 2018.
        lc_fetch_fn: Injectable light-curve fetcher for tests. Defaults to
            lightkurve + MAST. Signature: (tic_id) -> (time_jd, flux).
        sleep_between: Seconds to sleep between MAST requests. Set to 0 in tests.

    Returns:
        Dict with keys n_ok, n_error, n_skipped, n_total.
    """
    fetch = lc_fetch_fn if lc_fetch_fn is not None else _fetch_lc_lightkurve

    rows = _load_toi_rows(toi_csv)
    if not rows:
        raise ValueError("TOI CSV has no labelled rows with valid BJD ephemerides")
    done_ids = _load_done_tic_ids(output_path) if resume else set()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    targets = rows if max_targets is None else rows[:max_targets]
    n_ok = n_error = n_skipped = 0
    mode = "a" if resume else "w"

    with output_path.open(mode) as fh:
        for i, row in enumerate(targets):
            tic_id = row["tic_id"]

            if tic_id in done_ids:
                n_skipped += 1
                continue

            print(
                f"[{i + 1}/{len(targets)}] TIC {tic_id} ({row['disposition']}) … ",
                end="",
                flush=True,
            )

            try:
                time_jd, flux = fetch(tic_id)
                result = _extract_snippet(
                    time_jd, flux, row["period_days"], row["epoch_bjd"], n_bins
                )
                if result is None:
                    raise ValueError("Insufficient data for phase-fold extraction")

                phase_centers, flux_bins = result
                record: dict = {
                    "tic_id": tic_id,
                    "label": row["label"],
                    "disposition": row["disposition"],
                    "period_days": row["period_days"],
                    "epoch_bjd": row["epoch_bjd"],
                    "phase": [round(p, 6) for p in phase_centers],
                    "flux": [round(f, 8) for f in flux_bins],
                    "n_points": len(time_jd),
                    "status": "ok",
                }
                fh.write(json.dumps(record) + "\n")
                fh.flush()
                n_ok += 1
                print(f"ok  ({len(time_jd):,} cadences → {n_bins} bins)")

            except Exception as exc:
                record = {
                    "tic_id": tic_id,
                    "status": "error",
                    "reason": str(exc)[:300],
                }
                fh.write(json.dumps(record) + "\n")
                fh.flush()
                n_error += 1
                print(f"ERROR — {exc}")

            if sleep_between > 0 and lc_fetch_fn is None:
                time.sleep(sleep_between)

    return {
        "n_ok": n_ok,
        "n_error": n_error,
        "n_skipped": n_skipped,
        "n_total": len(targets),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="download_tess_lightcurves",
        description="Download TESS LCs and extract phase-folded CNN snippets.",
    )
    parser.add_argument(
        "--toi-csv", default="data/tess_toi.csv",
        metavar="PATH",
        help="TOI label CSV from fetch_tess_toi.py (default: data/tess_toi.csv)",
    )
    parser.add_argument(
        "--output", default="data/tess_snippets.jsonl",
        metavar="PATH",
        help="Output JSONL path (default: data/tess_snippets.jsonl)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip TIC IDs already written to --output",
    )
    parser.add_argument(
        "--max-targets", type=int, default=None,
        metavar="N",
        help="Stop after N targets (default: all 2668)",
    )
    parser.add_argument(
        "--n-bins", type=int, default=201,
        help="Phase bins per snippet (default: 201)",
    )
    parser.add_argument(
        "--sleep", type=float, default=0.5,
        metavar="SEC",
        help="Seconds to sleep between MAST requests (default: 0.5)",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Audit the existing --output corpus without downloading",
    )
    args = parser.parse_args(argv)

    output_path = Path(args.output)
    if args.audit_only:
        if not output_path.is_file():
            print(f"ERROR: {output_path} not found.")
            return 1
        audit = audit_snippet_corpus(output_path)
        print(json.dumps(audit, indent=2, sort_keys=True))
        return 0 if audit["valid"] else 1

    toi_csv = Path(args.toi_csv)
    if not toi_csv.exists():
        print(
            f"ERROR: {toi_csv} not found.\n"
            "Run first:  python Skills/fetch_tess_toi.py --output data/tess_toi.csv"
        )
        return 1

    try:
        result = download_and_extract(
            toi_csv,
            output_path,
            resume=args.resume,
            max_targets=args.max_targets,
            n_bins=args.n_bins,
            sleep_between=args.sleep,
        )
    except ValueError as error:
        print(f"ERROR: {error}")
        return 1
    print(
        f"\nDone: {result['n_ok']} ok, {result['n_error']} errors, "
        f"{result['n_skipped']} skipped of {result['n_total']} targets."
    )
    print(f"Snippets saved to: {args.output}")
    if result["n_error"] > 0:
        print("Re-run with --resume to retry failed targets after fixing any issues.")
    return 0 if result["n_error"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(_cli())
