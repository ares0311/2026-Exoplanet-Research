"""Download the Kepler KOI cumulative table (Thompson et al. 2018, DR25) from
NASA Exoplanet Archive and save as CSV.

Only CONFIRMED and FALSE POSITIVE dispositions are kept; CANDIDATEs are
excluded because they are noisy training labels.

Usage
-----
    python Skills/fetch_kepler_tce.py [--output data/kepler_koi.csv]

Output
------
    CSV with columns: kepoi_name, koi_disposition, koi_pdisposition,
    koi_model_snr, koi_count, koi_period, koi_duration, koi_depth,
    koi_prad, koi_dikco_msky, koi_steff, koi_slogg, koi_srad, koi_kepmag

The CLI entry point writes a structured completion record via
``Skills/run_report.py`` after each run (AGENTS.md Run Report Policy,
Rule 7) and commits/pushes only that record. A query failure propagates
uncaught (fail loudly) rather than writing a false-success report.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from Skills.run_report import RunReport, report_path_for, run_and_commit_report  # noqa: E402

# ---------------------------------------------------------------------------
# Column selection
# ---------------------------------------------------------------------------

_SELECT_COLS = ",".join([
    "kepoi_name",
    "koi_disposition",
    "koi_pdisposition",
    "koi_model_snr",
    "koi_count",
    "koi_period",
    "koi_duration",
    "koi_depth",
    "koi_prad",
    "koi_dikco_msky",
    "koi_steff",
    "koi_slogg",
    "koi_srad",
    "koi_kepmag",
])

_WHERE = "koi_disposition+like+'CONFIRMED'+or+koi_disposition+like+'FALSE+POSITIVE'"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_koi_table(
    output_path: str | Path = "data/kepler_koi.csv",
    *,
    query_fn: Callable[..., Any] | None = None,
    stats: dict[str, int] | None = None,
) -> Path:
    """Download KOI cumulative table and save to *output_path*.

    Args:
        output_path: Destination CSV path.
        query_fn: Injectable replacement for
            ``NasaExoplanetArchive.query_criteria`` (for testing); must
            return an object with a ``.to_pandas()`` method.
        stats: When given, populated with
            ``{"written", "errors", "total", "confirmed", "false_positive"}``
            in-place on return — a non-breaking side channel for callers
            (e.g. the CLI's Run Report), matching the pattern used by this
            project's other acquisition Skills.

    Returns:
        Path to the written CSV file.
    """
    if query_fn is None:
        from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

        query_fn = NasaExoplanetArchive.query_criteria

    print("Querying NASA Exoplanet Archive (KOI cumulative table) …")
    table = query_fn(
        table="cumulative",
        select=_SELECT_COLS,
        where="koi_disposition like 'CONFIRMED' or koi_disposition like 'FALSE POSITIVE'",
    )

    df = table.to_pandas()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    n_conf = int((df["koi_disposition"] == "CONFIRMED").sum())
    n_fp = int((df["koi_disposition"] == "FALSE POSITIVE").sum())
    print(f"Saved {len(df):,} KOIs → {output}")
    print(f"  Confirmed  : {n_conf:,}")
    print(f"  False pos. : {n_fp:,}")
    if stats is not None:
        stats["written"] = len(df)
        stats["errors"] = 0
        stats["total"] = len(df)
        stats["confirmed"] = n_conf
        stats["false_positive"] = n_fp
    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output",
        default="data/kepler_koi.csv",
        help="Destination CSV path (default: data/kepler_koi.csv)",
    )
    return p.parse_args(argv)


def _write_run_report(
    *,
    started_at: str,
    elapsed_seconds: float,
    stats: dict[str, int],
    output_path: Path,
    git_run_fn: Any = None,
) -> None:
    """Append and publish one fetch_kepler_tce completion report
    (AGENTS.md Rule 7)."""
    report = RunReport(
        script="fetch_kepler_tce",
        status="success",
        started_at=started_at,
        completed_at=datetime.now(UTC).isoformat(),
        elapsed_seconds=elapsed_seconds,
        items_processed=stats.get("total", 0),
        items_written=stats.get("written", 0),
        items_failed=stats.get("errors", 0),
        output_paths=(str(output_path),),
        notes=(
            f"confirmed={stats.get('confirmed', 0)} "
            f"false_positive={stats.get('false_positive', 0)}"
        ),
    )
    path = report_path_for("fetch_kepler_tce")
    kwargs: dict[str, Any] = {}
    if git_run_fn is not None:
        kwargs["run_fn"] = git_run_fn
    ok = run_and_commit_report(report, path, **kwargs)
    if ok:
        print(f"Run report committed and pushed: {path}", flush=True)
    else:
        print(
            f"Warning: run report written to {path} but commit/push failed",
            flush=True,
        )


def _cli(argv: list[str] | None = None, *, git_run_fn: Any = None) -> int:
    args = _parse_args(argv)
    started_at = datetime.now(UTC).isoformat()
    start = time.monotonic()
    stats: dict[str, int] = {}
    output_path = fetch_koi_table(args.output, stats=stats)
    elapsed = time.monotonic() - start
    _write_run_report(
        started_at=started_at,
        elapsed_seconds=elapsed,
        stats=stats,
        output_path=output_path,
        git_run_fn=git_run_fn,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
