"""Download the TESS TOI (Targets of Interest) disposition table from ExoFOP-TESS.

Fetches quality training labels: CP/KP (positive class) and FP/FA (negative
class).  Planet candidates (PC/APC) are excluded — unresolved, noisy labels.
ExoFOP does not use an "EB" disposition; eclipsing binaries appear as FP.

Usage
-----
    python Skills/fetch_tess_toi.py [--output data/tess_toi.csv]

Output
------
    CSV with columns: toi, tic_id, tfopwg_disposition, period_days,
    epoch_bjd, duration_hours, depth_mmag, planet_radius_earth, snr,
    n_sectors, stellar_radius_sun, stellar_teff, stellar_logg, tmag

DECISION-015 note
-----------------
    epoch_bjd is required by Skills/download_tess_lightcurves.py.
    Rows with epoch_bjd=0 or missing are rejected by the download script.

The CLI entry point writes a structured completion record via
``Skills/run_report.py`` after each run (AGENTS.md Run Report Policy,
Rule 7) and commits/pushes only that record. A download/parse failure
propagates uncaught (fail loudly) rather than writing a false-success
report.
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
# Constants
# ---------------------------------------------------------------------------

_EXOFOP_URL = (
    "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
)

# ExoFOP column → normalised name
_COL_MAP = {
    "TOI": "toi",
    "TIC ID": "tic_id",
    "TFOPWG Disposition": "tfopwg_disposition",
    "Period (days)": "period_days",
    "Epoch (BJD)": "epoch_bjd",
    "Duration (hours)": "duration_hours",
    "Depth (mmag)": "depth_mmag",
    "Planet Radius (R_Earth)": "planet_radius_earth",
    "Planet SNR": "snr",
    "Number of Sectors": "n_sectors",
    "Stellar Radius (R_Sun)": "stellar_radius_sun",
    "Stellar Eff Temp (K)": "stellar_teff",
    "Stellar log(g) (cm/s^2)": "stellar_logg",
    "TESS Mag": "tmag",
}

# Keep only these dispositions as training labels
# CP = Confirmed Planet, KP = Known Planet → positive class
# FP = False Positive, FA = False Alarm   → negative class
_KEEP_DISPOSITIONS = {"CP", "KP", "FP", "FA"}
_REQUIRED_TRAINING_COLUMNS = {
    "tic_id",
    "tfopwg_disposition",
    "period_days",
    "epoch_bjd",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _default_fetch(url: str) -> bytes:
    """Fetch URL bytes using certifi SSL context when available."""
    import ssl
    from urllib.request import urlopen

    try:
        import certifi
        ctx: ssl.SSLContext | None = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = None
    with urlopen(url, timeout=60, context=ctx) as resp:
        return resp.read()


def fetch_toi_table(
    output_path: str | Path = "data/tess_toi.csv",
    *,
    fetch_fn: Callable[[str], bytes] | None = None,
    stats: dict[str, int] | None = None,
) -> Path:
    """Download TESS TOI table from ExoFOP and save to *output_path*.

    Args:
        output_path: Destination CSV path.
        fetch_fn: Injectable fetch function (url -> bytes). Defaults to
            urlopen with certifi SSL context. Supply a mock in tests.
        stats: When given, populated with
            ``{"written", "errors", "total", "rejected_ephemerides"}``
            in-place on return — a non-breaking side channel for callers
            (e.g. the CLI's Run Report), matching the pattern used by this
            project's other acquisition Skills.

    Returns:
        Path to the written CSV file.
    """
    from io import BytesIO

    import pandas as pd

    _fetch = fetch_fn or _default_fetch

    print("Downloading TESS TOI table from ExoFOP …")
    df = pd.read_csv(BytesIO(_fetch(_EXOFOP_URL)), comment="#")

    # Rename to normalised column names
    df = df.rename(columns={k: v for k, v in _COL_MAP.items() if k in df.columns})
    missing = sorted(_REQUIRED_TRAINING_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(
            "ExoFOP TOI table is missing required CNN training columns: "
            + ", ".join(missing)
        )

    # Keep only columns we renamed (ignore extra ExoFOP columns)
    keep = [v for v in _COL_MAP.values() if v in df.columns]
    df = df[keep]

    # Filter to labelled dispositions only
    if "tfopwg_disposition" in df.columns:
        df = df[df["tfopwg_disposition"].isin(_KEEP_DISPOSITIONS)]
    for column in ("tic_id", "period_days", "epoch_bjd"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    before_ephemeris_filter = len(df)
    df = df[
        df["tic_id"].notna()
        & (df["period_days"] > 0.0)
        & (df["epoch_bjd"] >= 2_000_000.0)
    ]
    rejected_ephemerides = before_ephemeris_filter - len(df)
    if df.empty:
        raise ValueError("ExoFOP TOI table has no rows with valid BJD ephemerides")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    disp_counts = (
        df["tfopwg_disposition"].value_counts() if "tfopwg_disposition" in df.columns else {}
    )
    print(f"Saved {len(df):,} TOIs → {output}")
    if rejected_ephemerides:
        print(f"  rejected invalid/missing ephemerides: {rejected_ephemerides:,}")
    for disp, n in sorted(disp_counts.items()):
        print(f"  {disp:4s}: {n:,}")

    if stats is not None:
        stats["written"] = len(df)
        stats["errors"] = 0
        stats["total"] = len(df)
        stats["rejected_ephemerides"] = rejected_ephemerides

    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output",
        default="data/tess_toi.csv",
        help="Destination CSV path (default: data/tess_toi.csv)",
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
    """Append and publish one fetch_tess_toi completion report
    (AGENTS.md Rule 7)."""
    report = RunReport(
        script="fetch_tess_toi",
        status="success",
        started_at=started_at,
        completed_at=datetime.now(UTC).isoformat(),
        elapsed_seconds=elapsed_seconds,
        items_processed=stats.get("total", 0),
        items_written=stats.get("written", 0),
        items_failed=stats.get("errors", 0),
        output_paths=(str(output_path),),
        notes=f"rejected_ephemerides={stats.get('rejected_ephemerides', 0)}",
    )
    path = report_path_for("fetch_tess_toi")
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
    output_path = fetch_toi_table(args.output, stats=stats)
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
