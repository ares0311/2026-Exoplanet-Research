"""Download the TESS TOI (Targets of Interest) disposition table from ExoFOP-TESS.

Fetches confirmed planets (CP) and false positives (FP/EB) for use as
training labels.  Planet candidates (PC) are excluded — they are unresolved
and make noisy labels.

Usage
-----
    python Skills/fetch_tess_toi.py [--output data/tess_toi.csv]

Output
------
    CSV with columns: toi, tic_id, tfopwg_disposition, period_days,
    duration_hours, depth_mmag, planet_radius_earth, snr,
    n_sectors, stellar_radius_sun, stellar_teff, stellar_logg, tmag
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

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
_KEEP_DISPOSITIONS = {"CP", "FP", "EB"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_toi_table(output_path: str | Path = "data/tess_toi.csv") -> Path:
    """Download TESS TOI table from ExoFOP and save to *output_path*.

    Args:
        output_path: Destination CSV path.

    Returns:
        Path to the written CSV file.
    """
    import pandas as pd

    print("Downloading TESS TOI table from ExoFOP …")
    df = pd.read_csv(_EXOFOP_URL, comment="#")

    # Rename to normalised column names
    df = df.rename(columns={k: v for k, v in _COL_MAP.items() if k in df.columns})

    # Keep only columns we renamed (ignore extra ExoFOP columns)
    keep = [v for v in _COL_MAP.values() if v in df.columns]
    df = df[keep]

    # Filter to labelled dispositions only
    if "tfopwg_disposition" in df.columns:
        df = df[df["tfopwg_disposition"].isin(_KEEP_DISPOSITIONS)]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    disp_counts = (
        df["tfopwg_disposition"].value_counts() if "tfopwg_disposition" in df.columns else {}
    )
    print(f"Saved {len(df):,} TOIs → {output}")
    for disp, n in sorted(disp_counts.items()):
        print(f"  {disp:4s}: {n:,}")

    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output",
        default="data/tess_toi.csv",
        help="Destination CSV path (default: data/tess_toi.csv)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    fetch_toi_table(args.output)
