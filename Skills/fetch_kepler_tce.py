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
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

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


def fetch_koi_table(output_path: str | Path = "data/kepler_koi.csv") -> Path:
    """Download KOI cumulative table and save to *output_path*.

    Returns:
        Path to the written CSV file.
    """
    from astroquery.ipac.nexsci.nea_exoplanet import NasaExoplanetArchive

    print("Querying NASA Exoplanet Archive (KOI cumulative table) …")
    table = NasaExoplanetArchive.query_criteria(
        table="cumulative",
        select=_SELECT_COLS,
        where="koi_disposition like 'CONFIRMED' or koi_disposition like 'FALSE POSITIVE'",
    )

    df = table.to_pandas()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    n_conf = (df["koi_disposition"] == "CONFIRMED").sum()
    n_fp = (df["koi_disposition"] == "FALSE POSITIVE").sum()
    print(f"Saved {len(df):,} KOIs → {output}")
    print(f"  Confirmed  : {n_conf:,}")
    print(f"  False pos. : {n_fp:,}")
    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output",
        default="data/kepler_koi.csv",
        help="Destination CSV path (default: data/kepler_koi.csv)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    fetch_koi_table(args.output)
