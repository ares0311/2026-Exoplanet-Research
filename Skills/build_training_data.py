"""Map Kepler KOI table columns to CandidateFeatures and build a training set.

Reads the CSV produced by ``fetch_kepler_tce.py`` and maps available KOI
columns to the 35 ``OptScore`` fields.  Most fields remain ``None``; XGBoost
handles missing values natively.

Column mapping (KOI → CandidateFeatures)
-----------------------------------------
koi_model_snr           → snr_score, log_snr_score
koi_count               → transit_count_score
koi_duration / period   → duration_plausibility_score, duration_implausibility_score
koi_depth               → large_depth_score
koi_prad                → companion_radius_too_large_score
koi_dikco_msky          → centroid_offset_score

Labels
------
  CONFIRMED    → 1 (planet candidate)
  FALSE POSITIVE → 0 (false positive)
  CANDIDATE    → excluded (noisy label)

Usage
-----
    python Skills/build_training_data.py [--input data/kepler_koi.csv]
                                          [--output data/kepler_training.pkl]
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from exo_toolkit.schemas import CandidateFeatures

# ---------------------------------------------------------------------------
# Feature mapping helpers
# ---------------------------------------------------------------------------


def _safe(val: Any) -> float | None:
    """Return float or None if missing/NaN."""
    try:
        f = float(val)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _sigmoid(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    return float(1.0 / (1.0 + np.exp(-(x - mu) / sigma)))


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def row_to_features(row: pd.Series) -> CandidateFeatures:
    """Convert one KOI table row to a ``CandidateFeatures`` instance."""
    snr = _safe(row.get("koi_model_snr"))
    count = _safe(row.get("koi_count"))
    dur_h = _safe(row.get("koi_duration"))
    period_d = _safe(row.get("koi_period"))
    depth_ppm = _safe(row.get("koi_depth"))
    prad = _safe(row.get("koi_prad"))
    centroid = _safe(row.get("koi_dikco_msky"))

    # snr_score: sigmoid centred at snr=10, width=5
    snr_score = _sigmoid(snr, mu=10.0, sigma=5.0) if snr is not None else None

    # log_snr_score: log10(snr) normalised to [0,1] with cap at snr=100
    log_snr_score = (
        _clip01(np.log10(max(snr, 1.0)) / 2.0) if snr is not None else None
    )

    # transit_count_score: more transits → more confidence; cap at 20
    transit_count_score = _clip01(count / 20.0) if count is not None else None

    # duration_plausibility: transit duration should be a small fraction of period
    duration_plausibility_score: float | None = None
    duration_implausibility_score: float | None = None
    if dur_h is not None and period_d is not None and period_d > 0:
        frac = (dur_h / 24.0) / period_d
        # Typical planet: frac ~ 0.01–0.05. Score → 1 when frac small.
        duration_plausibility_score = _clip01(1.0 - frac / 0.15)
        duration_implausibility_score = _clip01(frac / 0.15)

    # large_depth_score: depth > 10 000 ppm (1%) suggests EB; cap at 50 000 ppm
    large_depth_score = (
        _clip01(depth_ppm / 50_000.0) if depth_ppm is not None else None
    )

    # companion_radius_too_large_score: Rp > 15 R⊕ → stellar companion; cap at 30
    companion_radius_too_large_score = (
        _clip01(prad / 30.0) if prad is not None else None
    )

    # centroid_offset_score: offset > 1 arcsec suspicious; cap at 5 arcsec
    centroid_offset_score = (
        _clip01(abs(centroid) / 5.0) if centroid is not None else None
    )

    return CandidateFeatures(
        snr_score=snr_score,
        log_snr_score=log_snr_score,
        transit_count_score=transit_count_score,
        duration_plausibility_score=duration_plausibility_score,
        duration_implausibility_score=duration_implausibility_score,
        large_depth_score=large_depth_score,
        companion_radius_too_large_score=companion_radius_too_large_score,
        centroid_offset_score=centroid_offset_score,
        # All other OptScore fields not mappable from KOI table → None
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_training_data(
    input_path: str | Path = "data/kepler_koi.csv",
    output_path: str | Path = "data/kepler_training.pkl",
) -> tuple[list[CandidateFeatures], list[int]]:
    """Build labelled training set from a KOI CSV.

    Returns:
        Tuple of ``(features_list, labels)`` where labels are 1 (planet) or
        0 (false positive).  Also writes a pickle to *output_path*.
    """
    df = pd.read_csv(input_path)

    # Keep only clean labels.
    df = df[df["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])].copy()
    df = df.dropna(subset=["koi_disposition"])

    features_list: list[CandidateFeatures] = []
    labels: list[int] = []

    for _, row in df.iterrows():
        features_list.append(row_to_features(row))
        labels.append(1 if row["koi_disposition"] == "CONFIRMED" else 0)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as fh:
        pickle.dump({"features_list": features_list, "labels": labels}, fh)

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    print(f"Built {len(labels):,} training examples → {output}")
    print(f"  Confirmed (label=1): {n_pos:,}")
    print(f"  False pos (label=0): {n_neg:,}")
    return features_list, labels


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        default="data/kepler_koi.csv",
        help="KOI CSV from fetch_kepler_tce.py (default: data/kepler_koi.csv)",
    )
    p.add_argument(
        "--output",
        default="data/kepler_training.pkl",
        help="Output pickle path (default: data/kepler_training.pkl)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_training_data(args.input, args.output)
