"""Map TESS TOI table columns to CandidateFeatures and build a training set.

Reads the CSV produced by ``fetch_tess_toi.py`` and maps available TOI
columns to the 35 ``OptScore`` fields.  Most fields remain ``None``; XGBoost
handles missing values natively.

Column mapping (TOI → CandidateFeatures)
-----------------------------------------
snr                  → snr_score, log_snr_score
n_sectors            → transit_count_score  (proxy for # transits observed)
duration_hours / period_days
                     → duration_plausibility_score, duration_implausibility_score
depth_mmag           → large_depth_score   (1 mmag = 1000 ppm)
planet_radius_earth  → companion_radius_too_large_score

Labels
------
  CP / confirmed      → 1 (planet candidate)
  FP / EB (false pos) → 0 (false positive)

Usage
-----
    python Skills/build_tess_training_data.py [--input data/tess_toi.csv]
                                               [--output data/tess_training.pkl]
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
# Feature mapping helpers (same as build_training_data.py)
# ---------------------------------------------------------------------------


def _safe(val: Any) -> float | None:
    """Return float or None if missing/NaN."""
    try:
        f = float(val)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _sigmoid(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    return 1.0 / (1.0 + np.exp(-(x - mu) / sigma))


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


# ---------------------------------------------------------------------------
# Row → CandidateFeatures
# ---------------------------------------------------------------------------


def row_to_features(row: pd.Series) -> CandidateFeatures:  # noqa: C901
    """Map one TOI row to a ``CandidateFeatures`` instance.

    Args:
        row: A pandas Series from the TESS TOI CSV (with normalised column
             names as produced by ``fetch_tess_toi.py``).

    Returns:
        A ``CandidateFeatures`` with all mappable fields set and the rest
        left as ``None``.
    """
    snr = _safe(row.get("snr"))
    period = _safe(row.get("period_days"))
    duration_hr = _safe(row.get("duration_hours"))
    depth_mmag = _safe(row.get("depth_mmag"))
    prad = _safe(row.get("planet_radius_earth"))
    n_sectors = _safe(row.get("n_sectors"))

    # SNR score — sigmoid centred at SNR=15, σ=5
    snr_score = _clip01(_sigmoid(snr, mu=15.0, sigma=5.0)) if snr is not None else None
    log_snr_score = None
    if snr is not None and snr > 0:
        log_snr_score = _clip01(_sigmoid(np.log10(snr), mu=1.2, sigma=0.4))

    # Transit count score — use n_sectors as proxy (each sector ≈ several transits)
    transit_count_score = None
    if n_sectors is not None:
        transit_count_score = _clip01(_sigmoid(n_sectors, mu=3.0, sigma=1.5))

    # Duration plausibility — expected duration for orbital period
    duration_plausibility_score = None
    duration_implausibility_score = None
    if period is not None and duration_hr is not None:
        duration_days = duration_hr / 24.0
        max_dur_days = 0.12 * (period ** (1.0 / 3.0))
        ratio = duration_days / max(max_dur_days, 1e-9)
        duration_plausibility_score = _clip01(_sigmoid(-ratio, mu=-1.0, sigma=0.3))
        duration_implausibility_score = _clip01(_sigmoid(ratio, mu=2.0, sigma=0.5))

    # Large depth score — convert mmag to ppm (1 mmag = 1000 ppm)
    large_depth_score = None
    if depth_mmag is not None:
        depth_ppm = depth_mmag * 1_000.0
        large_depth_score = _clip01(_sigmoid(np.log10(max(depth_ppm, 1.0)), mu=4.0, sigma=0.5))

    # Companion radius too large — > 15 R_Earth suggests stellar companion
    companion_radius_too_large_score = None
    if prad is not None and prad > 0:
        companion_radius_too_large_score = _clip01(_sigmoid(prad, mu=15.0, sigma=3.0))

    return CandidateFeatures(
        snr_score=snr_score,
        log_snr_score=log_snr_score,
        transit_count_score=transit_count_score,
        duration_plausibility_score=duration_plausibility_score,
        duration_implausibility_score=duration_implausibility_score,
        large_depth_score=large_depth_score,
        companion_radius_too_large_score=companion_radius_too_large_score,
    )


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

_POSITIVE_DISPOSITIONS = {"CP"}  # confirmed planet
_NEGATIVE_DISPOSITIONS = {"FP", "EB"}  # false positive / eclipsing binary


def _disposition_to_label(disp: str) -> int | None:
    """Return 1 (planet), 0 (FP), or None (exclude)."""
    d = str(disp).strip().upper()
    if d in _POSITIVE_DISPOSITIONS:
        return 1
    if d in _NEGATIVE_DISPOSITIONS:
        return 0
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_training_data(
    input_path: str | Path,
    output_path: str | Path,
) -> tuple[list[CandidateFeatures], list[int]]:
    """Build and pickle a labelled feature dataset from a TESS TOI CSV.

    Args:
        input_path: Path to the TOI CSV (as produced by ``fetch_tess_toi.py``).
        output_path: Where to write the pickled training data.

    Returns:
        ``(features_list, labels)`` — a list of ``CandidateFeatures`` objects
        and a matching list of binary labels (1 = planet, 0 = FP).
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    features_list: list[CandidateFeatures] = []
    labels: list[int] = []

    for _, row in df.iterrows():
        disp = row.get("tfopwg_disposition", "")
        label = _disposition_to_label(str(disp))
        if label is None:
            continue
        features_list.append(row_to_features(row))
        labels.append(label)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as fh:
        pickle.dump({"features_list": features_list, "labels": labels}, fh)

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    print(f"Built {len(labels):,} training examples → {out}")
    print(f"  Planet (CP): {n_pos:,}  |  False pos (FP/EB): {n_neg:,}")
    return features_list, labels


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="data/tess_toi.csv")
    p.add_argument("--output", default="data/tess_training.pkl")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_training_data(args.input, args.output)
