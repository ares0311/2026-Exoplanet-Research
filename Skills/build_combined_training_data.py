"""Merge Kepler KOI and TESS TOI training sets into one pickle.

Loads the pickles produced by ``build_training_data.py`` (Kepler) and
``build_tess_training_data.py`` (TESS), concatenates them, and writes a
combined pickle.  Optionally down-weights the majority source so neither
dataset dominates.

Usage
-----
    python Skills/build_combined_training_data.py \\
        --kepler data/kepler_training.pkl \\
        --tess   data/tess_training.pkl \\
        --output data/combined_training.pkl \\
        [--max-per-source 2000]
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from exo_toolkit.schemas import CandidateFeatures

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def merge_datasets(
    *pkl_paths: str | Path,
    max_per_source: int | None = None,
    seed: int = 42,
) -> tuple[list[CandidateFeatures], list[int]]:
    """Load and concatenate multiple training pickles.

    Args:
        *pkl_paths: One or more paths to training pickles
            (``{"features_list": ..., "labels": ...}``).
        max_per_source: If given, cap each source at this many samples
            (stratified subsample to preserve class balance).
        seed: Random seed for subsampling.

    Returns:
        ``(features_list, labels)`` combined across all sources.
    """
    rng = np.random.default_rng(seed)
    all_features: list[CandidateFeatures] = []
    all_labels: list[int] = []

    for path in pkl_paths:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(p)
        with p.open("rb") as fh:
            data = pickle.load(fh)
        fl: list[CandidateFeatures] = data["features_list"]
        lb: list[int] = data["labels"]

        if max_per_source is not None and len(lb) > max_per_source:
            fl, lb = _stratified_subsample(fl, lb, max_per_source, rng)

        all_features.extend(fl)
        all_labels.extend(lb)

    return all_features, all_labels


def _stratified_subsample(
    features: list[CandidateFeatures],
    labels: list[int],
    n: int,
    rng: np.random.Generator,
) -> tuple[list[CandidateFeatures], list[int]]:
    """Return a stratified random subsample of size *n*."""
    pos_idx = [i for i, lbl in enumerate(labels) if lbl == 1]
    neg_idx = [i for i, lbl in enumerate(labels) if lbl == 0]
    n_pos = min(len(pos_idx), n // 2)
    n_neg = min(len(neg_idx), n - n_pos)
    sel_pos = rng.choice(pos_idx, size=n_pos, replace=False).tolist()
    sel_neg = rng.choice(neg_idx, size=n_neg, replace=False).tolist()
    sel = sorted(sel_pos + sel_neg)
    return [features[i] for i in sel], [labels[i] for i in sel]


def build_combined(
    kepler_path: str | Path,
    tess_path: str | Path,
    output_path: str | Path,
    *,
    max_per_source: int | None = None,
    seed: int = 42,
) -> tuple[list[CandidateFeatures], list[int]]:
    """Build and save a combined Kepler + TESS training dataset.

    Args:
        kepler_path: Path to Kepler training pickle.
        tess_path: Path to TESS training pickle.
        output_path: Destination pickle path.
        max_per_source: Optional per-source cap (stratified).
        seed: Random seed.

    Returns:
        ``(features_list, labels)`` for the combined dataset.
    """
    features_list, labels = merge_datasets(
        kepler_path, tess_path, max_per_source=max_per_source, seed=seed
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as fh:
        pickle.dump({"features_list": features_list, "labels": labels}, fh)

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    print(f"Combined dataset: {len(labels):,} samples → {out}")
    print(f"  Planet: {n_pos:,}  |  False pos: {n_neg:,}")
    return features_list, labels


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kepler", default="data/kepler_training.pkl")
    p.add_argument("--tess", default="data/tess_training.pkl")
    p.add_argument("--output", default="data/combined_training.pkl")
    p.add_argument(
        "--max-per-source",
        type=int,
        default=None,
        help="Cap each source at this many samples (stratified)",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_combined(
        args.kepler, args.tess, args.output,
        max_per_source=args.max_per_source,
        seed=args.seed,
    )
