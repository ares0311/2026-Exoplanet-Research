"""Train an XGBoostScorer on Kepler KOI labelled data with k-fold CV.

Reads the pickle produced by ``build_training_data.py``, trains an
``XGBoostScorer``, evaluates with stratified k-fold cross-validation,
and saves the final model trained on the full dataset.

Metrics reported per fold and overall
--------------------------------------
  ROC-AUC, accuracy (threshold=0.5), precision, recall, F1

Usage
-----
    python Skills/train_xgboost.py [--input data/kepler_training.pkl]
                                    [--output models/xgboost_koi.json]
                                    [--folds 5]
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from exo_toolkit.ml.xgboost_scorer import XGBoostScorer
from exo_toolkit.schemas import CandidateFeatures

# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def _metrics(
    y_true: list[int],
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute binary classification metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(np.sum((y_pred == 1) & (np.array(y_true) == 1)))
    fp = int(np.sum((y_pred == 1) & (np.array(y_true) == 0)))
    tn = int(np.sum((y_pred == 0) & (np.array(y_true) == 0)))
    fn = int(np.sum((y_pred == 0) & (np.array(y_true) == 1)))

    acc = (tp + tn) / max(len(y_true), 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    auc = _roc_auc(y_true, y_prob)
    return {"auc": auc, "acc": acc, "precision": prec, "recall": rec, "f1": f1}


def _roc_auc(y_true: list[int], y_prob: np.ndarray) -> float:
    """Compute ROC-AUC via trapezoid rule (no sklearn needed)."""
    y_arr = np.array(y_true)
    order = np.argsort(-y_prob)
    y_sorted = y_arr[order]
    n_pos = y_arr.sum()
    n_neg = len(y_arr) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tp_cumsum = np.cumsum(y_sorted)
    fp_cumsum = np.cumsum(1 - y_sorted)
    tpr = tp_cumsum / n_pos
    fpr = fp_cumsum / n_neg
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])
    return float(np.trapz(tpr, fpr))


def _stratified_kfold_indices(
    labels: list[int], k: int, seed: int = 42
) -> list[tuple[list[int], list[int]]]:
    """Return list of (train_idx, val_idx) pairs for stratified k-fold."""
    rng = np.random.default_rng(seed)
    pos = [i for i, lbl in enumerate(labels) if lbl == 1]
    neg = [i for i, lbl in enumerate(labels) if lbl == 0]

    rng.shuffle(pos_arr := np.array(pos))
    rng.shuffle(neg_arr := np.array(neg))

    pos_folds = np.array_split(pos_arr, k)
    neg_folds = np.array_split(neg_arr, k)

    splits = []
    for i in range(k):
        val_idx = list(pos_folds[i]) + list(neg_folds[i])
        train_idx = [
            j for fold_i in range(k) if fold_i != i
            for j in (list(pos_folds[fold_i]) + list(neg_folds[fold_i]))
        ]
        splits.append((train_idx, val_idx))
    return splits


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def train_and_evaluate(
    features_list: list[CandidateFeatures],
    labels: list[int],
    *,
    k_folds: int = 5,
    output_path: str | Path = "models/xgboost_koi.json",
    seed: int = 42,
) -> dict[str, float]:
    """Train XGBoostScorer with k-fold CV and save model on full dataset.

    Args:
        features_list: Training samples.
        labels: Binary labels (1 = planet, 0 = FP).
        k_folds: Number of stratified CV folds.
        output_path: Where to save the trained model metadata JSON.
        seed: Random seed for reproducibility.

    Returns:
        Dict of mean CV metrics: auc, acc, precision, recall, f1.
    """
    print(f"Training XGBoostScorer — {len(labels):,} samples, {k_folds}-fold CV")
    splits = _stratified_kfold_indices(labels, k=k_folds, seed=seed)

    fold_metrics: list[dict[str, float]] = []
    for fold_i, (train_idx, val_idx) in enumerate(splits):
        X_train = [features_list[i] for i in train_idx]
        y_train = [labels[i] for i in train_idx]
        X_val = [features_list[i] for i in val_idx]
        y_val = [labels[i] for i in val_idx]

        scorer = XGBoostScorer()
        scorer.fit(X_train, y_train, seed=seed)
        y_prob = scorer.predict_proba_batch(X_val)
        m = _metrics(y_val, y_prob)
        fold_metrics.append(m)
        print(
            f"  Fold {fold_i + 1}/{k_folds}  "
            f"AUC={m['auc']:.3f}  Acc={m['acc']:.3f}  "
            f"P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}"
        )

    # Mean metrics across folds.
    keys = list(fold_metrics[0].keys())
    mean_metrics = {k: float(np.mean([fm[k] for fm in fold_metrics])) for k in keys}
    print(
        f"\nMean CV  "
        f"AUC={mean_metrics['auc']:.3f}  Acc={mean_metrics['acc']:.3f}  "
        f"F1={mean_metrics['f1']:.3f}"
    )

    # Train on full dataset and save.
    print("\nTraining on full dataset …")
    final_scorer = XGBoostScorer()
    final_scorer.fit(features_list, labels, seed=seed)
    final_scorer.save(output_path)
    print(f"Model saved → {output_path}")

    return mean_metrics


def load_training_data(
    pickle_path: str | Path = "data/kepler_training.pkl",
) -> tuple[list[CandidateFeatures], list[int]]:
    """Load features and labels from a pickle written by ``build_training_data``."""
    with Path(pickle_path).open("rb") as fh:
        data = pickle.load(fh)
    return data["features_list"], data["labels"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        default="data/kepler_training.pkl",
        help="Training pickle from build_training_data.py",
    )
    p.add_argument(
        "--output",
        default="models/xgboost_koi.json",
        help="Output model metadata JSON path (default: models/xgboost_koi.json)",
    )
    p.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of CV folds (default: 5)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    features_list, labels = load_training_data(args.input)
    train_and_evaluate(
        features_list,
        labels,
        k_folds=args.folds,
        output_path=args.output,
        seed=args.seed,
    )
