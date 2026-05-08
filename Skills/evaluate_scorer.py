"""Compare Bayesian vs XGBoost scorer performance on a labelled training set.

Loads a pickled training dataset (as produced by ``build_training_data.py``
or ``build_tess_training_data.py``) and evaluates both scorers using
stratified k-fold cross-validation.  Prints a comparison table of
ROC-AUC, F1, precision, and recall.

Optionally saves ROC curves to a PNG if ``--plot`` is specified.

Usage
-----
    python Skills/evaluate_scorer.py --data data/kepler_training.pkl
    python Skills/evaluate_scorer.py --data data/tess_training.pkl --model data/model.json
    python Skills/evaluate_scorer.py --data data/kepler_training.pkl \\
        --model data/model.json --plot reports/roc_curves.png
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from exo_toolkit.schemas import CandidateFeatures, CandidateSignal

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_data(path: Path) -> tuple[list[CandidateFeatures], list[int]]:
    """Load (features_list, labels) from a pickle file."""
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("rb") as fh:
        data = pickle.load(fh)
    return data["features_list"], data["labels"]


def _roc_auc(y_true: list[int], y_prob: np.ndarray) -> float:
    """Compute ROC-AUC via trapezoidal integration."""
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_prob)[::-1]
    y_sorted = np.array(y_true)[order]
    tpr = np.concatenate([[0.0], np.cumsum(y_sorted) / n_pos])
    fpr = np.concatenate([[0.0], np.cumsum(1 - y_sorted) / n_neg])
    return float(np.trapezoid(tpr, fpr))


def _threshold_metrics(
    y_true: list[int], y_prob: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (np.array(y_true) == 1)).sum())
    fp = int(((y_pred == 1) & (np.array(y_true) == 0)).sum())
    fn = int(((y_pred == 0) & (np.array(y_true) == 1)).sum())
    tn = int(((y_pred == 0) & (np.array(y_true) == 0)).sum())
    acc = (tp + tn) / max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {"acc": acc, "precision": precision, "recall": recall, "f1": f1,
            "auc": _roc_auc(y_true, y_prob)}


def _stratified_splits(
    labels: list[int], k: int = 5
) -> list[tuple[list[int], list[int]]]:
    """Return (train_idx, val_idx) pairs for stratified k-fold."""
    pos = [i for i, lbl in enumerate(labels) if lbl == 1]
    neg = [i for i, lbl in enumerate(labels) if lbl == 0]
    splits = []
    for fold in range(k):
        val_p = pos[fold::k]
        val_n = neg[fold::k]
        val = sorted(val_p + val_n)
        train = sorted(set(range(len(labels))) - set(val))
        splits.append((train, val))
    return splits


def _dummy_signal() -> CandidateSignal:
    """Minimal CandidateSignal for Bayesian scoring."""
    return CandidateSignal(
        candidate_id="eval-000",
        mission="Kepler",
        target_id="eval",
        period_days=5.0,
        epoch_bjd=2454833.0,
        duration_hours=2.0,
        depth_ppm=1000.0,
        transit_count=5,
        snr=10.0,
    )


# ---------------------------------------------------------------------------
# Scorer wrappers
# ---------------------------------------------------------------------------


def _bayesian_probs(
    features_list: list[CandidateFeatures],
) -> np.ndarray:
    """Return P(planet_candidate) from the Bayesian log-score model."""
    from exo_toolkit.scoring import score_candidate

    signal = _dummy_signal()
    probs = []
    for f in features_list:
        posterior, _ = score_candidate(signal, f)
        probs.append(posterior.planet_candidate)
    return np.array(probs, dtype=np.float64)


def _xgb_probs(
    features_list: list[CandidateFeatures],
    model_path: Path,
) -> np.ndarray:
    """Return P(planet_candidate) from a trained XGBoostScorer."""
    from exo_toolkit.ml.xgboost_scorer import XGBoostScorer

    scorer = XGBoostScorer.load(model_path)
    return scorer.predict_proba_batch(features_list)


# ---------------------------------------------------------------------------
# Cross-validation evaluation
# ---------------------------------------------------------------------------


def evaluate(
    features_list: list[CandidateFeatures],
    labels: list[int],
    *,
    model_path: Path | None = None,
    k_folds: int = 5,
    plot_path: Path | None = None,
) -> dict[str, dict[str, float]]:
    """Run stratified k-fold evaluation of Bayesian and (optionally) XGBoost.

    Args:
        features_list: Labelled feature vectors.
        labels: Binary labels (1 = planet, 0 = FP).
        model_path: Path to a saved XGBoostScorer metadata JSON.  When
            ``None`` only the Bayesian scorer is evaluated.
        k_folds: Number of cross-validation folds.
        plot_path: If given, save a ROC-curve PNG here (requires matplotlib).

    Returns:
        ``{"bayesian": {auc, f1, ...}, "xgboost": {auc, f1, ...}}``
    """
    from exo_toolkit.ml.xgboost_scorer import XGBoostScorer

    splits = _stratified_splits(labels, k=k_folds)

    bayes_metrics: list[dict[str, float]] = []
    xgb_metrics: list[dict[str, float]] = []

    roc_data: dict[str, list[Any]] = {"bayesian": [], "xgboost": []}

    for fold, (train_idx, val_idx) in enumerate(splits, 1):
        val_features = [features_list[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]

        # Bayesian — no fitting needed, purely deterministic
        bayes_p = _bayesian_probs(val_features)
        bayes_metrics.append(_threshold_metrics(val_labels, bayes_p))
        roc_data["bayesian"].append((val_labels, bayes_p))

        # XGBoost — train on fold then predict
        train_features = [features_list[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        xgb_scorer = XGBoostScorer()
        xgb_scorer.fit(train_features, train_labels)
        xgb_p = xgb_scorer.predict_proba_batch(val_features)
        xgb_metrics.append(_threshold_metrics(val_labels, xgb_p))
        roc_data["xgboost"].append((val_labels, xgb_p))

        print(
            f"  Fold {fold}/{k_folds} — "
            f"Bayes AUC={bayes_metrics[-1]['auc']:.3f}"
            + (f"  XGB AUC={xgb_metrics[-1]['auc']:.3f}" if xgb_metrics else "")
        )

    def _mean(lst: list[dict[str, float]], key: str) -> float:
        return float(np.mean([m[key] for m in lst]))

    result: dict[str, dict[str, float]] = {
        "bayesian": {k: _mean(bayes_metrics, k) for k in bayes_metrics[0]},
    }
    if xgb_metrics:
        result["xgboost"] = {k: _mean(xgb_metrics, k) for k in xgb_metrics[0]}

    if plot_path is not None:
        _save_roc_plot(roc_data, plot_path)

    return result


def _save_roc_plot(
    roc_data: dict[str, list[Any]], path: Path
) -> None:
    """Save ROC curves for all scorers to *path* (PNG)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping ROC plot")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    colours = {"bayesian": "steelblue", "xgboost": "darkorange"}

    for name, folds in roc_data.items():
        if not folds:
            continue
        all_true: list[int] = []
        all_prob: list[float] = []
        for y_true, y_prob in folds:
            all_true.extend(y_true)
            all_prob.extend(y_prob.tolist())

        y_arr = np.array(all_true)
        p_arr = np.array(all_prob)
        order = np.argsort(p_arr)[::-1]
        y_s = y_arr[order]
        n_pos, n_neg = int(y_s.sum()), int((1 - y_s).sum())
        tpr = np.concatenate([[0.0], np.cumsum(y_s) / max(n_pos, 1)])
        fpr = np.concatenate([[0.0], np.cumsum(1 - y_s) / max(n_neg, 1)])
        auc = float(np.trapezoid(tpr, fpr))
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=colours.get(name))

    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Exoplanet Candidate Scorer Comparison")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"ROC plot saved → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_table(results: dict[str, dict[str, float]]) -> None:
    keys = ["auc", "f1", "precision", "recall", "acc"]
    header = f"{'Scorer':<12}" + "".join(f"{k:>12}" for k in keys)
    print("\n" + header)
    print("-" * len(header))
    for name, m in results.items():
        row = f"{name:<12}" + "".join(f"{m.get(k, float('nan')):>12.4f}" for k in keys)
        print(row)
    print()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, help="Path to training .pkl file")
    p.add_argument("--model", default=None, help="XGBoostScorer metadata JSON (optional)")
    p.add_argument("--k-folds", type=int, default=5)
    p.add_argument("--plot", default=None, help="Save ROC plot to this PNG path")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    features_list, labels = _load_data(Path(args.data))
    n_pos, n_neg = sum(labels), len(labels) - sum(labels)
    print(f"Loaded {len(labels):,} examples  (pos={n_pos:,}, neg={n_neg:,})")
    print(f"Running {args.k_folds}-fold stratified cross-validation …\n")

    model_path = Path(args.model) if args.model else None
    plot_path = Path(args.plot) if args.plot else None
    results = evaluate(
        features_list, labels,
        model_path=model_path,
        k_folds=args.k_folds,
        plot_path=plot_path,
    )
    _print_table(results)
