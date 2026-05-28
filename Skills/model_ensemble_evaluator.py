"""End-to-end evaluation of the XGB+CNN+Bayesian ensemble on a test set.

Computes ROC-AUC, PR-AUC, F1, Brier score, and calibration error for each
model tier and the final stacked ensemble. Produces a side-by-side comparison
table.

Public API
----------
TierScore(name, auc_roc, auc_pr, f1, brier, ece)
EnsembleEvalResult(tiers, best_tier_by_auc, n_test, flag)
evaluate_ensemble(y_true, tier_scores, *, threshold, n_bins) -> EnsembleEvalResult
format_ensemble_eval(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TierScore:
    name: str           # "bayesian" | "xgboost" | "cnn" | "ensemble"
    auc_roc: float
    auc_pr: float
    f1: float
    brier: float
    ece: float          # Expected Calibration Error


@dataclass(frozen=True)
class EnsembleEvalResult:
    tiers: tuple[TierScore, ...]
    best_tier_by_auc: str
    n_test: int
    flag: str  # "OK" | "EMPTY" | "INVALID"


def _trapezoidal_auc_roc(y_true: list[int], scores: list[float]) -> float:
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    pairs = sorted(zip(scores, y_true, strict=True), key=lambda t: t[0], reverse=True)
    tpr_pts = [0.0]
    fpr_pts = [0.0]
    tp = fp = 0
    for _, lbl in pairs:
        if lbl == 1:
            tp += 1
        else:
            fp += 1
        tpr_pts.append(tp / n_pos)
        fpr_pts.append(fp / n_neg)
    auc = 0.0
    for i in range(1, len(fpr_pts)):
        auc += (fpr_pts[i] - fpr_pts[i - 1]) * (tpr_pts[i] + tpr_pts[i - 1]) / 2
    return auc


def _trapezoidal_auc_pr(y_true: list[int], scores: list[float]) -> float:
    pairs = sorted(zip(scores, y_true, strict=True), key=lambda t: t[0], reverse=True)
    n_pos = sum(y_true)
    if n_pos == 0:
        return 0.0
    tp = 0
    prev_rec = 0.0
    auc = 0.0
    for i, (_, lbl) in enumerate(pairs):
        if lbl == 1:
            tp += 1
        prec = tp / (i + 1)
        rec = tp / n_pos
        auc += prec * (rec - prev_rec)
        prev_rec = rec
    return auc


def _f1_at(y_true: list[int], scores: list[float], threshold: float) -> float:
    tp = fp = fn = 0
    for yt, ys in zip(y_true, scores, strict=True):
        pred = 1 if ys >= threshold else 0
        if pred == 1 and yt == 1:
            tp += 1
        elif pred == 1 and yt == 0:
            fp += 1
        elif pred == 0 and yt == 1:
            fn += 1
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def _brier(y_true: list[int], scores: list[float]) -> float:
    return sum((s - y) ** 2 for y, s in zip(y_true, scores, strict=True)) / len(y_true)


def _ece(y_true: list[int], scores: list[float], n_bins: int) -> float:
    bins: list[tuple[list[float], list[int]]] = [
        ([], []) for _ in range(n_bins)
    ]
    for y, s in zip(y_true, scores, strict=True):
        idx = min(int(s * n_bins), n_bins - 1)
        bins[idx][0].append(s)
        bins[idx][1].append(y)
    n = len(y_true)
    ece_val = 0.0
    for scrs, lbls in bins:
        if not scrs:
            continue
        avg_conf = sum(scrs) / len(scrs)
        avg_acc = sum(lbls) / len(lbls)
        ece_val += len(scrs) / n * abs(avg_conf - avg_acc)
    return ece_val


def evaluate_ensemble(
    y_true: list[int],
    tier_scores: dict[str, list[float]],
    *,
    threshold: float = 0.5,
    n_bins: int = 10,
) -> EnsembleEvalResult:
    """Compute evaluation metrics for each tier and the ensemble.

    Args:
        y_true: Binary ground-truth labels (0/1).
        tier_scores: Dict mapping tier name → predicted scores list.
        threshold: Classification threshold for F1 computation.
        n_bins: Number of calibration bins for ECE.

    Returns:
        EnsembleEvalResult with per-tier metrics and best tier.
    """
    if not y_true or not tier_scores:
        return EnsembleEvalResult(tiers=(), best_tier_by_auc="", n_test=0, flag="INVALID")

    n = len(y_true)
    for _name, scores in tier_scores.items():
        if len(scores) != n:
            return EnsembleEvalResult(
                tiers=(), best_tier_by_auc="", n_test=n, flag="INVALID"
            )

    tier_results: list[TierScore] = []
    for name, scores in tier_scores.items():
        tier_results.append(TierScore(
            name=name,
            auc_roc=_trapezoidal_auc_roc(y_true, scores),
            auc_pr=_trapezoidal_auc_pr(y_true, scores),
            f1=_f1_at(y_true, scores, threshold),
            brier=_brier(y_true, scores),
            ece=_ece(y_true, scores, n_bins),
        ))

    if not tier_results:
        return EnsembleEvalResult(tiers=(), best_tier_by_auc="", n_test=n, flag="EMPTY")

    best = max(tier_results, key=lambda t: t.auc_roc)
    return EnsembleEvalResult(
        tiers=tuple(tier_results),
        best_tier_by_auc=best.name,
        n_test=n,
        flag="OK",
    )


def format_ensemble_eval(result: EnsembleEvalResult) -> str:
    """Format a Markdown ensemble evaluation comparison table.

    Args:
        result: EnsembleEvalResult to format.

    Returns:
        Markdown string.
    """
    lines = [
        "## Ensemble Evaluation\n",
        f"Flag: `{result.flag}` | Test samples: {result.n_test}\n",
    ]
    if result.flag in ("INVALID", "EMPTY"):
        lines.append(f"\n_{result.flag}: cannot evaluate._\n")
        return "\n".join(lines)

    lines += [
        f"**Best by AUC-ROC**: {result.best_tier_by_auc}\n",
        "",
        "| Model | AUC-ROC | AUC-PR | F1 | Brier | ECE |",
        "|---|---|---|---|---|---|",
    ]
    for t in result.tiers:
        marker = " ★" if t.name == result.best_tier_by_auc else ""
        lines.append(
            f"| {t.name}{marker} | {t.auc_roc:.4f} | {t.auc_pr:.4f}"
            f" | {t.f1:.4f} | {t.brier:.4f} | {t.ece:.4f} |"
        )
    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Evaluate XGB+CNN+Bayesian ensemble.")
    parser.add_argument("eval_json",
                        help="JSON with y_true and tier_scores dict.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args(argv)

    data = json.loads(Path(args.eval_json).read_text())
    result = evaluate_ensemble(data["y_true"], data["tier_scores"],
                               threshold=args.threshold, n_bins=args.bins)
    print(format_ensemble_eval(result))
    return 0 if result.flag == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
