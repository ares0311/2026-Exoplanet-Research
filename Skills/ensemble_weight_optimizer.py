"""Grid-search XGB/CNN/Bayes blend weights to maximise AUC on validation set.

Enumerates all (w_xgb, w_cnn, w_bayes) triplets that sum to 1.0 on a
uniform grid (step size configurable). Uses an inline trapezoidal ROC-AUC
estimator; does NOT import from roc_auc_calculator.

Public API
----------
WeightSearchResult(best_weights, best_auc, n_combinations_tried,
                   grid_step, flag)
blend_scores(xgb_scores, cnn_scores, bayes_scores, weights) -> list[float]
optimize_weights(y_true, xgb_scores, cnn_scores, bayes_scores,
                 *, step) -> WeightSearchResult
format_weight_result(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WeightSearchResult:
    best_weights: tuple[float, float, float]  # (xgb, cnn, bayes)
    best_auc: float
    n_combinations_tried: int
    grid_step: float
    flag: str  # "OK" | "DEGENERATE" | "INVALID"


def blend_scores(
    xgb_scores: list[float],
    cnn_scores: list[float],
    bayes_scores: list[float],
    weights: tuple[float, float, float],
) -> list[float]:
    """Compute weighted average of three score lists.

    Args:
        xgb_scores: XGBoost model scores.
        cnn_scores: CNN model scores.
        bayes_scores: Bayesian model scores.
        weights: (w_xgb, w_cnn, w_bayes) — should sum to 1.0.

    Returns:
        List of blended scores.
    """
    w1, w2, w3 = weights
    return [
        w1 * x + w2 * c + w3 * b
        for x, c, b in zip(xgb_scores, cnn_scores, bayes_scores, strict=True)
    ]


def _trapezoidal_auc(y_true: list[int], scores: list[float]) -> float:
    """Compute ROC-AUC via trapezoidal rule.

    Args:
        y_true: Binary labels (0/1).
        scores: Predicted scores (higher = more positive).

    Returns:
        Area under the ROC curve in [0, 1].
    """
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5  # degenerate: return 0.5

    # Sort by score descending
    paired = sorted(zip(scores, y_true, strict=True), key=lambda t: t[0], reverse=True)

    tpr_list = [0.0]
    fpr_list = [0.0]
    tp = 0
    fp = 0
    for _, label in paired:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)

    # Trapezoidal integration
    auc = 0.0
    for i in range(1, len(fpr_list)):
        d_fpr = fpr_list[i] - fpr_list[i - 1]
        avg_tpr = (tpr_list[i] + tpr_list[i - 1]) / 2.0
        auc += d_fpr * avg_tpr
    return auc


def optimize_weights(
    y_true: list[int],
    xgb_scores: list[float],
    cnn_scores: list[float],
    bayes_scores: list[float],
    *,
    step: float = 0.1,
) -> WeightSearchResult:
    """Grid-search blend weights to maximise ROC-AUC.

    Args:
        y_true: Binary labels (0/1).
        xgb_scores: XGBoost scores (same length as y_true).
        cnn_scores: CNN scores (same length as y_true).
        bayes_scores: Bayesian scores (same length as y_true).
        step: Grid resolution; e.g. 0.1 gives 66 combinations.

    Returns:
        WeightSearchResult with best weights and AUC.
    """
    n = len(y_true)
    if n == 0 or n != len(xgb_scores) or n != len(cnn_scores) or n != len(bayes_scores):
        return WeightSearchResult(
            best_weights=(0.0, 0.0, 0.0),
            best_auc=0.0,
            n_combinations_tried=0,
            grid_step=step,
            flag="INVALID",
        )

    n_pos = sum(y_true)
    if n_pos == 0 or n_pos == n:
        # All same class — DEGENERATE
        return WeightSearchResult(
            best_weights=(1.0, 0.0, 0.0),
            best_auc=0.5,
            n_combinations_tried=0,
            grid_step=step,
            flag="DEGENERATE",
        )

    # Convert step to integer grid: N = round(1/step)
    N = round(1.0 / step)

    best_auc = -1.0
    best_w: tuple[float, float, float] = (1.0, 0.0, 0.0)
    n_tried = 0

    for i in range(N + 1):
        for j in range(N + 1 - i):
            k = N - i - j
            w1, w2, w3 = i * step, j * step, k * step
            blended = blend_scores(xgb_scores, cnn_scores, bayes_scores, (w1, w2, w3))
            auc = _trapezoidal_auc(y_true, blended)
            n_tried += 1
            if auc > best_auc:
                best_auc = auc
                best_w = (w1, w2, w3)

    return WeightSearchResult(
        best_weights=best_w,
        best_auc=best_auc,
        n_combinations_tried=n_tried,
        grid_step=step,
        flag="OK",
    )


def format_weight_result(result: WeightSearchResult) -> str:
    """Format weight search result as a Markdown summary.

    Args:
        result: WeightSearchResult to format.

    Returns:
        Markdown string.
    """
    w1, w2, w3 = result.best_weights
    lines: list[str] = [
        "## Ensemble Weight Optimisation\n",
        f"Flag: `{result.flag}` | Grid step: {result.grid_step}\n",
        f"Combinations tried: {result.n_combinations_tried}\n",
    ]

    if result.flag in ("INVALID", "DEGENERATE"):
        lines.append(f"\n_{result.flag}: could not optimise weights._\n")
        return "\n".join(lines)

    lines.append(f"**Best AUC**: {result.best_auc:.4f}\n")
    lines.append("")
    lines.append("| Scorer | Weight |")
    lines.append("|--------|--------|")
    lines.append(f"| XGBoost | {w1:.3f} |")
    lines.append(f"| CNN | {w2:.3f} |")
    lines.append(f"| Bayesian | {w3:.3f} |")

    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Grid-search blend weights to maximise AUC."
    )
    parser.add_argument(
        "validation_json",
        help="Path to JSON with keys y_true, xgb_scores, cnn_scores, bayes_scores.",
    )
    parser.add_argument("--step", type=float, default=0.1, help="Grid step size.")
    args = parser.parse_args(argv)

    with open(args.validation_json) as fh:
        data = json.load(fh)

    result = optimize_weights(
        data["y_true"],
        data["xgb_scores"],
        data["cnn_scores"],
        data["bayes_scores"],
        step=args.step,
    )
    print(format_weight_result(result))
    return 0 if result.flag == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
