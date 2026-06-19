"""Find the optimal classification threshold on a CNN validation set.

Sweeps thresholds from 0 to 1 and identifies the operating point that
maximises a configurable objective (F1, balanced accuracy, or Youden's J).

Public API
----------
ThresholdResult(threshold, precision, recall, f1, balanced_accuracy,
                youden_j, n_positive, n_negative, objective, flag)
optimize_threshold(y_true, y_score, *, objective, n_steps) -> ThresholdResult
format_threshold_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    precision: float
    recall: float
    f1: float
    balanced_accuracy: float
    youden_j: float         # TPR - FPR
    n_positive: int
    n_negative: int
    objective: str          # "f1" | "balanced_accuracy" | "youden_j"
    flag: str  # "OK" | "DEGENERATE" | "INVALID"


def _metrics_at(
    y_true: list[int],
    y_score: list[float],
    threshold: float,
) -> tuple[float, float, float, float, float]:
    """Return (precision, recall, f1, balanced_accuracy, youden_j) at threshold."""
    tp = fp = tn = fn = 0
    for yt, ys in zip(y_true, y_score, strict=True):
        pred = 1 if ys >= threshold else 0
        if pred == 1 and yt == 1:
            tp += 1
        elif pred == 1 and yt == 0:
            fp += 1
        elif pred == 0 and yt == 0:
            tn += 1
        else:
            fn += 1

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    tpr = rec
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    ba = (tpr + (1 - fpr)) / 2.0
    youden = tpr - fpr
    return prec, rec, f1, ba, youden


def _valid_inputs(
    y_true: list[int],
    y_score: list[float],
    objective: str,
    n_steps: int,
) -> bool:
    """Return True when threshold search inputs are safe to optimize."""
    if not y_true or len(y_true) != len(y_score):
        return False
    if objective not in {"f1", "balanced_accuracy", "youden_j"}:
        return False
    if n_steps <= 0:
        return False
    for label, score in zip(y_true, y_score, strict=True):
        if not isinstance(label, int) or isinstance(label, bool) or label not in (0, 1):
            return False
        if (
            not isinstance(score, int | float)
            or isinstance(score, bool)
            or not math.isfinite(float(score))
            or not 0.0 <= float(score) <= 1.0
        ):
            return False
    return True


def optimize_threshold(
    y_true: list[int],
    y_score: list[float],
    *,
    objective: str = "f1",
    n_steps: int = 100,
) -> ThresholdResult:
    """Find the threshold that maximises the chosen objective on validation data.

    Args:
        y_true: Binary labels (0/1).
        y_score: Predicted scores in [0, 1].
        objective: Metric to maximise — ``"f1"``, ``"balanced_accuracy"``,
            or ``"youden_j"``.
        n_steps: Number of threshold grid points to evaluate.

    Returns:
        ThresholdResult at the optimal threshold.
    """
    if not _valid_inputs(y_true, y_score, objective, n_steps):
        return ThresholdResult(
            threshold=0.5, precision=0.0, recall=0.0, f1=0.0,
            balanced_accuracy=0.0, youden_j=0.0, n_positive=0, n_negative=0,
            objective=objective, flag="INVALID",
        )

    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return ThresholdResult(
            threshold=0.5, precision=0.0, recall=0.0, f1=0.0,
            balanced_accuracy=0.5, youden_j=0.0,
            n_positive=n_pos, n_negative=n_neg,
            objective=objective, flag="DEGENERATE",
        )

    obj_idx = {"f1": 2, "balanced_accuracy": 3, "youden_j": 4}[objective]

    best_thresh = 0.5
    best_score = -1.0
    best_metrics = (0.0, 0.0, 0.0, 0.0, 0.0)

    for i in range(n_steps + 1):
        t = i / n_steps
        metrics = _metrics_at(y_true, y_score, t)
        score = metrics[obj_idx]
        if score > best_score:
            best_score = score
            best_thresh = t
            best_metrics = metrics

    prec, rec, f1, ba, youden = best_metrics
    return ThresholdResult(
        threshold=best_thresh,
        precision=prec,
        recall=rec,
        f1=f1,
        balanced_accuracy=ba,
        youden_j=youden,
        n_positive=n_pos,
        n_negative=n_neg,
        objective=objective,
        flag="OK",
    )


def format_threshold_result(result: ThresholdResult) -> str:
    """Format a Markdown threshold optimisation result.

    Args:
        result: ThresholdResult to format.

    Returns:
        Markdown string.
    """
    lines = [
        "## CNN Threshold Optimisation\n",
        f"Flag: `{result.flag}` | Objective: `{result.objective}`\n",
    ]
    if result.flag in ("INVALID", "DEGENERATE"):
        lines.append(f"\n_{result.flag}: cannot optimise threshold._\n")
        return "\n".join(lines)

    lines += [
        f"**Optimal threshold**: {result.threshold:.3f}\n",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Precision | {result.precision:.4f} |",
        f"| Recall | {result.recall:.4f} |",
        f"| F1 | {result.f1:.4f} |",
        f"| Balanced Accuracy | {result.balanced_accuracy:.4f} |",
        f"| Youden's J | {result.youden_j:.4f} |",
        f"| N Positive | {result.n_positive} |",
        f"| N Negative | {result.n_negative} |",
    ]
    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Optimise CNN classification threshold.")
    parser.add_argument("predictions_json",
                        help="JSON file with y_true and y_score arrays.")
    parser.add_argument("--objective", default="f1",
                        choices=["f1", "balanced_accuracy", "youden_j"])
    parser.add_argument("--n-steps", type=int, default=100)
    args = parser.parse_args(argv)

    data = json.loads(Path(args.predictions_json).read_text())
    result = optimize_threshold(data["y_true"], data["y_score"],
                                objective=args.objective, n_steps=args.n_steps)
    print(format_threshold_result(result))
    return 0 if result.flag == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
