"""Precision-recall curve, AUC, and optimal F1 threshold.

Stdlib-only implementation.  Useful for evaluating classifiers on imbalanced
datasets where ROC AUC can be overly optimistic.

Public API
----------
PrAucResult(auc, precision, recall, thresholds, optimal_threshold, optimal_f1,
            optimal_precision, optimal_recall, n_positive, n_negative, flag)
compute_pr_auc(y_true, y_score) -> PrAucResult
format_pr_auc(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PrAucResult:
    auc: float
    precision: tuple[float, ...]
    recall: tuple[float, ...]
    thresholds: tuple[float, ...]
    optimal_threshold: float
    optimal_f1: float
    optimal_precision: float
    optimal_recall: float
    n_positive: int
    n_negative: int
    flag: str   # "OK" | "DEGENERATE" | "INVALID"


def compute_pr_auc(y_true: list[int], y_score: list[float]) -> PrAucResult:
    """Compute the precision-recall curve, AUC, and optimal F1 threshold.

    The curve is built by sweeping a threshold from high to low.  A final
    point at (recall=0, precision=1) is prepended (standard sklearn convention).
    AUC is calculated via the trapezoidal rule over (recall, precision).

    Args:
        y_true: Binary ground-truth labels (0 or 1).
        y_score: Predicted probability / score for the positive class.

    Returns:
        :class:`PrAucResult`.
    """
    _INVALID = PrAucResult(
        auc=0.0, precision=(), recall=(), thresholds=(),
        optimal_threshold=0.0, optimal_f1=0.0,
        optimal_precision=0.0, optimal_recall=0.0,
        n_positive=0, n_negative=0, flag="INVALID",
    )

    if len(y_true) == 0 or len(y_true) != len(y_score):
        return _INVALID

    n_positive = sum(1 for lbl in y_true if lbl == 1)
    n_negative = sum(1 for lbl in y_true if lbl == 0)

    if n_positive == 0 or n_negative == 0:
        return PrAucResult(
            auc=0.0, precision=(), recall=(), thresholds=(),
            optimal_threshold=0.0, optimal_f1=0.0,
            optimal_precision=0.0, optimal_recall=0.0,
            n_positive=n_positive, n_negative=n_negative,
            flag="DEGENERATE",
        )

    # Sort by descending score
    pairs = sorted(zip(y_score, y_true, strict=True), key=lambda x: -x[0])

    tp = 0
    fp = 0
    # Start with prepended point: recall=0, precision=1
    precision_pts: list[float] = [1.0]
    recall_pts: list[float] = [0.0]
    thresh_pts: list[float] = []

    prev_score: float | None = None
    for score, label in pairs:
        if score != prev_score and prev_score is not None:
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / n_positive
            precision_pts.append(prec)
            recall_pts.append(rec)
            thresh_pts.append(score)
        if label == 1:
            tp += 1
        else:
            fp += 1
        prev_score = score

    # final point at the lowest threshold
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / n_positive
    precision_pts.append(prec)
    recall_pts.append(rec)
    thresh_pts.append(pairs[-1][0] if pairs else 0.0)

    # trapezoidal AUC over (recall, precision)
    auc = 0.0
    for i in range(len(recall_pts) - 1):
        dr = recall_pts[i + 1] - recall_pts[i]
        avg_p = 0.5 * (precision_pts[i + 1] + precision_pts[i])
        auc += dr * avg_p

    # optimal threshold: maximise F1 over (threshold, precision, recall) triples
    # We compare points corresponding to thresh_pts (len = len(precision_pts) - 1)
    opt_f1 = -1.0
    opt_thresh = thresh_pts[0] if thresh_pts else 0.0
    opt_prec = 0.0
    opt_rec = 0.0

    for idx, thresh in enumerate(thresh_pts):
        p = precision_pts[idx + 1]   # +1 because precision_pts[0] is the prepended point
        r = recall_pts[idx + 1]
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
        if f1 > opt_f1:
            opt_f1 = f1
            opt_thresh = thresh
            opt_prec = p
            opt_rec = r

    return PrAucResult(
        auc=round(auc, 8),
        precision=tuple(precision_pts),
        recall=tuple(recall_pts),
        thresholds=tuple(thresh_pts),
        optimal_threshold=opt_thresh,
        optimal_f1=max(opt_f1, 0.0),
        optimal_precision=opt_prec,
        optimal_recall=opt_rec,
        n_positive=n_positive,
        n_negative=n_negative,
        flag="OK",
    )


def format_pr_auc(result: PrAucResult) -> str:
    """Format a :class:`PrAucResult` as a Markdown string."""
    lines = [
        "## Precision-Recall AUC Calculator",
        "",
        f"- **AUC:** {result.auc:.4f}",
        f"- **Optimal threshold:** {result.optimal_threshold:.4f}",
        f"- **Optimal F1:** {result.optimal_f1:.4f}",
        f"- **Optimal precision:** {result.optimal_precision:.4f}",
        f"- **Optimal recall:** {result.optimal_recall:.4f}",
        f"- **n_positive:** {result.n_positive}",
        f"- **n_negative:** {result.n_negative}",
        f"- **Curve points:** {len(result.precision)}",
        f"- **Flag:** {result.flag}",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="pr_auc_calculator",
        description="Compute PR AUC from a JSON predictions file.",
    )
    parser.add_argument(
        "input",
        help='JSON file with list of {"label": 0|1, "score": float} dicts.',
    )
    args = parser.parse_args(argv)

    with open(args.input) as fh:  # noqa: PTH123
        data = json.load(fh)

    y_true = [int(d["label"]) for d in data]
    y_score = [float(d["score"]) for d in data]

    result = compute_pr_auc(y_true, y_score)
    print(format_pr_auc(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
