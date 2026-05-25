"""Canonical stdlib-only ROC curve and trapezoidal AUC calculator.

Computes the full ROC curve (FPR, TPR, thresholds) and the area under it
using the trapezoidal rule.  No numpy or sklearn required.

Public API
----------
RocAucResult(auc, fpr, tpr, thresholds, n_positive, n_negative, flag)
compute_roc_auc(y_true, y_score) -> RocAucResult
format_roc_auc(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RocAucResult:
    auc: float
    fpr: tuple[float, ...]
    tpr: tuple[float, ...]
    thresholds: tuple[float, ...]
    n_positive: int
    n_negative: int
    flag: str   # "OK" | "DEGENERATE" | "INVALID"


def compute_roc_auc(y_true: list[int], y_score: list[float]) -> RocAucResult:
    """Compute the ROC curve and AUC via the trapezoidal rule.

    Curve points are computed by sweeping a threshold from high to low.
    The final curve is bookended with (0, 0) prepended and (1, 1) appended.

    Args:
        y_true: Binary ground-truth labels (0 or 1).
        y_score: Predicted probability / score for the positive class.

    Returns:
        :class:`RocAucResult`.
    """
    _INVALID = RocAucResult(
        auc=0.0, fpr=(), tpr=(), thresholds=(),
        n_positive=0, n_negative=0, flag="INVALID",
    )

    if len(y_true) == 0 or len(y_true) != len(y_score):
        return _INVALID

    n_positive = sum(1 for lbl in y_true if lbl == 1)
    n_negative = sum(1 for lbl in y_true if lbl == 0)

    if n_positive == 0 or n_negative == 0:
        return RocAucResult(
            auc=0.0, fpr=(), tpr=(), thresholds=(),
            n_positive=n_positive, n_negative=n_negative, flag="DEGENERATE",
        )

    # Sort by descending score
    pairs = sorted(zip(y_score, y_true, strict=True), key=lambda x: -x[0])

    tp = 0
    fp = 0
    fpr_pts: list[float] = [0.0]
    tpr_pts: list[float] = [0.0]
    thresh_pts: list[float] = []

    prev_score: float | None = None
    for score, label in pairs:
        if score != prev_score and prev_score is not None:
            fpr_pts.append(fp / n_negative)
            tpr_pts.append(tp / n_positive)
            thresh_pts.append(score)
        if label == 1:
            tp += 1
        else:
            fp += 1
        prev_score = score

    # final point at (1, 1)
    fpr_pts.append(fp / n_negative)
    tpr_pts.append(tp / n_positive)
    thresh_pts.append(pairs[-1][0] if pairs else 0.0)

    # trapezoidal AUC
    auc = sum(
        0.5 * (fpr_pts[i + 1] - fpr_pts[i]) * (tpr_pts[i + 1] + tpr_pts[i])
        for i in range(len(fpr_pts) - 1)
    )

    return RocAucResult(
        auc=round(auc, 8),
        fpr=tuple(fpr_pts),
        tpr=tuple(tpr_pts),
        thresholds=tuple(thresh_pts),
        n_positive=n_positive,
        n_negative=n_negative,
        flag="OK",
    )


def format_roc_auc(result: RocAucResult) -> str:
    """Format a :class:`RocAucResult` as a Markdown string."""
    lines = [
        "## ROC AUC Calculator",
        "",
        f"- **AUC:** {result.auc:.4f}",
        f"- **n_positive:** {result.n_positive}",
        f"- **n_negative:** {result.n_negative}",
        f"- **Curve points:** {len(result.fpr)}",
        f"- **Flag:** {result.flag}",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="roc_auc_calculator",
        description="Compute ROC AUC from a JSON predictions file.",
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

    result = compute_roc_auc(y_true, y_score)
    print(format_roc_auc(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
