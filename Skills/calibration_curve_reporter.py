"""Reliability diagram data and Brier score for model calibration assessment.

Computes bin-by-bin mean predicted probability vs. fraction positive
(the reliability diagram) plus the overall Brier score.  Does NOT
import matplotlib — returns raw data that callers can plot themselves.

Public API
----------
CalibrationCurveResult(bin_edges, mean_pred_prob, fraction_positive,
                       bin_counts, brier_score, n_bins, flag)
compute_calibration_curve(y_true, y_score, *, n_bins) -> CalibrationCurveResult
format_calibration_curve(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CalibrationCurveResult:
    bin_edges: tuple[float, ...]       # length n_bins + 1
    mean_pred_prob: tuple[float, ...]  # length n_bins; NaN for empty bins
    fraction_positive: tuple[float, ...]  # length n_bins; NaN for empty bins
    bin_counts: tuple[int, ...]        # length n_bins
    brier_score: float
    n_bins: int
    flag: str  # "OK" | "INSUFFICIENT" | "INVALID"


def compute_calibration_curve(
    y_true: list[int],
    y_score: list[float],
    *,
    n_bins: int = 10,
) -> CalibrationCurveResult:
    """Compute reliability diagram data and Brier score.

    Args:
        y_true: Binary labels (0/1).
        y_score: Predicted probabilities in [0, 1].
        n_bins: Number of equal-width bins spanning [0, 1].

    Returns:
        CalibrationCurveResult with per-bin statistics and overall Brier score.
    """
    _nan = float("nan")

    if not y_true or len(y_true) != len(y_score):
        edges = tuple(i / n_bins for i in range(n_bins + 1))
        return CalibrationCurveResult(
            bin_edges=edges,
            mean_pred_prob=tuple(_nan for _ in range(n_bins)),
            fraction_positive=tuple(_nan for _ in range(n_bins)),
            bin_counts=tuple(0 for _ in range(n_bins)),
            brier_score=_nan,
            n_bins=n_bins,
            flag="INVALID",
        )

    n_pos = sum(y_true)
    flag = "INSUFFICIENT" if n_pos == 0 or n_pos == len(y_true) else "OK"

    # Brier score
    brier = sum((s - y) ** 2 for s, y in zip(y_score, y_true, strict=True)) / len(y_true)

    # Bin edges: uniform 0 to 1
    edges = tuple(i / n_bins for i in range(n_bins + 1))

    bin_scores: list[list[float]] = [[] for _ in range(n_bins)]
    bin_labels: list[list[int]] = [[] for _ in range(n_bins)]

    for score, label in zip(y_score, y_true, strict=True):
        # Clamp to [0, 1)
        idx = min(int(score * n_bins), n_bins - 1)
        bin_scores[idx].append(score)
        bin_labels[idx].append(label)

    mean_pred: list[float] = []
    frac_pos: list[float] = []
    counts: list[int] = []

    for bscores, blabels in zip(bin_scores, bin_labels, strict=True):
        if not bscores:
            mean_pred.append(_nan)
            frac_pos.append(_nan)
            counts.append(0)
        else:
            mean_pred.append(sum(bscores) / len(bscores))
            frac_pos.append(sum(blabels) / len(blabels))
            counts.append(len(bscores))

    return CalibrationCurveResult(
        bin_edges=edges,
        mean_pred_prob=tuple(mean_pred),
        fraction_positive=tuple(frac_pos),
        bin_counts=tuple(counts),
        brier_score=brier,
        n_bins=n_bins,
        flag=flag,
    )


def format_calibration_curve(result: CalibrationCurveResult) -> str:
    """Format calibration curve result as a Markdown report.

    Args:
        result: CalibrationCurveResult to format.

    Returns:
        Markdown string with reliability table and Brier score.
    """
    lines: list[str] = [
        "## Calibration Curve (Reliability Diagram)\n",
        f"Flag: `{result.flag}` | Bins: {result.n_bins}\n",
    ]

    if result.flag == "INVALID":
        lines.append("\n_INVALID: empty or mismatched input._\n")
        return "\n".join(lines)

    brier_str = (
        f"{result.brier_score:.4f}"
        if not math.isnan(result.brier_score)
        else "—"
    )
    lines.append(f"**Brier Score**: {brier_str}\n")

    lines.append("")
    lines.append("| Bin | Mean Pred | Fraction Pos | Count |")
    lines.append("|-----|-----------|--------------|-------|")
    for i in range(result.n_bins):
        lo = result.bin_edges[i]
        hi = result.bin_edges[i + 1]
        count = result.bin_counts[i]
        mp = result.mean_pred_prob[i]
        fp = result.fraction_positive[i]
        mp_str = f"{mp:.3f}" if not math.isnan(mp) else "—"
        fp_str = f"{fp:.3f}" if not math.isnan(fp) else "—"
        lines.append(f"| [{lo:.1f}, {hi:.1f}) | {mp_str} | {fp_str} | {count} |")

    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Compute reliability diagram data and Brier score."
    )
    parser.add_argument(
        "data_json",
        help="Path to JSON with keys y_true and y_score.",
    )
    parser.add_argument("--n-bins", type=int, default=10, help="Number of bins.")
    args = parser.parse_args(argv)

    with open(args.data_json) as fh:
        data = json.load(fh)

    result = compute_calibration_curve(
        data["y_true"], data["y_score"], n_bins=args.n_bins
    )
    print(format_calibration_curve(result))
    return 0 if result.flag in ("OK", "INSUFFICIENT") else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
