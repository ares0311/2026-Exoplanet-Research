"""Estimate the false-negative rate (missed detections) at a score threshold.

Computes the Type II error rate: fraction of true positives (real planets)
that fall below a detection threshold and would be missed.  Complements
``false_alarm_probability_estimator`` (Type I error) with the Type II
perspective needed for completeness-corrected occurrence rates.

Public API
----------
FNRResult(threshold, n_positives, n_negatives, n_missed, n_caught,
          false_negative_rate, true_positive_rate, flag)
estimate_false_negative_rate(scores, labels, *,
                             threshold) -> FNRResult
format_fnr_result(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FNRResult:
    threshold: float
    n_positives: int          # total true positives (label == 1)
    n_negatives: int          # total true negatives (label == 0)
    n_missed: int             # true positives below threshold
    n_caught: int             # true positives at or above threshold
    false_negative_rate: float | None   # n_missed / n_positives
    true_positive_rate: float | None    # n_caught / n_positives (recall)
    flag: str  # "OK" | "INSUFFICIENT" | "INVALID"


def estimate_false_negative_rate(
    scores: list[float],
    labels: list[int],
    *,
    threshold: float = 0.5,
) -> FNRResult:
    """Compute false-negative rate at *threshold*.

    Args:
        scores: Model scores in [0, 1] (higher = more likely planet).
        labels: Binary ground-truth labels (1 = planet, 0 = false positive).
        threshold: Score threshold; signals below this are classified as FP.

    Returns:
        :class:`FNRResult`.
    """
    if len(scores) != len(labels):
        return FNRResult(threshold, 0, 0, 0, 0, None, None, "INVALID")
    if not scores:
        return FNRResult(threshold, 0, 0, 0, 0, None, None, "INSUFFICIENT")

    n_pos = sum(1 for lb in labels if lb == 1)
    n_neg = sum(1 for lb in labels if lb == 0)

    if n_pos == 0:
        return FNRResult(threshold, 0, n_neg, 0, 0, None, None, "INSUFFICIENT")

    n_missed = sum(
        1 for s, lb in zip(scores, labels, strict=False) if lb == 1 and s < threshold
    )
    n_caught = n_pos - n_missed
    fnr = round(n_missed / n_pos, 6)
    tpr = round(n_caught / n_pos, 6)

    return FNRResult(
        threshold=threshold,
        n_positives=n_pos,
        n_negatives=n_neg,
        n_missed=n_missed,
        n_caught=n_caught,
        false_negative_rate=fnr,
        true_positive_rate=tpr,
        flag="OK",
    )


def format_fnr_result(result: FNRResult) -> str:
    """Format FNR result as Markdown."""
    lines = [
        "## False-Negative Rate Estimator",
        "",
        f"- Score threshold: {result.threshold}",
        f"- True positives: {result.n_positives}",
        f"- True negatives: {result.n_negatives}",
        f"- Missed (FN): {result.n_missed}",
        f"- Caught (TP): {result.n_caught}",
        f"- **False-negative rate: {result.false_negative_rate}**",
        f"- True-positive rate (recall): {result.true_positive_rate}",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="false_negative_rate_estimator",
        description="Estimate false-negative rate at a score threshold.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args(argv)

    result = estimate_false_negative_rate([], [], threshold=args.threshold)
    print(format_fnr_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
