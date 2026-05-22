"""Aggregate multiple false-positive diagnostic scores into a single FP probability.

Takes individual vetting scores (odd/even, secondary eclipse, centroid offset,
contamination, etc.) and combines them via a weighted geometric mean approach
into a composite FP probability in [0, 1].

Public API
----------
FPDiagnostic(name, score, weight, direction)
FPAggregateResult(n_diagnostics, n_active, composite_fp_prob,
                  dominant_diagnostic, flag)
aggregate_fp_scores(diagnostics, *, fp_threshold) -> FPAggregateResult
format_fp_aggregate_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FPDiagnostic:
    name: str
    score: float | None   # 0–1 raw diagnostic score (None = not run)
    weight: float         # relative weight for aggregation
    direction: str        # "fp_if_high" | "fp_if_low"


@dataclass(frozen=True)
class FPAggregateResult:
    n_diagnostics: int
    n_active: int               # diagnostics with non-None scores
    composite_fp_prob: float    # combined FP probability in [0, 1]
    dominant_diagnostic: str | None  # name of the highest-weight active diagnostic
    flag: str  # "OK" | "INSUFFICIENT" | "INVALID"


def _to_fp_score(score: float, direction: str) -> float:
    """Map a raw score to [0,1] where high = more FP-like."""
    if direction == "fp_if_high":
        return score
    return 1.0 - score


def aggregate_fp_scores(
    diagnostics: list[FPDiagnostic],
    *,
    fp_threshold: float = 0.5,
) -> FPAggregateResult:
    """Aggregate vetting diagnostics into a composite FP probability.

    Args:
        diagnostics: List of individual diagnostic scores.
        fp_threshold: FP probability above which the candidate is flagged.

    Returns:
        :class:`FPAggregateResult`.
    """
    if not diagnostics:
        return FPAggregateResult(0, 0, 0.0, None, "INVALID")

    for d in diagnostics:
        if d.weight < 0:
            return FPAggregateResult(len(diagnostics), 0, 0.0, None, "INVALID")
        if d.score is not None and not (0.0 <= d.score <= 1.0):
            return FPAggregateResult(len(diagnostics), 0, 0.0, None, "INVALID")

    active = [d for d in diagnostics if d.score is not None]
    if not active:
        return FPAggregateResult(len(diagnostics), 0, 0.5, None, "INSUFFICIENT")

    total_weight = sum(d.weight for d in active)
    if total_weight < 1e-12:
        return FPAggregateResult(len(diagnostics), len(active), 0.5, None, "INSUFFICIENT")

    # Weighted geometric mean in log space
    log_sum = 0.0
    for d in active:
        assert d.score is not None
        fp_s = _to_fp_score(d.score, d.direction)
        # Clamp to avoid log(0)
        fp_s = max(1e-9, min(1.0 - 1e-9, fp_s))
        log_sum += (d.weight / total_weight) * math.log(fp_s)

    composite = math.exp(log_sum)
    composite = max(0.0, min(1.0, composite))

    dominant = max(active, key=lambda d: d.weight).name

    return FPAggregateResult(
        n_diagnostics=len(diagnostics),
        n_active=len(active),
        composite_fp_prob=round(composite, 6),
        dominant_diagnostic=dominant,
        flag="OK",
    )


def format_fp_aggregate_result(result: FPAggregateResult) -> str:
    """Format FP aggregate result as Markdown."""
    lines = [
        "## False-Positive Score Aggregator",
        "",
        f"- Diagnostics: {result.n_diagnostics} total, {result.n_active} active",
        f"- **Composite FP probability: {result.composite_fp_prob:.4f}**",
        f"- Dominant diagnostic: {result.dominant_diagnostic}",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="false_positive_score_aggregator",
        description="Aggregate vetting diagnostic scores into a composite FP probability.",
    )
    parser.add_argument("--fp-threshold", type=float, default=0.5)
    args = parser.parse_args(argv)

    result = aggregate_fp_scores([], fp_threshold=args.fp_threshold)
    print(format_fp_aggregate_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
