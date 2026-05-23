"""Score cadence irregularity in a time array.

Quantifies gap jitter and outlier gaps to flag data with unstable sampling
that can mimic transit-like features.  Distinct from ``sector_gap_finder``
(inter-sector gaps only) and ``data_gap_interpolator`` (fills gaps).

Public API
----------
CadenceIrregularityResult(n_cadences, median_gap_min, gap_std_min,
                           irregularity_score, n_outlier_gaps, flag)
score_cadence_irregularity(time, *, expected_cadence_min,
                            outlier_sigma) -> CadenceIrregularityResult
format_cadence_irregularity_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CadenceIrregularityResult:
    n_cadences: int
    median_gap_min: float | None    # median inter-point spacing (minutes)
    gap_std_min: float | None       # standard deviation of gaps (minutes)
    irregularity_score: float | None  # gap_std / median_gap (0 = perfect)
    n_outlier_gaps: int             # gaps > outlier_sigma × median
    flag: str  # "OK" | "INSUFFICIENT" | "INVALID"


def _median(vals: list[float]) -> float:
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def score_cadence_irregularity(
    time: list[float],
    *,
    expected_cadence_min: float | None = None,
    outlier_sigma: float = 5.0,
) -> CadenceIrregularityResult:
    """Score the cadence irregularity of a time array.

    Args:
        time: Time array (days), sorted ascending.
        expected_cadence_min: Expected inter-point spacing (minutes).
            If ``None``, the median gap is used as the reference.
        outlier_sigma: Threshold (in units of median gap) for flagging
            a gap as an outlier.

    Returns:
        :class:`CadenceIrregularityResult`.
    """
    n = len(time)
    if n < 3:
        return CadenceIrregularityResult(n, None, None, None, 0, "INSUFFICIENT")

    gaps_days = [time[i + 1] - time[i] for i in range(n - 1)]
    if any(g < 0 for g in gaps_days):
        return CadenceIrregularityResult(n, None, None, None, 0, "INVALID")

    gaps_min = [g * 1440.0 for g in gaps_days]
    med = _median(gaps_min)

    if med <= 0:
        return CadenceIrregularityResult(n, 0.0, None, None, 0, "INVALID")

    mean = sum(gaps_min) / len(gaps_min)
    variance = sum((g - mean) ** 2 for g in gaps_min) / len(gaps_min)
    std = math.sqrt(variance)

    irr_score = round(std / med, 6)

    threshold = (expected_cadence_min if expected_cadence_min is not None else med) * outlier_sigma
    n_outliers = sum(1 for g in gaps_min if g > threshold)

    return CadenceIrregularityResult(
        n_cadences=n,
        median_gap_min=round(med, 4),
        gap_std_min=round(std, 4),
        irregularity_score=irr_score,
        n_outlier_gaps=n_outliers,
        flag="OK",
    )


def format_cadence_irregularity_result(result: CadenceIrregularityResult) -> str:
    """Format cadence irregularity result as Markdown."""
    lines = [
        "## Cadence Irregularity Scorer",
        "",
        f"- Cadences: {result.n_cadences}",
        f"- Median gap: {result.median_gap_min} min",
        f"- Gap std: {result.gap_std_min} min",
        f"- **Irregularity score (std/median): {result.irregularity_score}**",
        f"- Outlier gaps: {result.n_outlier_gaps}",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="cadence_irregularity_scorer",
        description="Score cadence irregularity of a time array.",
    )
    parser.add_argument("--expected-cadence-min", type=float, default=None)
    args = parser.parse_args(argv)

    result = score_cadence_irregularity([], expected_cadence_min=args.expected_cadence_min)
    print(format_cadence_irregularity_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
