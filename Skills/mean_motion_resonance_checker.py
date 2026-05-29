"""Check if a candidate period is near a mean-motion resonance with known planets.

Checks ratios p:q for p, q in 1..6 between candidate period and each known period.
Reports the nearest resonance ratio (by fractional deviation).

Public API
----------
MMRResult(nearest_ratio_str, nearest_period_days, delta_fraction, is_near_mmr, flag)
check_mmr(candidate_period_days, known_periods_days, *, tolerance) -> MMRResult
format_mmr_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass

_MAX_P = 6  # check ratios up to 6:1
_DEFAULT_TOLERANCE = 0.02


@dataclass(frozen=True)
class MMRResult:
    nearest_ratio_str: str
    nearest_period_days: float
    delta_fraction: float
    is_near_mmr: bool
    flag: str = "OK"


def check_mmr(
    candidate_period_days: float,
    known_periods_days: list[float],
    *,
    tolerance: float = _DEFAULT_TOLERANCE,
) -> MMRResult:
    """Check if candidate period is near a mean-motion resonance.

    Args:
        candidate_period_days: Candidate orbital period in days.
        known_periods_days: List of known planet periods in days.
        tolerance: Fractional tolerance to declare near-MMR (default 0.02).

    Returns:
        :class:`MMRResult`.
    """
    if candidate_period_days <= 0:
        return MMRResult(
            nearest_ratio_str="N/A", nearest_period_days=0.0,
            delta_fraction=1.0, is_near_mmr=False, flag="ERROR",
        )
    if not known_periods_days:
        return MMRResult(
            nearest_ratio_str="N/A", nearest_period_days=0.0,
            delta_fraction=1.0, is_near_mmr=False, flag="WARNING",
        )

    best_delta = math.inf
    best_ratio = "N/A"
    best_period = 0.0

    for p_known in known_periods_days:
        if p_known <= 0:
            continue
        # ratio = P_candidate / P_known = p/q  => P_candidate = P_known * p/q
        for p in range(1, _MAX_P + 1):
            for q in range(1, _MAX_P + 1):
                expected = p_known * p / q
                delta = abs(candidate_period_days - expected) / candidate_period_days
                if delta < best_delta:
                    best_delta = delta
                    best_ratio = f"{p}:{q}"
                    best_period = p_known

    is_near = best_delta < tolerance
    return MMRResult(
        nearest_ratio_str=best_ratio,
        nearest_period_days=round(best_period, 6),
        delta_fraction=round(best_delta, 6),
        is_near_mmr=is_near,
        flag="OK",
    )


def format_mmr_result(result: MMRResult) -> str:
    """Format MMR check result as Markdown."""
    near_str = "Yes" if result.is_near_mmr else "No"
    lines = [
        "## Mean-Motion Resonance Check",
        "",
        f"- Nearest ratio: **{result.nearest_ratio_str}**",
        f"- Reference period: {result.nearest_period_days:.4f} days",
        f"- Fractional deviation: {result.delta_fraction:.4f}",
        f"- Near MMR: **{near_str}**",
        f"- Flag: {result.flag}",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="mean_motion_resonance_checker",
        description="Check if a period is near a mean-motion resonance.",
    )
    parser.add_argument("candidate_period_days", type=float)
    parser.add_argument(
        "--known-periods",
        type=str,
        default="[]",
        help="JSON list of known planet periods in days",
    )
    parser.add_argument("--tolerance", type=float, default=_DEFAULT_TOLERANCE)
    args = parser.parse_args(argv)

    known = json.loads(args.known_periods)
    result = check_mmr(args.candidate_period_days, known, tolerance=args.tolerance)
    print(format_mmr_result(result))
    return 0 if result.flag == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
