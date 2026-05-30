"""Validate that a claimed transit count is consistent with baseline and period.

Public API:
    TransitCountValidation  -- frozen dataclass
    validate_transit_count(n_transits, period_days, baseline_days) -> TransitCountValidation
    format_transit_count_validation(result) -> str
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TransitCountValidation:
    n_transits: int
    expected_min: int
    expected_max: int
    is_consistent: bool
    flag: str


def validate_transit_count(
    n_transits: int,
    period_days: float,
    baseline_days: float,
    *,
    gap_fraction: float = 0.0,
) -> TransitCountValidation:
    if period_days <= 0:
        return TransitCountValidation(
            n_transits=n_transits, expected_min=0, expected_max=0,
            is_consistent=False, flag="INVALID_PERIOD",
        )
    if baseline_days <= 0:
        return TransitCountValidation(
            n_transits=n_transits, expected_min=0, expected_max=0,
            is_consistent=False, flag="INVALID_BASELINE",
        )
    if not 0.0 <= gap_fraction < 1.0:
        return TransitCountValidation(
            n_transits=n_transits, expected_min=0, expected_max=0,
            is_consistent=False, flag="INVALID_GAP_FRACTION",
        )
    max_possible = math.floor(baseline_days / period_days) + 1
    effective = baseline_days * (1.0 - gap_fraction)
    min_expected = max(0, math.floor(effective / period_days))
    is_consistent = min_expected <= n_transits <= max_possible
    flag = "OK" if is_consistent else "INCONSISTENT_COUNT"
    return TransitCountValidation(
        n_transits=n_transits,
        expected_min=min_expected,
        expected_max=max_possible,
        is_consistent=is_consistent,
        flag=flag,
    )


def format_transit_count_validation(result: TransitCountValidation) -> str:
    lines = [
        "## Transit Count Validation",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Claimed N Transits | {result.n_transits} |",
        f"| Expected Min | {result.expected_min} |",
        f"| Expected Max | {result.expected_max} |",
        f"| Consistent | {result.is_consistent} |",
        f"| Flag | {result.flag} |",
    ]
    return "\n".join(lines)


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Validate transit count.")
    parser.add_argument("n_transits", type=int)
    parser.add_argument("period_days", type=float)
    parser.add_argument("baseline_days", type=float)
    parser.add_argument("--gap-fraction", type=float, default=0.0)
    args = parser.parse_args()
    result = validate_transit_count(
        args.n_transits, args.period_days, args.baseline_days,
        gap_fraction=args.gap_fraction,
    )
    print(format_transit_count_validation(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
