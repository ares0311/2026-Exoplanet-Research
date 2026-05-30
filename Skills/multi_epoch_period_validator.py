"""Cross-validate a period estimate across two or more observing seasons/epochs.

Public API:
    EpochValidationResult  -- frozen dataclass
    validate_period_multi_epoch(period_days, epoch_lists, *, tolerance_minutes)
        -> EpochValidationResult
    format_epoch_validation_result(result) -> str
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class EpochValidationResult:
    period_days: float
    n_epochs: int
    n_transits_total: int
    rms_oc_minutes: float
    max_oc_minutes: float
    consistent: bool
    flag: str


def validate_period_multi_epoch(
    period_days: float,
    epoch_lists: list[list[float]],
    *,
    tolerance_minutes: float = 15.0,
) -> EpochValidationResult:
    """
    epoch_lists: list of lists of BJD transit midpoints, one list per season.
    Flattens all midpoints, computes O-C residuals against the first midpoint as epoch.
    """
    if period_days <= 0:
        return EpochValidationResult(
            period_days=period_days, n_epochs=0, n_transits_total=0,
            rms_oc_minutes=0.0, max_oc_minutes=0.0, consistent=False, flag="INVALID_PERIOD",
        )
    if len(epoch_lists) < 2:
        return EpochValidationResult(
            period_days=period_days, n_epochs=len(epoch_lists), n_transits_total=0,
            rms_oc_minutes=0.0, max_oc_minutes=0.0, consistent=False, flag="INSUFFICIENT_EPOCHS",
        )
    all_midpoints = [t for season in epoch_lists for t in season]
    n_total = len(all_midpoints)
    if n_total < 2:
        return EpochValidationResult(
            period_days=period_days, n_epochs=len(epoch_lists), n_transits_total=n_total,
            rms_oc_minutes=0.0, max_oc_minutes=0.0, consistent=False, flag="INSUFFICIENT_TRANSITS",
        )
    epoch0 = min(all_midpoints)
    residuals: list[float] = []
    for t in all_midpoints:
        n_i = round((t - epoch0) / period_days)
        predicted = epoch0 + n_i * period_days
        residuals.append((t - predicted) * 24.0 * 60.0)  # minutes
    mean_res = sum(residuals) / len(residuals)
    rms = math.sqrt(sum((r - mean_res) ** 2 for r in residuals) / len(residuals))
    max_oc = max(abs(r) for r in residuals)
    consistent = rms <= tolerance_minutes
    flag = "CONSISTENT" if consistent else "INCONSISTENT"
    return EpochValidationResult(
        period_days=period_days,
        n_epochs=len(epoch_lists),
        n_transits_total=n_total,
        rms_oc_minutes=rms,
        max_oc_minutes=max_oc,
        consistent=consistent,
        flag=flag,
    )


def format_epoch_validation_result(result: EpochValidationResult) -> str:
    lines = [
        "## Multi-Epoch Period Validation",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Period (days) | {result.period_days:.6f} |",
        f"| N Epochs | {result.n_epochs} |",
        f"| N Transits Total | {result.n_transits_total} |",
        f"| RMS O-C (min) | {result.rms_oc_minutes:.2f} |",
        f"| Max O-C (min) | {result.max_oc_minutes:.2f} |",
        f"| Consistent | {result.consistent} |",
        f"| Flag | {result.flag} |",
    ]
    return "\n".join(lines)


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Cross-validate period across observing epochs.")
    parser.add_argument("period_days", type=float)
    parser.add_argument("--epoch", nargs="+", type=float, action="append", dest="epochs",
                        default=[], metavar="BJD")
    parser.add_argument("--tolerance", type=float, default=15.0)
    args = parser.parse_args()
    result = validate_period_multi_epoch(args.period_days, args.epochs,
                                          tolerance_minutes=args.tolerance)
    print(format_epoch_validation_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
