"""
Checks whether a known list of candidates (period, epoch) is recovered in a list of detected signals.

Public API:
    RecoveryCheckResult   -- frozen dataclass holding recovery statistics
    check_recovery(known, detected, period_tol_frac, epoch_tol_days) -> RecoveryCheckResult
    format_recovery_check(result) -> str
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RecoveryCheckResult:
    n_known: int
    n_recovered: int
    n_missed: int
    recovered_indices: tuple[int, ...]
    missed_indices: tuple[int, ...]
    flag: str


def check_recovery(
    known: list[dict],
    detected: list[dict],
    period_tol_frac: float = 0.01,
    epoch_tol_days: float = 0.1,
) -> RecoveryCheckResult:
    recovered: list[int] = []
    missed: list[int] = []

    for i, k in enumerate(known):
        k_period = float(k["period_days"])
        k_epoch = float(k["epoch_bjd"])
        found = False
        for d in detected:
            d_period = float(d["period_days"])
            d_epoch = float(d["epoch_bjd"])
            period_match = abs(d_period - k_period) <= period_tol_frac * k_period
            # Epoch agreement checked modulo the known period to handle phase shifts
            if period_match and k_period > 0.0:
                phase_diff = abs(d_epoch - k_epoch) % k_period
                # Wrap to [-P/2, P/2]
                if phase_diff > k_period / 2.0:
                    phase_diff = k_period - phase_diff
                epoch_match = phase_diff <= epoch_tol_days
            else:
                epoch_match = abs(d_epoch - k_epoch) <= epoch_tol_days
            if period_match and epoch_match:
                found = True
                break
        if found:
            recovered.append(i)
        else:
            missed.append(i)

    n_missed = len(missed)
    flag = "INCOMPLETE_RECOVERY" if n_missed > 0 else "OK"

    return RecoveryCheckResult(
        n_known=len(known),
        n_recovered=len(recovered),
        n_missed=n_missed,
        recovered_indices=tuple(recovered),
        missed_indices=tuple(missed),
        flag=flag,
    )


def format_recovery_check(result: RecoveryCheckResult) -> str:
    recovered_str = ", ".join(str(i) for i in result.recovered_indices) or "none"
    missed_str = ", ".join(str(i) for i in result.missed_indices) or "none"
    lines = [
        "## Candidate Recovery Check",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Known candidates | {result.n_known} |",
        f"| Recovered | {result.n_recovered} |",
        f"| Missed | {result.n_missed} |",
        f"| Recovered indices | {recovered_str} |",
        f"| Missed indices | {missed_str} |",
        f"| Flag | {result.flag} |",
    ]
    return "\n".join(lines)


def _cli() -> int:
    parser = argparse.ArgumentParser(
        description="Check recovery of known candidates in detected signals."
    )
    parser.add_argument("known_file", help="JSON file with list of known {period_days, epoch_bjd} dicts")
    parser.add_argument("detected_file", help="JSON file with list of detected {period_days, epoch_bjd} dicts")
    parser.add_argument(
        "--period-tol-frac",
        type=float,
        default=0.01,
        help="Fractional period tolerance (default 0.01)",
    )
    parser.add_argument(
        "--epoch-tol-days",
        type=float,
        default=0.1,
        help="Epoch tolerance in days (default 0.1)",
    )
    args = parser.parse_args()

    with open(args.known_file) as fh:
        known = json.load(fh)
    with open(args.detected_file) as fh:
        detected = json.load(fh)

    result = check_recovery(known, detected, args.period_tol_frac, args.epoch_tol_days)
    print(format_recovery_check(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
