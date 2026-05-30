"""Check what fraction of expected transit windows are covered by observations.

For a given orbital period, epoch, and total baseline, this tool counts how
many of the predicted transit windows contain at least one observation and
reports the coverage fraction together with which windows were missed.

Public API
----------
WindowCoverageResult(coverage_fraction, n_windows_observed, n_windows_total,
                     missed_windows, flag)
check_period_window_coverage(observation_times_bjd, period_days, epoch_bjd,
                             total_baseline_days) -> WindowCoverageResult
format_window_coverage(result) -> str
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class WindowCoverageResult:
    coverage_fraction: float
    n_windows_observed: int
    n_windows_total: int
    missed_windows: list[int]
    flag: str  # "OK" | "INVALID_PERIOD" | "INVALID_BASELINE" | "NO_OBSERVATIONS"


def check_period_window_coverage(
    observation_times_bjd: list[float],
    period_days: float,
    epoch_bjd: float,
    total_baseline_days: float,
) -> WindowCoverageResult:
    """Compute the fraction of transit windows covered by observations.

    Parameters
    ----------
    observation_times_bjd:
        List of observation timestamps in BJD.
    period_days:
        Orbital period in days.
    epoch_bjd:
        Reference transit epoch in BJD.
    total_baseline_days:
        Total time baseline over which windows are counted.

    Returns
    -------
    WindowCoverageResult
    """
    if period_days <= 0:
        return WindowCoverageResult(
            coverage_fraction=0.0,
            n_windows_observed=0,
            n_windows_total=0,
            missed_windows=[],
            flag="INVALID_PERIOD",
        )

    if total_baseline_days <= 0:
        return WindowCoverageResult(
            coverage_fraction=0.0,
            n_windows_observed=0,
            n_windows_total=0,
            missed_windows=[],
            flag="INVALID_BASELINE",
        )

    if len(observation_times_bjd) == 0:
        n_windows_total = max(1, int(total_baseline_days / period_days))
        missed_windows = list(range(n_windows_total))
        return WindowCoverageResult(
            coverage_fraction=0.0,
            n_windows_observed=0,
            n_windows_total=n_windows_total,
            missed_windows=missed_windows,
            flag="NO_OBSERVATIONS",
        )

    n_windows_total = max(1, int(total_baseline_days / period_days))
    half_window = period_days / 4.0

    observed_set: set[int] = set()
    for i in range(n_windows_total):
        window_center = epoch_bjd + i * period_days
        for t in observation_times_bjd:
            if abs(t - window_center) <= half_window:
                observed_set.add(i)
                break

    n_windows_observed = len(observed_set)
    missed_windows = [i for i in range(n_windows_total) if i not in observed_set]
    coverage_fraction = n_windows_observed / n_windows_total

    return WindowCoverageResult(
        coverage_fraction=coverage_fraction,
        n_windows_observed=n_windows_observed,
        n_windows_total=n_windows_total,
        missed_windows=missed_windows,
        flag="OK",
    )


def format_window_coverage(result: WindowCoverageResult) -> str:
    """Return a Markdown table summarising the window coverage result."""
    missed_str = str(result.missed_windows) if result.missed_windows else "none"
    lines = [
        "| Metric | Value |",
        "| --- | --- |",
        f"| Coverage Fraction | {result.coverage_fraction:.3f} |",
        f"| Windows Observed | {result.n_windows_observed} |",
        f"| Windows Total | {result.n_windows_total} |",
        f"| Missed Windows | {missed_str} |",
        f"| Flag | {result.flag} |",
    ]
    return "\n".join(lines)


def _cli() -> int:
    parser = argparse.ArgumentParser(
        description="Check what fraction of transit windows are covered by observations."
    )
    parser.add_argument(
        "obs_times",
        nargs="*",
        type=float,
        help="Observation timestamps in BJD.",
    )
    parser.add_argument(
        "--period",
        type=float,
        required=True,
        help="Orbital period in days.",
    )
    parser.add_argument(
        "--epoch",
        type=float,
        required=True,
        help="Reference transit epoch in BJD.",
    )
    parser.add_argument(
        "--baseline",
        type=float,
        required=True,
        help="Total observation baseline in days.",
    )
    args = parser.parse_args()

    result = check_period_window_coverage(
        args.obs_times,
        args.period,
        args.epoch,
        args.baseline,
    )
    print(format_window_coverage(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
