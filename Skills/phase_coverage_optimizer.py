"""Phase coverage optimizer for transit search.

Bins timestamps into phase space and reports which phase bins have observations,
identifying gaps that reduce transit detection efficiency.

Public API
----------
PhaseCoverageResult(coverage_fraction, n_bins_covered, n_bins_total,
                    gap_phases, flag)
optimize_phase_coverage(timestamps_bjd, period_days, epoch_bjd, *,
                        n_bins=20) -> PhaseCoverageResult
format_phase_coverage(result) -> str
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class PhaseCoverageResult:
    coverage_fraction: float
    n_bins_covered: int
    n_bins_total: int
    gap_phases: list[float]
    flag: str


def optimize_phase_coverage(
    timestamps_bjd: list[float],
    period_days: float,
    epoch_bjd: float,
    *,
    n_bins: int = 20,
) -> PhaseCoverageResult:
    """Compute phase coverage across a period-folded grid of bins.

    Parameters
    ----------
    timestamps_bjd: observation timestamps in BJD
    period_days:    orbital period in days (must be > 0)
    epoch_bjd:      reference epoch in BJD
    n_bins:         number of equal phase bins across [0, 1)

    Returns
    -------
    PhaseCoverageResult with flag one of:
    INVALID_PERIOD, NO_DATA, OK
    """
    if period_days <= 0.0:
        return PhaseCoverageResult(
            coverage_fraction=0.0,
            n_bins_covered=0,
            n_bins_total=n_bins,
            gap_phases=list(range(n_bins)),
            flag="INVALID_PERIOD",
        )

    if len(timestamps_bjd) < 1:
        gap_phases = [i / n_bins for i in range(n_bins)]
        return PhaseCoverageResult(
            coverage_fraction=0.0,
            n_bins_covered=0,
            n_bins_total=n_bins,
            gap_phases=gap_phases,
            flag="NO_DATA",
        )

    covered_bins: set[int] = set()
    for t in timestamps_bjd:
        phase = ((t - epoch_bjd) / period_days) % 1.0
        bin_idx = int(phase * n_bins) % n_bins
        covered_bins.add(bin_idx)

    n_bins_covered = len(covered_bins)
    coverage_fraction = n_bins_covered / n_bins
    gap_phases = [i / n_bins for i in range(n_bins) if i not in covered_bins]

    return PhaseCoverageResult(
        coverage_fraction=coverage_fraction,
        n_bins_covered=n_bins_covered,
        n_bins_total=n_bins,
        gap_phases=gap_phases,
        flag="OK",
    )


def format_phase_coverage(result: PhaseCoverageResult) -> str:
    """Return a Markdown table summarising the phase coverage result."""
    gap_str = ", ".join(f"{g:.3f}" for g in result.gap_phases) if result.gap_phases else "none"
    lines = [
        "| Metric | Value |",
        "| --- | --- |",
        f"| Coverage Fraction | {result.coverage_fraction:.4f} |",
        f"| Bins Covered | {result.n_bins_covered} |",
        f"| Total Bins | {result.n_bins_total} |",
        f"| Gap Phases | {gap_str} |",
        f"| Flag | {result.flag} |",
    ]
    return "\n".join(lines)


def _cli() -> int:
    parser = argparse.ArgumentParser(
        description="Compute phase coverage for a set of observation timestamps."
    )
    parser.add_argument(
        "--timestamps",
        nargs="+",
        type=float,
        required=True,
        metavar="BJD",
        help="Observation timestamps in BJD.",
    )
    parser.add_argument(
        "--period",
        type=float,
        required=True,
        metavar="DAYS",
        help="Orbital period in days.",
    )
    parser.add_argument(
        "--epoch",
        type=float,
        required=True,
        metavar="BJD",
        help="Reference epoch in BJD.",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=20,
        metavar="N",
        help="Number of phase bins (default 20).",
    )
    args = parser.parse_args()
    result = optimize_phase_coverage(
        args.timestamps,
        args.period,
        args.epoch,
        n_bins=args.n_bins,
    )
    print(format_phase_coverage(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
