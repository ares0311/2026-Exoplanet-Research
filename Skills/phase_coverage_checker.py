"""Check the phase coverage of a light curve at a given period.

Bins the folded light curve into N_bins phase bins and measures the
fraction that are covered by at least one data point.

Public API
----------
PhaseCoverageResult(period_days, n_bins, n_covered, coverage_fraction,
                    gap_phases, flag)
check_phase_coverage(time, period_days, epoch_bjd, *,
                     n_bins, min_coverage) -> PhaseCoverageResult
format_phase_coverage_result(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PhaseCoverageResult:
    period_days: float
    n_bins: int
    n_covered: int
    coverage_fraction: float
    gap_phases: tuple[float, ...]   # bin-centre phases of uncovered bins
    flag: str                        # "OK", "POOR_COVERAGE", "INSUFFICIENT"


def check_phase_coverage(
    time: list[float],
    period_days: float,
    epoch_bjd: float,
    *,
    n_bins: int = 100,
    min_coverage: float = 0.80,
) -> PhaseCoverageResult:
    """Check phase coverage of a time series.

    Args:
        time: Time array (BJD).
        period_days: Orbital period in days.
        epoch_bjd: Reference epoch in BJD.
        n_bins: Number of phase bins.
        min_coverage: Minimum fraction of bins to consider coverage OK.

    Returns:
        :class:`PhaseCoverageResult`.
    """
    if not time or period_days <= 0 or n_bins <= 0:
        return PhaseCoverageResult(period_days, n_bins, 0, 0.0, (), "INSUFFICIENT")

    covered = [False] * n_bins
    for t in time:
        ph = ((t - epoch_bjd) % period_days) / period_days
        idx = min(int(ph * n_bins), n_bins - 1)
        covered[idx] = True

    n_covered = sum(covered)
    coverage_fraction = n_covered / n_bins
    gap_phases = tuple(
        round((i + 0.5) / n_bins, 4)
        for i, c in enumerate(covered)
        if not c
    )

    flag = "OK" if coverage_fraction >= min_coverage else "POOR_COVERAGE"
    return PhaseCoverageResult(
        period_days=period_days,
        n_bins=n_bins,
        n_covered=n_covered,
        coverage_fraction=round(coverage_fraction, 4),
        gap_phases=gap_phases,
        flag=flag,
    )


def format_phase_coverage_result(result: PhaseCoverageResult) -> str:
    """Format phase coverage result as Markdown."""
    lines = [
        "## Phase Coverage",
        "",
        f"- Period: {result.period_days:.4f} days",
        f"- Bins: {result.n_bins}",
        f"- Covered bins: {result.n_covered}",
        f"- Coverage: {result.coverage_fraction:.1%}",
        f"- **Flag: {result.flag}**",
    ]
    if result.flag == "POOR_COVERAGE" and result.gap_phases:
        top_gaps = result.gap_phases[:5]
        lines.append(f"- Gap phases (first 5): {', '.join(f'{g:.3f}' for g in top_gaps)}")
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="phase_coverage_checker",
        description="Check phase coverage of a light curve.",
    )
    parser.add_argument("period_days", type=float)
    parser.add_argument("epoch_bjd", type=float)
    parser.add_argument("--n-bins", type=int, default=100)
    parser.add_argument("--min-coverage", type=float, default=0.80)
    args = parser.parse_args(argv)

    result = check_phase_coverage(
        [], args.period_days, args.epoch_bjd,
        n_bins=args.n_bins, min_coverage=args.min_coverage,
    )
    print(format_phase_coverage_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
