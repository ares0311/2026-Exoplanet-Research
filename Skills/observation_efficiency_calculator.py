"""Compute the fraction of orbital phase covered by available observations.

Low phase coverage means transits may be missed at certain phases, and
period aliases are more likely.  This module bins the phase-folded observation
timestamps to measure the coverage fraction and identify uncovered gaps.

Public API
----------
ObsEfficiencyResult(period_days, n_phase_bins, bins_covered, coverage_fraction,
                    max_gap_phase, is_sufficient, flag)
compute_obs_efficiency(time, period_days, epoch_bjd, *,
                       n_bins, min_coverage) -> ObsEfficiencyResult
format_efficiency_result(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObsEfficiencyResult:
    period_days: float
    n_phase_bins: int
    bins_covered: int
    coverage_fraction: float   # bins_covered / n_phase_bins
    max_gap_phase: float       # largest contiguous gap in phase units
    is_sufficient: bool
    flag: str  # "OK" | "INSUFFICIENT" | "INVALID"


def compute_obs_efficiency(
    time: list[float],
    period_days: float,
    epoch_bjd: float,
    *,
    n_bins: int = 50,
    min_coverage: float = 0.5,
) -> ObsEfficiencyResult:
    """Compute phase coverage fraction from observation timestamps.

    Args:
        time: Time array (days).
        period_days: Orbital period in days.
        epoch_bjd: Reference epoch.
        n_bins: Number of phase bins.
        min_coverage: Minimum fraction for ``is_sufficient``.

    Returns:
        :class:`ObsEfficiencyResult`.
    """
    n = len(time)
    if n < 2:
        return ObsEfficiencyResult(period_days, n_bins, 0, 0.0, 1.0, False, "INVALID")
    if period_days <= 0 or n_bins < 2:
        return ObsEfficiencyResult(period_days, n_bins, 0, 0.0, 1.0, False, "INVALID")

    covered = [False] * n_bins
    for t in time:
        ph = ((t - epoch_bjd) % period_days) / period_days
        idx = min(int(ph * n_bins), n_bins - 1)
        covered[idx] = True

    n_covered = sum(covered)
    coverage_frac = n_covered / n_bins

    # Find largest gap (circular)
    # Extend covered by one wrap-around for circular gap detection
    ext = covered + covered
    max_gap = 0
    cur_gap = 0
    for c in ext:
        if not c:
            cur_gap += 1
            max_gap = max(max_gap, cur_gap)
        else:
            cur_gap = 0
    # Cap gap at n_bins (can't exceed full period in circular sense)
    max_gap = min(max_gap, n_bins)
    max_gap_phase = max_gap / n_bins

    flag = "INSUFFICIENT" if n_covered < 3 else "OK"
    return ObsEfficiencyResult(
        period_days=period_days,
        n_phase_bins=n_bins,
        bins_covered=n_covered,
        coverage_fraction=round(coverage_frac, 4),
        max_gap_phase=round(max_gap_phase, 4),
        is_sufficient=coverage_frac >= min_coverage,
        flag=flag,
    )


def format_efficiency_result(result: ObsEfficiencyResult) -> str:
    """Format observation efficiency result as Markdown."""
    lines = [
        "## Observation Efficiency",
        "",
        f"- Period: {result.period_days} days",
        f"- Phase bins: {result.n_phase_bins}",
        f"- Bins covered: {result.bins_covered} / {result.n_phase_bins}",
        f"- **Coverage fraction: {result.coverage_fraction:.3f}**",
        f"- Max phase gap: {result.max_gap_phase:.3f}",
        f"- Sufficient coverage: {'Yes' if result.is_sufficient else 'No'}",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="observation_efficiency_calculator",
        description="Compute phase coverage fraction from observation times.",
    )
    parser.add_argument("period_days", type=float)
    parser.add_argument("epoch_bjd", type=float)
    parser.add_argument("--n-bins", type=int, default=50)
    args = parser.parse_args(argv)

    result = compute_obs_efficiency([], args.period_days, args.epoch_bjd, n_bins=args.n_bins)
    print(format_efficiency_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
