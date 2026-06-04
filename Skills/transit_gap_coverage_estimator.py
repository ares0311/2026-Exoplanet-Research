"""Estimate orbital phase coverage and transit detection probability from timestamps."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TransitGapCoverageResult:
    n_timestamps: int
    baseline_days: float
    phase_coverage_fraction: float
    largest_gap_phase: float
    n_phase_bins_covered: int
    n_phase_bins_total: int
    prob_transit_missed: float
    flag: str


def compute_transit_gap_coverage(
    timestamps_bjd: list[float],
    period_days: float,
    epoch_bjd: float | None = None,
    transit_duration_phase: float = 0.05,
    n_bins: int = 100,
) -> TransitGapCoverageResult:
    """Estimate phase coverage and probability of missing a transit.

    Folds timestamps onto the orbital period and counts what fraction of
    phase bins contain at least one observation.

    Args:
        timestamps_bjd: list of observation timestamps (BJD)
        period_days: orbital period (days)
        epoch_bjd: reference epoch; defaults to timestamps_bjd[0]
        transit_duration_phase: transit duration as fraction of period (for gap assessment)
        n_bins: number of phase bins for coverage computation
    """
    if len(timestamps_bjd) < 2:
        return TransitGapCoverageResult(
            n_timestamps=len(timestamps_bjd),
            baseline_days=0.0,
            phase_coverage_fraction=0.0,
            largest_gap_phase=1.0,
            n_phase_bins_covered=0,
            n_phase_bins_total=n_bins,
            prob_transit_missed=1.0,
            flag="INSUFFICIENT_TIMESTAMPS",
        )
    if period_days <= 0.0:
        return TransitGapCoverageResult(
            n_timestamps=len(timestamps_bjd),
            baseline_days=0.0,
            phase_coverage_fraction=0.0,
            largest_gap_phase=1.0,
            n_phase_bins_covered=0,
            n_phase_bins_total=n_bins,
            prob_transit_missed=1.0,
            flag="INVALID_PERIOD",
        )

    t0 = epoch_bjd if epoch_bjd is not None else timestamps_bjd[0]
    baseline = max(timestamps_bjd) - min(timestamps_bjd)

    phases = sorted(((t - t0) / period_days % 1.0) for t in timestamps_bjd)

    covered = [False] * n_bins
    for ph in phases:
        b = int(ph * n_bins) % n_bins
        covered[b] = True

    n_covered = sum(covered)
    coverage = n_covered / n_bins

    # Largest contiguous gap in phase
    gaps = []
    in_gap = False
    gap_start = 0
    circ = covered + covered  # wrap-around
    for i, c in enumerate(circ):
        if not c and not in_gap:
            in_gap = True
            gap_start = i
        elif c and in_gap:
            in_gap = False
            gaps.append((i - gap_start) / n_bins)
    if in_gap:
        gaps.append((len(circ) - gap_start) / n_bins)
    largest_gap = min(max(gaps) if gaps else 0.0, 1.0)

    # Probability of transit falling in largest contiguous gap
    prob_missed = min(largest_gap / max(transit_duration_phase, 1e-9), 1.0)
    prob_missed = min(prob_missed, 1.0 - coverage)

    return TransitGapCoverageResult(
        n_timestamps=len(timestamps_bjd),
        baseline_days=baseline,
        phase_coverage_fraction=coverage,
        largest_gap_phase=largest_gap,
        n_phase_bins_covered=n_covered,
        n_phase_bins_total=n_bins,
        prob_transit_missed=prob_missed,
        flag="OK",
    )


def format_transit_gap_coverage_result(r: TransitGapCoverageResult) -> str:
    if r.flag != "OK":
        return f"TransitGapCoverage | flag={r.flag}"
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Timestamps | {r.n_timestamps} |\n"
        f"| Baseline | {r.baseline_days:.2f} days |\n"
        f"| Phase coverage | {r.phase_coverage_fraction:.3f} "
        f"({r.n_phase_bins_covered}/{r.n_phase_bins_total} bins) |\n"
        f"| Largest gap | {r.largest_gap_phase:.3f} (phase fraction) |\n"
        f"| P(transit missed) | {r.prob_transit_missed:.3f} |\n"
        f"| Flag | {r.flag} |"
    )


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Transit gap coverage estimator")
    p.add_argument("period_days", type=float, help="Orbital period (days)")
    p.add_argument("timestamps", type=float, nargs="+", help="Observation timestamps (BJD)")
    args = p.parse_args()
    r = compute_transit_gap_coverage(args.timestamps, args.period_days)
    print(format_transit_gap_coverage_result(r))


if __name__ == "__main__":
    _cli()
