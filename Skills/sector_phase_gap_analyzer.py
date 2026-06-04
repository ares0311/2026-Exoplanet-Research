"""Identify unobserved phase ranges in a folded transit coverage."""
from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class PhaseGapResult:
    n_points: int
    n_bins: int
    n_covered_bins: int
    coverage_fraction: float
    largest_gap_width: float
    gap_phases: tuple[float, ...]
    flag: str


def analyze_phase_gaps(
    phase: list[float],
    n_bins: int = 50,
    gap_threshold: float = 0.05,
) -> PhaseGapResult:
    """
    Identify gaps in phase coverage from a folded light curve.

    Divides [-0.5, 0.5) into n_bins equal bins. Bins with no data are gaps.
    gap_phases: bin centres of uncovered bins wider than gap_threshold.
    coverage_fraction: fraction of bins containing ≥1 data point.
    largest_gap_width: width of the longest contiguous gap in phase units.
    """
    if len(phase) < 2:
        return PhaseGapResult(
            n_points=len(phase), n_bins=n_bins, n_covered_bins=0,
            coverage_fraction=0.0, largest_gap_width=1.0,
            gap_phases=(), flag="INSUFFICIENT_DATA",
        )
    if n_bins < 2:
        return PhaseGapResult(
            n_points=len(phase), n_bins=n_bins, n_covered_bins=0,
            coverage_fraction=0.0, largest_gap_width=float("nan"),
            gap_phases=(), flag="INVALID_N_BINS",
        )

    bin_width = 1.0 / n_bins
    covered = [False] * n_bins

    for ph in phase:
        # Wrap to [-0.5, 0.5)
        ph_w = ph % 1.0
        if ph_w >= 0.5:
            ph_w -= 1.0
        idx = int((ph_w + 0.5) / bin_width)
        idx = min(idx, n_bins - 1)
        covered[idx] = True

    n_covered = sum(covered)
    coverage_frac = n_covered / n_bins

    # Find gap centres (consecutive uncovered bins)
    gap_phases: list[float] = []
    max_gap = 0
    cur_gap = 0
    for i, cov in enumerate(covered):
        if not cov:
            cur_gap += 1
            if cur_gap > max_gap:
                max_gap = cur_gap
            gap_centre = -0.5 + (i + 0.5) * bin_width
            if bin_width * cur_gap >= gap_threshold and (
                not gap_phases or abs(gap_phases[-1] - gap_centre) >= bin_width
            ):
                gap_phases.append(round(gap_centre, 4))
        else:
            cur_gap = 0

    largest_gap = max_gap * bin_width

    return PhaseGapResult(
        n_points=len(phase),
        n_bins=n_bins,
        n_covered_bins=n_covered,
        coverage_fraction=round(coverage_frac, 4),
        largest_gap_width=round(largest_gap, 4),
        gap_phases=tuple(gap_phases),
        flag="OK",
    )


def format_phase_gap_result(r: PhaseGapResult) -> str:
    gap_str = ", ".join(f"{g:.3f}" for g in r.gap_phases) if r.gap_phases else "none"
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| N data points | {r.n_points} |\n"
        f"| Phase bins | {r.n_bins} |\n"
        f"| Covered bins | {r.n_covered_bins} |\n"
        f"| Coverage fraction | {r.coverage_fraction:.4f} |\n"
        f"| Largest gap width | {r.largest_gap_width:.4f} |\n"
        f"| Gap phase centres | {gap_str} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Analyze phase gaps in folded light curve.")
    p.add_argument("phase_json", help="JSON array of phase values in [-0.5, 0.5)")
    p.add_argument("--n-bins", type=int, default=50)
    p.add_argument("--gap-threshold", type=float, default=0.05)
    args = p.parse_args()
    import json
    phase = json.loads(args.phase_json)
    r = analyze_phase_gaps(phase, args.n_bins, args.gap_threshold)
    print(format_phase_gap_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
