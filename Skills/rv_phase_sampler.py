"""Select RV observation epochs that maximize phase coverage for a given period.

Evenly spaces N_obs phases offset by 0.1 for quadrature coverage and
computes the fraction of [0, 1) covered within gap_width of any observation.

Public API
----------
RvPhaseResult
sample_rv_phases(period_days, n_obs, *, reference_bjd, gap_width) -> RvPhaseResult
format_rv_phases(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RvPhaseResult:
    period_days: float
    n_obs: int
    phases: tuple[float, ...]
    bjd_times: tuple[float, ...]
    phase_coverage: float
    max_gap: float
    flag: str  # "OK" | "INVALID"


def _compute_coverage(phases: list[float], gap_width: float) -> float:
    """Compute fraction of [0, 1) within gap_width of any phase."""
    if not phases:
        return 0.0
    # Sample [0, 1) at fine resolution and count covered points
    n_sample = 1000
    covered = 0
    for i in range(n_sample):
        p = i / n_sample
        for obs_p in phases:
            # Handle wrap-around
            diff = abs(p - obs_p)
            diff = min(diff, 1.0 - diff)
            if diff <= gap_width:
                covered += 1
                break
    return covered / n_sample


def _compute_max_gap(phases: list[float]) -> float:
    """Compute the largest gap between consecutive sorted phases (circular)."""
    if not phases:
        return 1.0
    sorted_p = sorted(phases)
    gaps = []
    for i in range(len(sorted_p) - 1):
        gaps.append(sorted_p[i + 1] - sorted_p[i])
    # Wrap-around gap
    gaps.append(1.0 - sorted_p[-1] + sorted_p[0])
    return max(gaps)


def sample_rv_phases(
    period_days: float,
    n_obs: int,
    *,
    reference_bjd: float = 2460000.0,
    gap_width: float = 0.05,
) -> RvPhaseResult:
    """Select RV observation epochs maximizing phase coverage.

    Phases are placed at (i + 0.1) / n_obs for i in 0..n_obs-1.

    Args:
        period_days: Orbital period in days.
        n_obs: Number of RV observations.
        reference_bjd: BJD reference epoch.
        gap_width: Phase width counted as covered around each observation.

    Returns:
        :class:`RvPhaseResult`.
    """
    if not math.isfinite(period_days) or period_days <= 0 or n_obs <= 0:
        return RvPhaseResult(
            period_days=period_days,
            n_obs=n_obs,
            phases=(),
            bjd_times=(),
            phase_coverage=0.0,
            max_gap=1.0,
            flag="INVALID",
        )

    phases = [(i + 0.1) / n_obs % 1.0 for i in range(n_obs)]
    bjd_times = tuple(
        round(reference_bjd + p * period_days, 6) for p in phases
    )

    phase_coverage = _compute_coverage(phases, gap_width)
    max_gap = _compute_max_gap(phases)

    return RvPhaseResult(
        period_days=period_days,
        n_obs=n_obs,
        phases=tuple(round(p, 6) for p in phases),
        bjd_times=bjd_times,
        phase_coverage=round(phase_coverage, 4),
        max_gap=round(max_gap, 6),
        flag="OK",
    )


def format_rv_phases(result: RvPhaseResult) -> str:
    """Format RV phase sampling result as Markdown."""
    phase_list = ", ".join(f"{p:.4f}" for p in result.phases)
    lines = [
        "## RV Phase Sampler",
        "",
        f"- Period: {result.period_days} days",
        f"- N_obs: {result.n_obs}",
        f"- Phases: {phase_list}",
        f"- **Phase coverage: {result.phase_coverage:.3f}**",
        f"- Max gap: {result.max_gap:.4f}",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="rv_phase_sampler",
        description="Select RV observation epochs maximizing phase coverage.",
    )
    parser.add_argument("period_days", type=float)
    parser.add_argument("n_obs", type=int)
    parser.add_argument("--reference-bjd", type=float, default=2460000.0)
    parser.add_argument("--gap-width", type=float, default=0.05)
    args = parser.parse_args(argv)

    result = sample_rv_phases(
        args.period_days,
        args.n_obs,
        reference_bjd=args.reference_bjd,
        gap_width=args.gap_width,
    )
    print(format_rv_phases(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
