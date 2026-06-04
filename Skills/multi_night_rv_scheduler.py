"""Optimally schedule RV observations to maximise orbital phase coverage."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RvScheduleEntry:
    night_index: int
    bjd: float
    orbital_phase: float
    phase_gap_filled: float   # nearest gap this point fills


@dataclass(frozen=True)
class RvScheduleResult:
    n_nights: int
    period_days: float
    phase_coverage: float    # fraction of phase bins covered
    phase_rms: float         # RMS spacing of phases around the orbit
    entries: tuple[RvScheduleEntry, ...]
    flag: str


def schedule_rv_observations(
    period_days: float,
    n_observations: int,
    t0_bjd: float = 0.0,
    baseline_days: float | None = None,
    n_phase_bins: int = 20,
) -> RvScheduleResult:
    """
    Compute optimal BJD times for N RV observations to maximise phase coverage.

    Strategy: space observations at irrational-ratio intervals using the
    golden-ratio method (Δphase = φ - 1 ≈ 0.618). This fills the phase
    circle evenly and is robust to baseline limitations.

    Parameters
    ----------
    period_days:       Planet orbital period.
    n_observations:    Number of RV observations to schedule.
    t0_bjd:            Reference epoch (BJD).
    baseline_days:     Total available observing baseline. If None, unconstrained.
    n_phase_bins:      Number of bins used to measure phase coverage.
    """
    if not math.isfinite(period_days) or period_days <= 0.0:
        return RvScheduleResult(
            n_nights=0, period_days=period_days, phase_coverage=float("nan"),
            phase_rms=float("nan"), entries=(), flag="INVALID_PERIOD",
        )
    if n_observations < 1:
        return RvScheduleResult(
            n_nights=0, period_days=period_days, phase_coverage=float("nan"),
            phase_rms=float("nan"), entries=(), flag="INVALID_N_OBSERVATIONS",
        )

    phi = (1.0 + math.sqrt(5.0)) / 2.0  # golden ratio
    delta_phase = 1.0 / phi              # ≈ 0.618

    entries: list[RvScheduleEntry] = []
    phases: list[float] = []

    for i in range(n_observations):
        phase = (i * delta_phase) % 1.0
        bjd = t0_bjd + phase * period_days

        # If baseline limited, wrap into available window
        if baseline_days is not None and baseline_days > 0:
            n_full = math.floor(baseline_days / period_days)
            period_window = max(1.0, n_full * period_days)
            bjd = t0_bjd + (phase * period_window) % period_window

        phases.append(phase)
        # Nearest gap from previous phases
        if len(phases) > 1:
            sorted_ph = sorted(phases[:-1])
            gaps = [abs(phase - q) for q in sorted_ph]
            nearest_gap = min(gaps)
        else:
            nearest_gap = 0.5

        entries.append(RvScheduleEntry(
            night_index=i,
            bjd=round(bjd, 5),
            orbital_phase=round(phase, 5),
            phase_gap_filled=round(nearest_gap, 5),
        ))

    # Phase coverage: fraction of bins covered
    bin_width = 1.0 / n_phase_bins
    covered = set()
    for ph in phases:
        covered.add(int(ph / bin_width) % n_phase_bins)
    coverage = len(covered) / n_phase_bins

    # RMS of phase spacings
    sorted_ph = sorted(phases)
    spacings = [sorted_ph[i + 1] - sorted_ph[i] for i in range(len(sorted_ph) - 1)]
    spacings.append(1.0 + sorted_ph[0] - sorted_ph[-1])  # wrap-around gap
    ideal = 1.0 / len(phases)
    rms = math.sqrt(sum((s - ideal) ** 2 for s in spacings) / len(spacings))

    return RvScheduleResult(
        n_nights=n_observations,
        period_days=period_days,
        phase_coverage=round(coverage, 4),
        phase_rms=round(rms, 5),
        entries=tuple(entries),
        flag="OK",
    )


def format_rv_schedule(r: RvScheduleResult) -> str:
    if r.flag != "OK":
        return f"No schedule (flag: {r.flag}).\n"
    lines = [
        f"**RV Schedule** — {r.n_nights} observations, "
        f"P={r.period_days:.3f} d, "
        f"phase coverage={r.phase_coverage:.0%}, "
        f"phase RMS={r.phase_rms:.4f}\n",
        "| Night | BJD | Phase | Gap filled |",
        "|---|---|---|---|",
    ]
    for e in r.entries:
        lines.append(
            f"| {e.night_index} | {e.bjd:.5f} | {e.orbital_phase:.4f} | {e.phase_gap_filled:.4f} |"
        )
    return "\n".join(lines)


def _cli() -> int:
    p = argparse.ArgumentParser(description="Schedule RV observations for phase coverage.")
    p.add_argument("period_days", type=float)
    p.add_argument("n_observations", type=int)
    p.add_argument("--t0-bjd", type=float, default=0.0)
    p.add_argument("--baseline-days", type=float, default=None)
    args = p.parse_args()
    r = schedule_rv_observations(
        args.period_days, args.n_observations, args.t0_bjd, args.baseline_days,
    )
    print(format_rv_schedule(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
