"""Count transits missed due to TESS inter-sector gaps."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MissedTransitResult:
    period_days: float
    baseline_days: float
    n_transits_total: int       # transits expected over baseline
    n_transits_observed: int    # transits falling outside gap windows
    n_transits_missed: int      # transits falling inside gap windows
    fraction_missed: float
    gap_windows: tuple[tuple[float, float], ...]
    flag: str


def count_missed_transits(
    period_days: float,
    epoch_bjd: float,
    baseline_start_bjd: float,
    baseline_end_bjd: float,
    gap_windows: list[tuple[float, float]] | None = None,
) -> MissedTransitResult:
    """
    Count transits missed due to data gaps (e.g. TESS sector gaps).

    Parameters
    ----------
    period_days:          Orbital period.
    epoch_bjd:            Transit mid-point reference epoch (BJD).
    baseline_start_bjd:   Start of observing baseline (BJD).
    baseline_end_bjd:     End of observing baseline (BJD).
    gap_windows:          List of (gap_start_bjd, gap_end_bjd) tuples.
                          If None, uses default TESS ~1-day inter-sector gaps
                          every 27.4 days starting from baseline_start.
    """
    if not math.isfinite(period_days) or period_days <= 0:
        return MissedTransitResult(
            period_days=period_days, baseline_days=0.0,
            n_transits_total=0, n_transits_observed=0,
            n_transits_missed=0, fraction_missed=float("nan"),
            gap_windows=(), flag="INVALID_PERIOD",
        )

    baseline_days = baseline_end_bjd - baseline_start_bjd
    if not math.isfinite(baseline_days) or baseline_days <= 0:
        return MissedTransitResult(
            period_days=period_days, baseline_days=baseline_days,
            n_transits_total=0, n_transits_observed=0,
            n_transits_missed=0, fraction_missed=float("nan"),
            gap_windows=(), flag="INVALID_BASELINE",
        )

    # Default gap windows: ~1-day gap every 27.4 days (TESS sector boundaries)
    if gap_windows is None:
        gaps: list[tuple[float, float]] = []
        t_gap = baseline_start_bjd + 27.4
        while t_gap < baseline_end_bjd:
            gaps.append((t_gap - 0.5, t_gap + 0.5))
            t_gap += 27.4
        gap_windows = gaps

    # Find all transit times in baseline
    n_start = math.ceil((baseline_start_bjd - epoch_bjd) / period_days)
    n_end = math.floor((baseline_end_bjd - epoch_bjd) / period_days)

    transit_times = [
        epoch_bjd + n * period_days
        for n in range(n_start, n_end + 1)
        if baseline_start_bjd <= epoch_bjd + n * period_days <= baseline_end_bjd
    ]

    # Check which transits fall in a gap
    n_missed = 0
    for t in transit_times:
        for g_start, g_end in gap_windows:
            if g_start <= t <= g_end:
                n_missed += 1
                break

    n_total = len(transit_times)
    n_obs = n_total - n_missed
    frac = n_missed / n_total if n_total > 0 else 0.0

    return MissedTransitResult(
        period_days=period_days,
        baseline_days=round(baseline_days, 2),
        n_transits_total=n_total,
        n_transits_observed=n_obs,
        n_transits_missed=n_missed,
        fraction_missed=round(frac, 4),
        gap_windows=tuple(gap_windows),
        flag="OK",
    )


def format_missed_transit_result(r: MissedTransitResult) -> str:
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Period (days) | {r.period_days:.4f} |\n"
        f"| Baseline (days) | {r.baseline_days:.2f} |\n"
        f"| Total transits | {r.n_transits_total} |\n"
        f"| Observed transits | {r.n_transits_observed} |\n"
        f"| Missed transits | {r.n_transits_missed} |\n"
        f"| Fraction missed | {r.fraction_missed:.4f} |\n"
        f"| N gap windows | {len(r.gap_windows)} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Count transits missed in TESS gaps.")
    p.add_argument("period_days", type=float)
    p.add_argument("epoch_bjd", type=float)
    p.add_argument("baseline_start_bjd", type=float)
    p.add_argument("baseline_end_bjd", type=float)
    args = p.parse_args()
    r = count_missed_transits(
        args.period_days, args.epoch_bjd,
        args.baseline_start_bjd, args.baseline_end_bjd,
    )
    print(format_missed_transit_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
