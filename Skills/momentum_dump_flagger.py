"""Flag TESS momentum-dump events from reaction-wheel desaturation.

Momentum dumps occur approximately every 2.5 days in TESS sectors and cause
discontinuities in the light curve. This skill identifies cadences affected
by dumps using either a supplied event list or a periodic heuristic.

Public API
----------
MomentumDumpResult(n_dumps_found, flagged_cadences, dump_times,
                   fraction_flagged, flag)
flag_momentum_dumps(time, flux, *, dump_times, period_days,
                    window_hours) -> MomentumDumpResult
format_momentum_dump_result(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MomentumDumpResult:
    n_dumps_found: int
    flagged_cadences: int
    dump_times: tuple[float, ...]
    fraction_flagged: float
    flag: str               # "CLEAN", "MINOR", "SIGNIFICANT"


def flag_momentum_dumps(
    time: list[float],
    flux: list[float],
    *,
    dump_times: list[float] | None = None,
    period_days: float = 2.5,
    window_hours: float = 2.0,
    significant_threshold: float = 0.05,
) -> MomentumDumpResult:
    """Identify cadences affected by TESS momentum dumps.

    Args:
        time: BJD time array.
        flux: Normalised flux array (used to estimate array length).
        dump_times: Known dump mid-times (BJD). If None, uses periodic heuristic.
        period_days: Period for heuristic dump placement (default 2.5 days).
        window_hours: Half-window around each dump to flag (hours).
        significant_threshold: Fraction of cadences flagged above which flag=SIGNIFICANT.

    Returns:
        :class:`MomentumDumpResult`.
    """
    if not time:
        return MomentumDumpResult(0, 0, (), 0.0, "CLEAN")

    t_arr = [float(t) for t in time]
    window_days = window_hours / 24.0

    if dump_times is not None:
        candidate_dumps = [float(d) for d in dump_times]
    else:
        t_min, t_max = t_arr[0], t_arr[-1]
        candidate_dumps = []
        t = t_min + period_days / 2.0
        while t <= t_max:
            candidate_dumps.append(t)
            t += period_days

    # Find dumps that actually fall within the time range
    t_min, t_max = t_arr[0], t_arr[-1]
    active_dumps = [d for d in candidate_dumps if t_min - window_days <= d <= t_max + window_days]

    flagged = set()
    for dump_t in active_dumps:
        for i, t in enumerate(t_arr):
            if abs(t - dump_t) <= window_days:
                flagged.add(i)

    n_total = len(t_arr)
    n_flagged = len(flagged)
    fraction = n_flagged / n_total if n_total > 0 else 0.0

    if fraction == 0.0:
        flag = "CLEAN"
    elif fraction >= significant_threshold:
        flag = "SIGNIFICANT"
    else:
        flag = "MINOR"

    return MomentumDumpResult(
        n_dumps_found=len(active_dumps),
        flagged_cadences=n_flagged,
        dump_times=tuple(sorted(active_dumps)),
        fraction_flagged=round(fraction, 5),
        flag=flag,
    )


def format_momentum_dump_result(result: MomentumDumpResult) -> str:
    """Format momentum dump result as Markdown."""
    lines = [
        "## Momentum Dump Flags",
        "",
        f"- Dumps identified: {result.n_dumps_found}",
        f"- Flagged cadences: {result.flagged_cadences} ({result.fraction_flagged * 100:.2f}%)",
        f"- Flag: **{result.flag}**",
    ]
    if result.dump_times:
        times_str = ", ".join(f"{t:.4f}" for t in result.dump_times[:5])
        if len(result.dump_times) > 5:
            times_str += f" … (+{len(result.dump_times) - 5} more)"
        lines.append(f"- Dump times (BJD): {times_str}")
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(
        prog="momentum_dump_flagger",
        description="Flag TESS momentum-dump cadences.",
    )
    parser.add_argument("--lc", required=True, metavar="JSON")
    parser.add_argument("--period", type=float, default=2.5)
    parser.add_argument("--window-hours", type=float, default=2.0)
    args = parser.parse_args(argv)

    lc = json.loads(Path(args.lc).read_text())
    result = flag_momentum_dumps(
        lc["time"], lc.get("flux", []),
        period_days=args.period,
        window_hours=args.window_hours,
    )
    print(format_momentum_dump_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
