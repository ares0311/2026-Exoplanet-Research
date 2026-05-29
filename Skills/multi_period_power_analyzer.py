"""Compute and compare BLS-like power for multiple period hypotheses.

Given a phase-folded power metric and a list of candidate periods, ranks
them by fold depth / scatter ratio as a lightweight period power proxy.

Public API
----------
PeriodPower(period_days, depth_ppm, scatter_ppm, snr, power, rank)
MultiPeriodResult(periods_tested, best_period, top_results,
                  period_spacing_days, flag)
analyze_multi_period_power(time, flux, *, periods, duration_days,
                           n_phase_bins) -> MultiPeriodResult
format_multi_period_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PeriodPower:
    period_days: float
    depth_ppm: float
    scatter_ppm: float
    snr: float
    power: float  # depth / scatter (dimensionless)
    rank: int


@dataclass(frozen=True)
class MultiPeriodResult:
    periods_tested: int
    best_period: float | None
    top_results: tuple[PeriodPower, ...]
    period_spacing_days: float | None
    flag: str  # "OK" | "AMBIGUOUS" | "NO_SIGNAL" | "INVALID"


def _phase_fold_stats(
    time: list[float],
    flux: list[float],
    period: float,
    epoch: float,
    duration: float,
    n_bins: int,
) -> tuple[float, float]:
    """Return (depth_ppm, scatter_ppm) for a given fold."""
    if period <= 0 or not time:
        return 0.0, 0.0

    phases = [((t - epoch) / period % 1.0) for t in time]
    phase_width = duration / period

    in_transit: list[float] = []
    out_of_transit: list[float] = []
    for ph, fl in zip(phases, flux, strict=False):
        ph_sym = ph if ph < 0.5 else ph - 1.0
        if abs(ph_sym) < phase_width / 2:
            in_transit.append(fl)
        else:
            out_of_transit.append(fl)

    if len(in_transit) < 2 or len(out_of_transit) < 2:
        return 0.0, 0.0

    mu_in = sum(in_transit) / len(in_transit)
    mu_out = sum(out_of_transit) / len(out_of_transit)
    depth = (mu_out - mu_in) * 1e6  # in ppm (positive = dip)

    if len(out_of_transit) >= 2:
        var = sum((x - mu_out) ** 2 for x in out_of_transit) / (len(out_of_transit) - 1)
        scatter = math.sqrt(var) * 1e6
    else:
        scatter = 0.0

    return max(depth, 0.0), max(scatter, 1e-9)


def analyze_multi_period_power(
    time: list[float] | tuple[float, ...],
    flux: list[float] | tuple[float, ...],
    *,
    periods: list[float],
    duration_days: float = 0.1,
    n_phase_bins: int = 64,
    top_n: int = 5,
) -> MultiPeriodResult:
    """Rank period hypotheses by phase-fold SNR.

    Args:
        time: Observation timestamps in days.
        flux: Normalised flux (mean ≈ 1.0).
        periods: List of candidate periods to test.
        duration_days: Transit duration assumed for folding window.
        n_phase_bins: Phase bins (passed to internal fold but unused in stdlib path).
        top_n: Number of top-ranked results to return.

    Returns:
        MultiPeriodResult with ranked period powers.
    """
    t = list(time)
    f = list(flux)

    if not t or not periods:
        return MultiPeriodResult(
            periods_tested=0,
            best_period=None,
            top_results=(),
            period_spacing_days=None,
            flag="INVALID",
        )

    epoch = t[0] if t else 0.0
    powers: list[PeriodPower] = []
    for period in periods:
        if period <= 0:
            continue
        depth, scatter = _phase_fold_stats(t, f, period, epoch, duration_days, n_phase_bins)
        snr = depth / scatter if scatter > 0 else 0.0
        power = snr  # power proxy = SNR
        powers.append(PeriodPower(
            period_days=period,
            depth_ppm=round(depth, 2),
            scatter_ppm=round(scatter, 2),
            snr=round(snr, 3),
            power=round(power, 4),
            rank=0,  # set below
        ))

    powers.sort(key=lambda p: p.power, reverse=True)
    ranked = [
        PeriodPower(
            period_days=p.period_days,
            depth_ppm=p.depth_ppm,
            scatter_ppm=p.scatter_ppm,
            snr=p.snr,
            power=p.power,
            rank=i + 1,
        )
        for i, p in enumerate(powers)
    ]

    if not ranked:
        return MultiPeriodResult(
            periods_tested=len(periods),
            best_period=None,
            top_results=(),
            period_spacing_days=None,
            flag="NO_SIGNAL",
        )

    best = ranked[0]
    top = tuple(ranked[:top_n])

    # Ambiguous if top two are within 10% power of each other
    flag = "OK"
    if len(ranked) >= 2 and ranked[0].power > 0 and \
            abs(ranked[1].power - ranked[0].power) / ranked[0].power < 0.10:
        flag = "AMBIGUOUS"
    if best.snr < 5.0:
        flag = "NO_SIGNAL"

    # Period spacing: min gap among tested periods (sorted)
    sorted_periods = sorted(periods)
    spacing = None
    if len(sorted_periods) >= 2:
        spacing = min(sorted_periods[i + 1] - sorted_periods[i]
                      for i in range(len(sorted_periods) - 1))

    return MultiPeriodResult(
        periods_tested=len(ranked),
        best_period=best.period_days,
        top_results=top,
        period_spacing_days=round(spacing, 6) if spacing is not None else None,
        flag=flag,
    )


def format_multi_period_result(result: MultiPeriodResult) -> str:
    """Format multi-period power analysis as Markdown.

    Args:
        result: MultiPeriodResult to format.

    Returns:
        Markdown string.
    """
    best_str = (f"{result.best_period:.4f} d"
                if result.best_period is not None else "—")
    lines = [
        "## Multi-Period Power Analysis\n",
        f"**Status**: `{result.flag}` | "
        f"Periods tested: {result.periods_tested} | Best: {best_str}\n",
        "",
        "| Rank | Period (d) | Depth (ppm) | Scatter (ppm) | SNR | Power |",
        "|---|---|---|---|---|---|",
    ]
    for p in result.top_results:
        lines.append(
            f"| {p.rank} | {p.period_days:.4f} | {p.depth_ppm:.1f} | "
            f"{p.scatter_ppm:.1f} | {p.snr:.2f} | {p.power:.4f} |"
        )
    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Analyze BLS power at multiple periods.")
    parser.add_argument("lc", help="Light curve JSON with 'time' and 'flux'.")
    parser.add_argument("--periods", required=True,
                        help="Comma-separated candidate periods in days.")
    parser.add_argument("--duration", type=float, default=0.1)
    args = parser.parse_args(argv)

    from pathlib import Path
    data = json.loads(Path(args.lc).read_text())
    periods = [float(x) for x in args.periods.split(",")]
    result = analyze_multi_period_power(data["time"], data["flux"],
                                        periods=periods, duration_days=args.duration)
    print(format_multi_period_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
