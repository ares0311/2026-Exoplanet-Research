"""Refine individual transit mid-times from a phase-folded light curve.

For each observed transit, fits a simple parabola around the flux minimum to
refine the mid-time, then computes O-C residuals against the linear ephemeris.

Public API
----------
TransitTiming(transit_number, mid_bjd, oc_minutes, snr)
TransitTimingResult(period_days, epoch_bjd, timings, rms_oc_minutes, n_transits)
fit_transit_times(time, flux, period, epoch, *, duration_days, min_snr,
                  flux_err) -> TransitTimingResult
format_timing_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TransitTiming:
    transit_number: int
    mid_bjd: float
    oc_minutes: float   # observed minus calculated, in minutes
    snr: float          # transit depth / scatter


@dataclass(frozen=True)
class TransitTimingResult:
    period_days: float
    epoch_bjd: float
    timings: tuple[TransitTiming, ...]
    rms_oc_minutes: float
    n_transits: int


def _parabola_min(xs: list[float], ys: list[float]) -> float | None:
    """Fit a parabola to (x, y) and return the x of the minimum."""
    n = len(xs)
    if n < 3:
        return None
    # Vandermonde system: y = a*x^2 + b*x + c  → minimum at x = -b/(2a)
    # Use simple 3-point formula around index of min
    i_min = min(range(n), key=lambda i: ys[i])
    if i_min == 0 or i_min == n - 1:
        return float(xs[i_min])
    x0, x1, x2 = xs[i_min - 1], xs[i_min], xs[i_min + 1]
    y0, y1, y2 = ys[i_min - 1], ys[i_min], ys[i_min + 1]
    denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
    if abs(denom) < 1e-15:
        return float(x1)
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
    b = (x2 ** 2 * (y0 - y1) + x1 ** 2 * (y2 - y0) + x0 ** 2 * (y1 - y2)) / denom
    if abs(a) < 1e-15:
        return float(x1)
    return -b / (2.0 * a)


def fit_transit_times(
    time: list[float],
    flux: list[float],
    period: float,
    epoch: float,
    *,
    duration_days: float = 0.1,
    min_snr: float = 3.0,
    flux_err: list[float] | None = None,
) -> TransitTimingResult:
    """Refine individual transit mid-times.

    Args:
        time: BJD time array.
        flux: Normalised flux array.
        period: Orbital period in days.
        epoch: Reference mid-transit epoch (BJD).
        duration_days: Expected transit duration.
        min_snr: Minimum transit SNR to include.
        flux_err: Per-cadence uncertainties (optional).

    Returns:
        :class:`TransitTimingResult`.
    """
    if not time or period <= 0:
        return TransitTimingResult(period, epoch, (), 0.0, 0)

    # Group cadences into transit windows
    half_dur = duration_days * 1.5  # search window
    t_arr = [float(t) for t in time]
    f_arr = [float(f) for f in flux]
    e_arr = ([float(e) for e in flux_err] if flux_err is not None
             else [1.0] * len(f_arr))

    # Find all expected transit numbers within the data span
    t_min, t_max = t_arr[0], t_arr[-1]
    n_start = math.ceil((t_min - epoch) / period)
    n_end = math.floor((t_max - epoch) / period)

    timings: list[TransitTiming] = []

    for n in range(int(n_start), int(n_end) + 1):
        t_expected = epoch + n * period
        if t_expected < t_min or t_expected > t_max:
            continue

        # Collect cadences within window
        win_t: list[float] = []
        win_f: list[float] = []
        win_e: list[float] = []
        for t, f, e in zip(t_arr, f_arr, e_arr, strict=False):
            if abs(t - t_expected) <= half_dur:
                win_t.append(t)
                win_f.append(f)
                win_e.append(e)

        if len(win_t) < 3:
            continue

        # Estimate SNR: (1 - min_flux) / scatter_out_of_transit
        oot_f = [f for t, f in zip(win_t, win_f, strict=False)
                 if abs(t - t_expected) > duration_days / 2.0]
        if len(oot_f) < 2:
            scatter = 1e-3
        else:
            med = sorted(oot_f)[len(oot_f) // 2]
            sq = sum((f - med) ** 2 for f in oot_f)
            scatter = math.sqrt(sq / len(oot_f)) or 1e-9

        depth = 1.0 - min(win_f)
        snr = depth / scatter

        if snr < min_snr:
            continue

        # Refine mid-time via parabola fit to flux minimum region
        mid_refined = _parabola_min(win_t, win_f)
        if mid_refined is None:
            mid_refined = t_expected

        oc_min = (mid_refined - t_expected) * 1440.0  # days → minutes

        timings.append(TransitTiming(
            transit_number=n,
            mid_bjd=round(mid_refined, 6),
            oc_minutes=round(oc_min, 4),
            snr=round(snr, 2),
        ))

    oc_vals = [t.oc_minutes for t in timings]
    if len(oc_vals) >= 2:
        mean_oc = sum(oc_vals) / len(oc_vals)
        rms_oc = math.sqrt(sum((v - mean_oc) ** 2 for v in oc_vals) / len(oc_vals))
    else:
        rms_oc = 0.0

    return TransitTimingResult(
        period_days=period,
        epoch_bjd=epoch,
        timings=tuple(timings),
        rms_oc_minutes=round(rms_oc, 4),
        n_transits=len(timings),
    )


def format_timing_result(result: TransitTimingResult) -> str:
    """Format transit timing result as Markdown."""
    lines = [
        "## Transit Timing Analysis",
        "",
        f"- Period: {result.period_days:.4f} d",
        f"- Epoch: {result.epoch_bjd:.4f} BJD",
        f"- Transits measured: {result.n_transits}",
        f"- RMS O-C: {result.rms_oc_minutes:.2f} min",
    ]
    if result.timings:
        lines += ["", "| Transit # | Mid BJD | O-C (min) | SNR |",
                  "|---|---|---|---|"]
        for t in result.timings:
            lines.append(
                f"| {t.transit_number} | {t.mid_bjd:.4f} |"
                f" {t.oc_minutes:+.2f} | {t.snr:.1f} |"
            )
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(
        prog="transit_timing_fitter",
        description="Refine individual transit mid-times.",
    )
    parser.add_argument("--lc", required=True, metavar="JSON")
    parser.add_argument("--period", type=float, required=True)
    parser.add_argument("--epoch", type=float, required=True)
    parser.add_argument("--duration", type=float, default=0.1)
    args = parser.parse_args(argv)

    lc = json.loads(Path(args.lc).read_text())
    result = fit_transit_times(
        lc["time"], lc["flux"],
        args.period, args.epoch,
        duration_days=args.duration,
    )
    print(format_timing_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
