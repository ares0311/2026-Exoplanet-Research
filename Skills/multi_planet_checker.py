"""Search for additional transit signals after masking the primary candidate.

Masks the primary transit windows from the light curve and re-runs BLS on the
residuals.  Returns up to *max_additional* secondary period candidates.

Public API
----------
MultiPlanetResult(primary_period, additional_signals, n_additional, masked_fraction)
check_for_additional_planets(time, flux, primary_period, primary_epoch, *,
                              duration_days, max_additional, search_fn) -> MultiPlanetResult
format_multi_planet_result(result) -> str
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AdditionalSignal:
    period_days: float
    epoch_bjd: float
    depth_ppm: float
    snr: float
    duration_hours: float


@dataclass(frozen=True)
class MultiPlanetResult:
    primary_period: float
    additional_signals: tuple[AdditionalSignal, ...]
    n_additional: int
    masked_fraction: float          # fraction of cadences masked


def _mask_transits(
    time: list[float],
    flux: list[float],
    period: float,
    epoch: float,
    half_window: float,
) -> tuple[list[float], list[float]]:
    """Return time/flux arrays with primary transit windows removed."""
    out_t, out_f = [], []
    for t, f in zip(time, flux, strict=False):
        phase = (t - epoch) % period
        if phase > period / 2:
            phase -= period
        if abs(phase) > half_window:
            out_t.append(t)
            out_f.append(f)
    return out_t, out_f


def _default_search(
    time: list[float],
    flux: list[float],
) -> list[dict[str, Any]]:
    """Thin wrapper around astropy BLS; returns list of signal dicts."""
    import numpy as np
    from astropy.timeseries import BoxLeastSquares

    t = np.asarray(time)
    f = np.asarray(flux)
    if len(t) < 20:
        return []
    span = float(t[-1] - t[0])
    period_min = max(0.5, span / 50)
    period_max = span / 2.0
    if period_min >= period_max:
        return []
    bls = BoxLeastSquares(t, f)
    result = bls.autopower(
        [0.05, 0.10, 0.15],
        minimum_period=period_min,
        maximum_period=period_max,
    )
    best_idx = int(np.argmax(result.power))
    p = float(result.period[best_idx])
    stats = bls.compute_stats(
        result.period[best_idx],
        result.duration[best_idx],
        result.transit_time[best_idx],
    )
    snr = float(result.power[best_idx])
    depth_ppm = float(stats["depth"][0]) * 1e6
    return [{
        "period_days": p,
        "epoch_bjd": float(result.transit_time[best_idx]),
        "depth_ppm": max(0.0, depth_ppm),
        "snr": snr,
        "duration_hours": float(result.duration[best_idx]) * 24,
    }]


def check_for_additional_planets(
    time: list[float],
    flux: list[float],
    primary_period: float,
    primary_epoch: float,
    *,
    duration_days: float = 0.1,
    max_additional: int = 3,
    min_snr: float = 5.0,
    search_fn: Callable[
        [list[float], list[float]], list[dict]
    ] | None = None,
) -> MultiPlanetResult:
    """Run multi-planet search by iterative transit masking.

    Args:
        time: BJD time array.
        flux: Normalised flux array (mean ≈ 1.0).
        primary_period: Period of the already-detected signal (days).
        primary_epoch: Mid-transit epoch of the primary (BJD).
        duration_days: Duration used as the mask half-window.
        max_additional: Maximum number of additional signals to search for.
        min_snr: Minimum BLS power (proxy SNR) for a secondary to be reported.
        search_fn: Injectable search function ``(time, flux) -> list[dict]``.

    Returns:
        :class:`MultiPlanetResult`.
    """
    fn = search_fn if search_fn is not None else _default_search

    masked_t, masked_f = _mask_transits(
        time, flux, primary_period, primary_epoch, duration_days / 2
    )
    masked_fraction = 1.0 - len(masked_t) / max(len(time), 1)

    signals: list[AdditionalSignal] = []
    cur_t, cur_f = masked_t, masked_f

    for _ in range(max_additional):
        if len(cur_t) < 20:
            break
        found = fn(cur_t, cur_f)
        if not found:
            break
        best = found[0]
        if best.get("snr", 0) < min_snr:
            break
        sig = AdditionalSignal(
            period_days=best["period_days"],
            epoch_bjd=best["epoch_bjd"],
            depth_ppm=best["depth_ppm"],
            snr=best["snr"],
            duration_hours=best["duration_hours"],
        )
        signals.append(sig)
        # mask this signal too for the next iteration
        cur_t, cur_f = _mask_transits(
            cur_t, cur_f,
            sig.period_days, sig.epoch_bjd,
            duration_days / 2,
        )

    return MultiPlanetResult(
        primary_period=primary_period,
        additional_signals=tuple(signals),
        n_additional=len(signals),
        masked_fraction=masked_fraction,
    )


def format_multi_planet_result(result: MultiPlanetResult) -> str:
    """Format multi-planet check as Markdown."""
    lines = [
        "## Multi-Planet Check",
        "",
        f"- Primary period: {result.primary_period:.4f} d",
        f"- Cadences masked: {result.masked_fraction:.1%}",
        f"- Additional signals found: {result.n_additional}",
    ]
    for i, sig in enumerate(result.additional_signals, 1):
        lines += [
            "",
            f"### Signal {i}",
            f"- Period: {sig.period_days:.4f} d",
            f"- Depth: {sig.depth_ppm:.0f} ppm",
            f"- SNR: {sig.snr:.1f}",
            f"- Duration: {sig.duration_hours:.2f} h",
        ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(
        prog="multi_planet_checker",
        description="Search for additional planets after masking the primary.",
    )
    parser.add_argument("--lc", required=True, metavar="JSON",
                        help="Light-curve JSON file with 'time' and 'flux' keys.")
    parser.add_argument("--period", type=float, required=True)
    parser.add_argument("--epoch", type=float, required=True)
    parser.add_argument("--duration", type=float, default=0.1,
                        help="Transit duration in days.")
    parser.add_argument("--max-signals", type=int, default=3)
    parser.add_argument("--min-snr", type=float, default=5.0)
    args = parser.parse_args(argv)

    lc = json.loads(Path(args.lc).read_text())
    result = check_for_additional_planets(
        lc["time"], lc["flux"],
        args.period, args.epoch,
        duration_days=args.duration,
        max_additional=args.max_signals,
        min_snr=args.min_snr,
    )
    print(format_multi_planet_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
