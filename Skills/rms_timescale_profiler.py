"""Profile RMS noise as a function of bin timescale (noise floor analysis).

Bins the light curve at logarithmically-spaced timescales and measures the
RMS of bin-mean fluxes.  The white-noise floor scales as 1/√N_bins; deviations
above it indicate correlated (red) noise.  Distinct from
``scatter_metric_calculator`` (single-timescale RMS) and
``correlated_noise_estimator`` (beta-factor method).

Public API
----------
TimescaleBin(timescale_hours, n_bins, rms_ppm, expected_white_rms_ppm,
             red_noise_ratio)
RMSTimescaleResult(n_points, cadence_hours, baseline_rms_ppm,
                   timescale_bins, flag)
profile_rms_timescales(time, flux, *, n_timescales,
                       min_timescale_hours, max_timescale_hours) -> RMSTimescaleResult
format_rms_timescale_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TimescaleBin:
    timescale_hours: float
    n_bins: int                    # number of bins that contain data
    rms_ppm: float                 # RMS of bin-mean fluxes (ppm)
    expected_white_rms_ppm: float  # baseline_rms / sqrt(N_points_per_bin)
    red_noise_ratio: float         # rms_ppm / expected_white_rms_ppm


@dataclass(frozen=True)
class RMSTimescaleResult:
    n_points: int
    cadence_hours: float | None     # median cadence
    baseline_rms_ppm: float | None  # single-point RMS in ppm
    timescale_bins: tuple[TimescaleBin, ...]
    flag: str  # "OK" | "INSUFFICIENT" | "INVALID"


def _rms(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def profile_rms_timescales(
    time: list[float],
    flux: list[float],
    *,
    n_timescales: int = 10,
    min_timescale_hours: float | None = None,
    max_timescale_hours: float | None = None,
) -> RMSTimescaleResult:
    """Profile RMS vs. binning timescale.

    Args:
        time: Time array (days).
        flux: Normalised flux array (out-of-transit ≈ 1.0).
        n_timescales: Number of logarithmically-spaced timescales to evaluate.
        min_timescale_hours: Minimum timescale (hours).  Defaults to 2× cadence.
        max_timescale_hours: Maximum timescale (hours).  Defaults to 10% of baseline.

    Returns:
        :class:`RMSTimescaleResult`.
    """
    if len(time) != len(flux):
        return RMSTimescaleResult(len(time), None, None, (), "INVALID")
    if len(time) < 10:
        return RMSTimescaleResult(len(time), None, None, (), "INSUFFICIENT")
    if n_timescales < 1:
        return RMSTimescaleResult(len(time), None, None, (), "INVALID")

    # Sort by time
    pairs = sorted(zip(time, flux, strict=False), key=lambda p: p[0])
    time_s = [p[0] for p in pairs]
    flux_s = [p[1] for p in pairs]

    # Median cadence
    diffs = [(time_s[i + 1] - time_s[i]) * 24.0 for i in range(len(time_s) - 1) if
             time_s[i + 1] > time_s[i]]
    if not diffs:
        return RMSTimescaleResult(len(time), None, None, (), "INSUFFICIENT")
    diffs.sort()
    cadence_h = diffs[len(diffs) // 2]

    # Baseline RMS in ppm
    mean_flux = sum(flux_s) / len(flux_s)
    if mean_flux <= 0:
        return RMSTimescaleResult(len(time), round(cadence_h, 6), None, (), "INVALID")
    flux_norm = [f / mean_flux for f in flux_s]
    baseline_rms = _rms(flux_norm) * 1e6  # ppm

    baseline_h = (time_s[-1] - time_s[0]) * 24.0
    min_ts = min_timescale_hours if min_timescale_hours is not None else max(cadence_h * 2, 0.1)
    default_max = max(baseline_h * 0.1, min_ts * 2)
    max_ts = max_timescale_hours if max_timescale_hours is not None else default_max

    if max_ts <= min_ts:
        return RMSTimescaleResult(len(time), round(cadence_h, 6),
                                   round(baseline_rms, 3), (), "INSUFFICIENT")

    # Logarithmically-spaced timescales
    log_min = math.log10(min_ts)
    log_max = math.log10(max_ts)
    timescales = [
        10.0 ** (log_min + (log_max - log_min) * i / max(n_timescales - 1, 1))
        for i in range(n_timescales)
    ]

    bins_list: list[TimescaleBin] = []
    for ts_h in timescales:
        ts_days = ts_h / 24.0
        t_start = time_s[0]
        t_end = time_s[-1]
        bin_means: list[float] = []
        t = t_start
        while t < t_end:
            pts = [flux_norm[i] for i, tt in enumerate(time_s) if t <= tt < t + ts_days]
            if pts:
                bin_means.append(sum(pts) / len(pts))
            t += ts_days

        if len(bin_means) < 2:
            continue

        rms_val = _rms(bin_means) * 1e6
        avg_pts = len(flux_norm) / len(bin_means)
        expected = baseline_rms / math.sqrt(max(avg_pts, 1.0))
        ratio = rms_val / expected if expected > 0 else 1.0

        bins_list.append(TimescaleBin(
            timescale_hours=round(ts_h, 4),
            n_bins=len(bin_means),
            rms_ppm=round(rms_val, 3),
            expected_white_rms_ppm=round(expected, 3),
            red_noise_ratio=round(ratio, 4),
        ))

    return RMSTimescaleResult(
        n_points=len(time),
        cadence_hours=round(cadence_h, 6),
        baseline_rms_ppm=round(baseline_rms, 3),
        timescale_bins=tuple(bins_list),
        flag="OK" if bins_list else "INSUFFICIENT",
    )


def format_rms_timescale_result(result: RMSTimescaleResult) -> str:
    """Format RMS timescale profile as Markdown."""
    lines = [
        "## RMS Timescale Profiler",
        "",
        f"- Data points: {result.n_points}",
        f"- Cadence: {result.cadence_hours} hours",
        f"- **Baseline RMS: {result.baseline_rms_ppm} ppm**",
        f"- Timescales evaluated: {len(result.timescale_bins)}",
        f"- **Flag: {result.flag}**",
    ]
    if result.timescale_bins:
        lines += ["", "| Timescale (h) | N bins | RMS (ppm) | Expected (ppm) | Red-noise ratio |",
                  "|---|---|---|---|---|"]
        for b in result.timescale_bins:
            lines.append(
                f"| {b.timescale_hours:.3f} | {b.n_bins}"
                f" | {b.rms_ppm:.2f} | {b.expected_white_rms_ppm:.2f} | {b.red_noise_ratio:.3f} |"
            )
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="rms_timescale_profiler",
        description="Profile photometric RMS vs. binning timescale.",
    )
    parser.add_argument("--n-timescales", type=int, default=10)
    args = parser.parse_args(argv)

    result = profile_rms_timescales([], [], n_timescales=args.n_timescales)
    print(format_rms_timescale_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
