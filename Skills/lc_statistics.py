"""Compute light curve quality statistics: CDPP, RMS, photon noise floor.

Public API
----------
LCStats(cdpp_ppm, rms_ppm, photon_noise_ppm, n_cadences, n_outliers,
        coverage_fraction, median_flux)
compute_lc_stats(time, flux, *, flux_err, cadence_minutes, transit_duration_hours) -> LCStats
format_lc_stats(stats) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class LCStats:
    cdpp_ppm: float           # Combined differential photometric precision
    rms_ppm: float            # RMS scatter in ppm
    photon_noise_ppm: float   # Estimated photon noise floor in ppm
    n_cadences: int
    n_outliers: int           # Cadences > 5-sigma from median
    coverage_fraction: float  # Fraction of expected cadences present
    median_flux: float        # Median flux level (for noise floor estimate)


def _median(vals: list[float]) -> float:
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def _mad(vals: list[float], med: float) -> float:
    return _median([abs(v - med) for v in vals])


def _cdpp(
    time: list[float],
    flux: list[float],
    transit_duration_hours: float,
    cadence_minutes: float,
) -> float:
    """Estimate CDPP by sliding-window RMS on a timescale matching transit duration."""
    n_cadences_window = max(1, round(transit_duration_hours * 60.0 / cadence_minutes))
    n = len(flux)
    if n < n_cadences_window:
        rms_list = [abs(f) for f in flux]
        return _median(rms_list) * 1e6 if rms_list else 0.0

    window_means: list[float] = []
    for i in range(n - n_cadences_window + 1):
        chunk = flux[i : i + n_cadences_window]
        window_means.append(sum(chunk) / len(chunk))

    if not window_means:
        return 0.0
    med = _median(window_means)
    sq_sum = sum((x - med) ** 2 for x in window_means)
    rms = math.sqrt(sq_sum / len(window_means))
    return rms * 1e6


def compute_lc_stats(
    time: list[float],
    flux: list[float],
    *,
    flux_err: list[float] | None = None,
    cadence_minutes: float = 2.0,
    transit_duration_hours: float = 2.0,
) -> LCStats:
    """Compute quality statistics for a light curve.

    Args:
        time: BJD time array.
        flux: Normalised flux array (median ≈ 1.0).
        flux_err: Per-cadence uncertainties (optional).
        cadence_minutes: Nominal cadence in minutes.
        transit_duration_hours: Expected transit duration for CDPP estimate.

    Returns:
        :class:`LCStats`.
    """
    n = len(flux)
    if n == 0:
        return LCStats(
            cdpp_ppm=0.0, rms_ppm=0.0, photon_noise_ppm=0.0,
            n_cadences=0, n_outliers=0, coverage_fraction=0.0, median_flux=0.0,
        )

    med = _median(flux)
    mad = _mad(flux, med)
    mad_scaled = max(mad * 1.4826, 1e-9)

    # RMS in ppm
    sq_sum = sum((f - med) ** 2 for f in flux)
    rms_ppm = math.sqrt(sq_sum / n) * 1e6

    # Outliers (>5-sigma from median)
    n_outliers = sum(1 for f in flux if abs(f - med) > 5.0 * mad_scaled)

    # CDPP
    cdpp = _cdpp(time, flux, transit_duration_hours, cadence_minutes)

    # Photon noise floor: median of flux_err, or sqrt(median flux) approximation
    if flux_err is not None and len(flux_err) > 0:
        photon_noise_ppm = _median(flux_err) * 1e6
    else:
        photon_noise_ppm = (1.0 / math.sqrt(max(abs(med), 1e-9))) * 1e6

    # Coverage fraction
    if len(time) >= 2:
        span_days = float(time[-1]) - float(time[0])
        expected = max(1, round(span_days * 24.0 * 60.0 / cadence_minutes))
        coverage_fraction = min(1.0, n / expected)
    else:
        coverage_fraction = 1.0

    return LCStats(
        cdpp_ppm=round(cdpp, 3),
        rms_ppm=round(rms_ppm, 3),
        photon_noise_ppm=round(photon_noise_ppm, 3),
        n_cadences=n,
        n_outliers=n_outliers,
        coverage_fraction=round(coverage_fraction, 4),
        median_flux=round(med, 6),
    )


def format_lc_stats(stats: LCStats) -> str:
    """Format LCStats as a Markdown block."""
    lines = [
        "## Light Curve Statistics",
        "",
        f"- Cadences: {stats.n_cadences}",
        f"- Coverage: {stats.coverage_fraction:.1%}",
        f"- Median flux: {stats.median_flux:.6f}",
        f"- RMS: {stats.rms_ppm:.1f} ppm",
        f"- CDPP: {stats.cdpp_ppm:.1f} ppm",
        f"- Photon noise floor: {stats.photon_noise_ppm:.1f} ppm",
        f"- Outliers (>5σ): {stats.n_outliers}",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(
        prog="lc_statistics",
        description="Compute light curve quality statistics.",
    )
    parser.add_argument("--lc", required=True, metavar="JSON")
    parser.add_argument("--cadence", type=float, default=2.0,
                        help="Cadence in minutes.")
    parser.add_argument("--duration", type=float, default=2.0,
                        help="Transit duration in hours for CDPP.")
    args = parser.parse_args(argv)

    lc = json.loads(Path(args.lc).read_text())
    stats = compute_lc_stats(
        lc["time"], lc["flux"],
        cadence_minutes=args.cadence,
        transit_duration_hours=args.duration,
    )
    print(format_lc_stats(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
