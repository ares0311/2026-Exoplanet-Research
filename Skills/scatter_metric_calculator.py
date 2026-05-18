"""Compute multiple scatter metrics for a light curve.

Provides RMS, median absolute deviation (MAD), 6-hour CDPP approximation,
and a normalised point-to-point scatter metric to characterise light curve
quality.

Public API
----------
ScatterMetricResult(n_points, rms_ppm, mad_ppm, cdpp_6hr_ppm,
                    point_to_point_ppm, flag)
compute_scatter_metrics(time, flux, *, cadence_days) -> ScatterMetricResult
format_scatter_metrics(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ScatterMetricResult:
    n_points: int
    rms_ppm: float
    mad_ppm: float
    cdpp_6hr_ppm: float
    point_to_point_ppm: float
    flag: str  # "OK", "INSUFFICIENT"


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def compute_scatter_metrics(
    time: list[float],
    flux: list[float],
    *,
    cadence_days: float = 2.0 / 1440.0,
) -> ScatterMetricResult:
    """Compute scatter metrics for a light curve.

    Args:
        time: Time array (BJD) — used only for point-to-point.
        flux: Normalised flux array.
        cadence_days: Typical cadence in days (default 2 min).

    Returns:
        :class:`ScatterMetricResult`.
    """
    n = len(flux)
    if n < 2:
        return ScatterMetricResult(n, 0.0, 0.0, 0.0, 0.0, "INSUFFICIENT")

    mean_f = sum(flux) / n
    flux_ppm = [(f - mean_f) * 1e6 for f in flux]

    # RMS
    rms_ppm = math.sqrt(sum(x ** 2 for x in flux_ppm) / n)

    # MAD
    med = _median(flux_ppm)
    mad_ppm = _median([abs(x - med) for x in flux_ppm]) * 1.4826

    # CDPP approximation: scale per-point RMS by sqrt(N per 6-hr window)
    n_per_6hr = max(1, int(round(6.0 / (cadence_days * 24.0))))
    cdpp_6hr_ppm = rms_ppm / math.sqrt(n_per_6hr)

    # Point-to-point scatter (normalised difference of adjacent points)
    diffs = [abs(flux_ppm[i + 1] - flux_ppm[i]) for i in range(n - 1)]
    ptp_ppm = _median(diffs) / math.sqrt(2)

    return ScatterMetricResult(
        n_points=n,
        rms_ppm=round(rms_ppm, 2),
        mad_ppm=round(mad_ppm, 2),
        cdpp_6hr_ppm=round(cdpp_6hr_ppm, 2),
        point_to_point_ppm=round(ptp_ppm, 2),
        flag="OK",
    )


def format_scatter_metrics(result: ScatterMetricResult) -> str:
    """Format scatter metrics result as Markdown."""
    lines = [
        "## Scatter Metrics",
        "",
        f"- Points: {result.n_points}",
    ]
    if result.flag == "INSUFFICIENT":
        lines.append("- **Flag: INSUFFICIENT** — fewer than 2 points")
    else:
        lines += [
            f"- RMS: {result.rms_ppm:.2f} ppm",
            f"- MAD (normalised): {result.mad_ppm:.2f} ppm",
            f"- CDPP (6 hr): {result.cdpp_6hr_ppm:.2f} ppm",
            f"- Point-to-point: {result.point_to_point_ppm:.2f} ppm",
            f"- **Flag: {result.flag}**",
        ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="scatter_metric_calculator",
        description="Compute scatter metrics for a light curve.",
    )
    parser.add_argument("--cadence-days", type=float, default=2.0 / 1440.0)
    args = parser.parse_args(argv)

    result = compute_scatter_metrics([], [], cadence_days=args.cadence_days)
    print(format_scatter_metrics(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
