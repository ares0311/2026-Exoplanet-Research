"""Detect stellar rotation periods via the autocorrelation function (ACF).

Independent of BLS — useful for identifying stellar rotation as a false-positive
source and for filtering out rotationally-modulated variability.

Public API
----------
ACFResult(lags_days, acf_values, period_days, period_uncertainty_days,
          peak_acf, n_peaks, flag)
compute_acf(time, flux, *, max_lag_days, oversample) -> ACFResult
find_acf_period(result, *, min_period_days, max_period_days) -> ACFResult
format_acf_result(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ACFResult:
    lags_days: tuple[float, ...]
    acf_values: tuple[float, ...]
    period_days: float | None
    period_uncertainty_days: float | None
    peak_acf: float | None
    n_peaks: int
    flag: str  # "OK" | "NO_PERIOD" | "INSUFFICIENT" | "INVALID"


def _interpolate_onto_grid(
    time: list[float], flux: list[float], dt: float
) -> tuple[list[float], list[float]]:
    """Resample onto a uniform grid using linear interpolation."""
    if not time:
        return [], []
    t0 = min(time)
    t1 = max(time)
    n = max(2, int((t1 - t0) / dt) + 1)
    grid_t = [t0 + i * dt for i in range(n)]
    grid_f = []
    j = 0
    for gt in grid_t:
        while j < len(time) - 2 and time[j + 1] < gt:
            j += 1
        if j >= len(time) - 1:
            grid_f.append(flux[-1])
        else:
            dt_seg = time[j + 1] - time[j]
            if dt_seg < 1e-12:
                grid_f.append(flux[j])
            else:
                alpha = (gt - time[j]) / dt_seg
                grid_f.append(flux[j] + alpha * (flux[j + 1] - flux[j]))
    return grid_t, grid_f


def compute_acf(
    time: list[float],
    flux: list[float],
    *,
    max_lag_days: float = 30.0,
    oversample: int = 4,
) -> ACFResult:
    """Compute the autocorrelation function of a light curve.

    Args:
        time: Time array (days).
        flux: Flux array, same length as time.
        max_lag_days: Maximum lag to compute (days).
        oversample: Grid oversampling factor relative to median cadence.

    Returns:
        :class:`ACFResult`.
    """
    n = len(time)
    if n < 10 or len(flux) != n:
        return ACFResult((), (), None, None, None, 0, "INVALID")

    # Estimate median cadence
    diffs = sorted(time[i + 1] - time[i] for i in range(n - 1) if time[i + 1] > time[i])
    if not diffs:
        return ACFResult((), (), None, None, None, 0, "INVALID")
    med_cad = diffs[len(diffs) // 2]
    dt = med_cad / max(oversample, 1)

    grid_t, grid_f = _interpolate_onto_grid(time, flux, dt)
    m = len(grid_f)
    if m < 4:
        return ACFResult((), (), None, None, None, 0, "INSUFFICIENT")

    mean_f = sum(grid_f) / m
    centered = [f - mean_f for f in grid_f]
    var = sum(c ** 2 for c in centered) / m
    if var < 1e-20:
        return ACFResult((), (), None, None, None, 0, "INSUFFICIENT")

    max_lag_pts = min(m - 1, int(max_lag_days / dt))
    lags: list[float] = []
    acf: list[float] = []

    for k in range(0, max_lag_pts + 1):
        cov = sum(centered[i] * centered[i + k] for i in range(m - k)) / (m - k)
        lags.append(round(k * dt, 6))
        acf.append(round(cov / var, 6))

    return ACFResult(
        lags_days=tuple(lags),
        acf_values=tuple(acf),
        period_days=None,
        period_uncertainty_days=None,
        peak_acf=None,
        n_peaks=0,
        flag="OK",
    )


def find_acf_period(
    result: ACFResult,
    *,
    min_period_days: float = 1.0,
    max_period_days: float = 30.0,
) -> ACFResult:
    """Find the dominant period from an ACF result.

    Args:
        result: Output of :func:`compute_acf`.
        min_period_days: Minimum period to search.
        max_period_days: Maximum period to search.

    Returns:
        Updated :class:`ACFResult` with ``period_days`` filled.
    """
    if result.flag not in ("OK",) or len(result.lags_days) < 3:
        return result

    lags = list(result.lags_days)
    acf = list(result.acf_values)
    n = len(lags)

    # Find local maxima (excluding lag=0)
    peaks: list[tuple[float, float]] = []
    for i in range(1, n - 1):
        if (acf[i] > acf[i - 1] and acf[i] > acf[i + 1]
                and min_period_days <= lags[i] <= max_period_days):
            peaks.append((lags[i], acf[i]))

    if not peaks:
        return ACFResult(
            lags_days=result.lags_days,
            acf_values=result.acf_values,
            period_days=None,
            period_uncertainty_days=None,
            peak_acf=None,
            n_peaks=0,
            flag="NO_PERIOD",
        )

    # Best peak = highest ACF value in valid range
    best_lag, best_acf = max(peaks, key=lambda p: p[1])

    # Uncertainty: half-width at half-max of the peak
    peak_idx = next(i for i, lg in enumerate(lags) if abs(lg - best_lag) < 1e-9)
    half_max = best_acf / 2.0
    left = peak_idx
    right = peak_idx
    while left > 0 and acf[left] > half_max:
        left -= 1
    while right < n - 1 and acf[right] > half_max:
        right += 1
    hwhm = (lags[right] - lags[left]) / 2.0

    return ACFResult(
        lags_days=result.lags_days,
        acf_values=result.acf_values,
        period_days=round(best_lag, 4),
        period_uncertainty_days=round(hwhm, 4),
        peak_acf=round(best_acf, 4),
        n_peaks=len(peaks),
        flag="OK",
    )


def format_acf_result(result: ACFResult) -> str:
    """Format ACF result as Markdown."""
    lines = [
        "## Autocorrelation Period Finder",
        "",
        f"- ACF lags computed: {len(result.lags_days)}",
        f"- Detected period: {result.period_days} days",
        f"- Period uncertainty: {result.period_uncertainty_days} days",
        f"- Peak ACF value: {result.peak_acf}",
        f"- N peaks found: {result.n_peaks}",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="autocorrelation_period_finder",
        description="Find stellar rotation period via ACF.",
    )
    parser.add_argument("--max-lag", type=float, default=30.0)
    parser.add_argument("--min-period", type=float, default=1.0)
    parser.add_argument("--max-period", type=float, default=30.0)
    args = parser.parse_args(argv)

    result = compute_acf([], [], max_lag_days=args.max_lag)
    result = find_acf_period(
        result, min_period_days=args.min_period, max_period_days=args.max_period
    )
    print(format_acf_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
