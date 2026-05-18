"""Filter stellar-activity flares and spot-crossing events from light curves.

Uses sigma-clipping on the residuals after a smoothed baseline to flag
likely flare/activity cadences.

Public API
----------
ActivityFilterResult(n_flagged, flagged_indices, flare_times, activity_fraction,
                     baseline_rms_ppm, flag)
filter_stellar_activity(time, flux, *, flux_err, window_points,
                        sigma_upper, sigma_lower) -> ActivityFilterResult
apply_activity_mask(time, flux, flux_err, result) -> tuple
format_activity_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ActivityFilterResult:
    n_flagged: int
    flagged_indices: tuple[int, ...]
    flare_times: tuple[float, ...]
    activity_fraction: float
    baseline_rms_ppm: float
    flag: str                  # "QUIET", "ACTIVE", "VERY_ACTIVE"


def _running_median(values: list[float], half_window: int) -> list[float]:
    """Simple running-median with edge reflection padding."""
    n = len(values)
    result = []
    for i in range(n):
        lo = max(0, i - half_window)
        hi = min(n, i + half_window + 1)
        window = sorted(values[lo:hi])
        m = len(window)
        if m % 2 == 1:
            result.append(window[m // 2])
        else:
            result.append((window[m // 2 - 1] + window[m // 2]) / 2.0)
    return result


def filter_stellar_activity(
    time: list[float],
    flux: list[float],
    *,
    flux_err: list[float] | None = None,
    window_points: int = 49,
    sigma_upper: float = 3.0,
    sigma_lower: float = 5.0,
    active_threshold: float = 0.05,
    very_active_threshold: float = 0.15,
) -> ActivityFilterResult:
    """Identify flare/activity cadences via running-median sigma-clip.

    Args:
        time: BJD time array.
        flux: Normalised flux array.
        flux_err: Per-cadence uncertainties (unused in current implementation).
        window_points: Running-median window size (should be odd).
        sigma_upper: Upper sigma clip threshold (flares are upward).
        sigma_lower: Lower sigma clip threshold (spot crossings).
        active_threshold: Fraction threshold for ACTIVE flag.
        very_active_threshold: Fraction threshold for VERY_ACTIVE flag.

    Returns:
        :class:`ActivityFilterResult`.
    """
    if not time:
        return ActivityFilterResult(0, (), (), 0.0, 0.0, "QUIET")

    n = len(flux)
    f_arr = [float(f) for f in flux]
    t_arr = [float(t) for t in time]
    half_w = window_points // 2

    baseline = _running_median(f_arr, half_w)
    residuals = [f - b for f, b in zip(f_arr, baseline, strict=False)]

    # RMS of residuals
    rms = math.sqrt(sum(r ** 2 for r in residuals) / n) if n > 0 else 0.0
    rms_ppm = rms * 1e6

    if rms == 0.0:
        return ActivityFilterResult(0, (), (), 0.0, 0.0, "QUIET")

    flagged: list[int] = []
    for i, r in enumerate(residuals):
        if r / rms > sigma_upper or r / rms < -sigma_lower:
            flagged.append(i)

    n_flagged = len(flagged)
    fraction = n_flagged / n if n > 0 else 0.0
    flare_times = tuple(t_arr[i] for i in flagged)

    if fraction >= very_active_threshold:
        flag = "VERY_ACTIVE"
    elif fraction >= active_threshold:
        flag = "ACTIVE"
    else:
        flag = "QUIET"

    return ActivityFilterResult(
        n_flagged=n_flagged,
        flagged_indices=tuple(flagged),
        flare_times=flare_times,
        activity_fraction=round(fraction, 5),
        baseline_rms_ppm=round(rms_ppm, 2),
        flag=flag,
    )


def apply_activity_mask(
    time: list[float],
    flux: list[float],
    flux_err: list[float] | None,
    result: ActivityFilterResult,
) -> tuple[list[float], list[float], list[float] | None]:
    """Remove flagged cadences from time/flux arrays.

    Returns:
        (time_clean, flux_clean, flux_err_clean) with flagged cadences removed.
    """
    bad = set(result.flagged_indices)
    t_clean = [t for i, t in enumerate(time) if i not in bad]
    f_clean = [f for i, f in enumerate(flux) if i not in bad]
    e_clean = (
        [e for i, e in enumerate(flux_err) if i not in bad]
        if flux_err is not None else None
    )
    return t_clean, f_clean, e_clean


def format_activity_result(result: ActivityFilterResult) -> str:
    """Format activity filter result as Markdown."""
    lines = [
        "## Stellar Activity Filter",
        "",
        f"- Baseline RMS: {result.baseline_rms_ppm:.1f} ppm",
        f"- Flagged cadences: {result.n_flagged} ({result.activity_fraction * 100:.2f}%)",
        f"- Flag: **{result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(
        prog="stellar_activity_filter",
        description="Flag flare/activity cadences in a light curve.",
    )
    parser.add_argument("--lc", required=True, metavar="JSON")
    parser.add_argument("--window", type=int, default=49)
    parser.add_argument("--sigma-upper", type=float, default=3.0)
    parser.add_argument("--sigma-lower", type=float, default=5.0)
    args = parser.parse_args(argv)

    lc = json.loads(Path(args.lc).read_text())
    result = filter_stellar_activity(
        lc["time"], lc["flux"],
        window_points=args.window,
        sigma_upper=args.sigma_upper,
        sigma_lower=args.sigma_lower,
    )
    print(format_activity_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
