"""Compare in-transit flux to out-of-transit baseline for depth verification.

Computes the mean in-transit flux, the OOT baseline, the measured depth,
and a consistency score comparing the measured depth to the expected depth.

Public API
----------
BaselineComparison(period_days, epoch_btjd, duration_hours,
                   oot_mean, oot_std, in_transit_mean, measured_depth_ppm,
                   expected_depth_ppm, depth_ratio, n_in_transit,
                   n_out_transit, flag)
compare_transit_baseline(time, flux, *, period_days, epoch_btjd,
                          duration_hours, expected_depth_ppm) -> BaselineComparison
format_baseline_comparison(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BaselineComparison:
    period_days: float
    epoch_btjd: float
    duration_hours: float
    oot_mean: float
    oot_std: float
    in_transit_mean: float
    measured_depth_ppm: float
    expected_depth_ppm: float | None
    depth_ratio: float | None  # measured / expected; 1.0 = perfect match
    n_in_transit: int
    n_out_transit: int
    flag: str  # "OK" | "SHALLOW" | "DEEP" | "SPARSE" | "INVALID"


def compare_transit_baseline(
    time: list[float] | tuple[float, ...],
    flux: list[float] | tuple[float, ...],
    *,
    period_days: float,
    epoch_btjd: float,
    duration_hours: float,
    expected_depth_ppm: float | None = None,
    depth_tolerance: float = 0.30,
) -> BaselineComparison:
    """Compare in-transit flux to the OOT baseline.

    Args:
        time: Observation timestamps in BTJD days.
        flux: Normalised flux array (OOT ≈ 1.0).
        period_days: Orbital period.
        epoch_btjd: Transit epoch.
        duration_hours: Transit duration in hours.
        expected_depth_ppm: Expected depth from BLS/model; used for ratio.
        depth_tolerance: Fractional tolerance for depth ratio consistency.

    Returns:
        BaselineComparison with measured depth and depth ratio.
    """
    if period_days <= 0 or duration_hours <= 0:
        return BaselineComparison(
            period_days=period_days,
            epoch_btjd=epoch_btjd,
            duration_hours=duration_hours,
            oot_mean=0.0, oot_std=0.0, in_transit_mean=0.0,
            measured_depth_ppm=0.0,
            expected_depth_ppm=expected_depth_ppm,
            depth_ratio=None,
            n_in_transit=0, n_out_transit=0,
            flag="INVALID",
        )

    t = list(time)
    f = list(flux)
    half_dur = duration_hours / 24.0 / 2.0

    in_transit: list[float] = []
    out_transit: list[float] = []

    for ti, fi in zip(t, f, strict=False):
        phase = ((ti - epoch_btjd) / period_days) % 1.0
        if phase > 0.5:
            phase -= 1.0
        if abs(phase) * period_days < half_dur:
            in_transit.append(fi)
        else:
            out_transit.append(fi)

    if len(in_transit) < 2 or len(out_transit) < 2:
        return BaselineComparison(
            period_days=period_days,
            epoch_btjd=epoch_btjd,
            duration_hours=duration_hours,
            oot_mean=0.0, oot_std=0.0, in_transit_mean=0.0,
            measured_depth_ppm=0.0,
            expected_depth_ppm=expected_depth_ppm,
            depth_ratio=None,
            n_in_transit=len(in_transit),
            n_out_transit=len(out_transit),
            flag="SPARSE",
        )

    oot_mean = sum(out_transit) / len(out_transit)
    oot_var = sum((x - oot_mean) ** 2 for x in out_transit) / (len(out_transit) - 1)
    oot_std = math.sqrt(oot_var)
    in_mean = sum(in_transit) / len(in_transit)
    depth_ppm = max((oot_mean - in_mean) * 1e6, 0.0)

    depth_ratio: float | None = None
    if expected_depth_ppm is not None and expected_depth_ppm > 0:
        depth_ratio = round(depth_ppm / expected_depth_ppm, 4)

    if len(in_transit) < 3:
        flag = "SPARSE"
    elif depth_ratio is not None:
        if depth_ratio < (1.0 - depth_tolerance):
            flag = "SHALLOW"
        elif depth_ratio > (1.0 + depth_tolerance):
            flag = "DEEP"
        else:
            flag = "OK"
    else:
        flag = "OK"

    return BaselineComparison(
        period_days=period_days,
        epoch_btjd=epoch_btjd,
        duration_hours=duration_hours,
        oot_mean=round(oot_mean, 6),
        oot_std=round(oot_std, 6),
        in_transit_mean=round(in_mean, 6),
        measured_depth_ppm=round(depth_ppm, 2),
        expected_depth_ppm=expected_depth_ppm,
        depth_ratio=depth_ratio,
        n_in_transit=len(in_transit),
        n_out_transit=len(out_transit),
        flag=flag,
    )


def format_baseline_comparison(result: BaselineComparison) -> str:
    """Format baseline comparison as Markdown.

    Args:
        result: BaselineComparison to format.

    Returns:
        Markdown string.
    """
    ratio_str = f"{result.depth_ratio:.3f}" if result.depth_ratio is not None else "—"
    exp_str = (f"{result.expected_depth_ppm:.0f} ppm"
               if result.expected_depth_ppm is not None else "—")
    lines = [
        "## Transit Baseline Comparison\n",
        f"**Status**: `{result.flag}` | Period: {result.period_days:.4f} d\n",
        "",
        "| Parameter | Value |",
        "|---|---|",
        f"| OOT mean | {result.oot_mean:.6f} |",
        f"| OOT std | {result.oot_std:.6f} |",
        f"| In-transit mean | {result.in_transit_mean:.6f} |",
        f"| Measured depth | {result.measured_depth_ppm:.1f} ppm |",
        f"| Expected depth | {exp_str} |",
        f"| Depth ratio (meas/exp) | {ratio_str} |",
        f"| N in-transit | {result.n_in_transit} |",
        f"| N out-of-transit | {result.n_out_transit} |",
    ]
    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Compare transit baseline.")
    parser.add_argument("lc", help="Light curve JSON with 'time' and 'flux'.")
    parser.add_argument("--period", type=float, required=True)
    parser.add_argument("--epoch", type=float, required=True)
    parser.add_argument("--duration-hours", type=float, required=True)
    parser.add_argument("--expected-depth-ppm", type=float, default=None)
    args = parser.parse_args(argv)

    from pathlib import Path
    data = json.loads(Path(args.lc).read_text())
    result = compare_transit_baseline(
        data["time"], data["flux"],
        period_days=args.period,
        epoch_btjd=args.epoch,
        duration_hours=args.duration_hours,
        expected_depth_ppm=args.expected_depth_ppm,
    )
    print(format_baseline_comparison(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
