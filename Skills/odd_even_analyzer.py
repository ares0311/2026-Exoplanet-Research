"""Rigorous odd/even transit depth analysis.

Computes per-transit depth measurements and tests for asymmetry between
odd-numbered and even-numbered transits — a key eclipsing binary diagnostic.

Public API
----------
OddEvenResult(depth_odd_ppm, depth_even_ppm, err_odd_ppm, err_even_ppm,
              delta_ppm, sigma_asymmetry, n_odd, n_even,
              is_asymmetric, flag)
analyze_odd_even(time, flux, period, epoch, *, flux_err, duration_days,
                 asymmetry_threshold_sigma) -> OddEvenResult
format_odd_even_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class OddEvenResult:
    depth_odd_ppm: float | None
    depth_even_ppm: float | None
    err_odd_ppm: float | None
    err_even_ppm: float | None
    delta_ppm: float | None          # |odd - even|
    sigma_asymmetry: float | None    # |odd-even| / combined error
    n_odd: int
    n_even: int
    is_asymmetric: bool
    flag: str                        # "PASS", "WARN", "FAIL", "INSUFFICIENT"


def _weighted_mean_err(depths: list[float], errs: list[float] | None) -> tuple[float, float]:
    """Return (weighted mean depth, error on mean) in ppm."""
    n = len(depths)
    if n == 0:
        return 0.0, 0.0
    if errs is not None and len(errs) == n and all(e > 0 for e in errs):
        weights = [1.0 / e ** 2 for e in errs]
        w_sum = sum(weights)
        mean = sum(d * w for d, w in zip(depths, weights, strict=False)) / w_sum
        err = 1.0 / math.sqrt(w_sum)
    else:
        mean = sum(depths) / n
        if n > 1:
            sq = sum((d - mean) ** 2 for d in depths)
            err = math.sqrt(sq / (n - 1)) / math.sqrt(n)
            # Identical synthetic depths should not imply perfect measurement precision.
            err = max(err, abs(mean) * 1e-3, 1.0)
        else:
            err = abs(mean) * 0.1 or 0.1
    return mean, err


def analyze_odd_even(
    time: list[float],
    flux: list[float],
    period: float,
    epoch: float,
    *,
    flux_err: list[float] | None = None,
    duration_days: float = 0.1,
    asymmetry_threshold_sigma: float = 3.0,
) -> OddEvenResult:
    """Measure odd- and even-transit depths and test for asymmetry.

    Args:
        time: BJD time array.
        flux: Normalised flux array.
        period: Orbital period in days.
        epoch: Reference mid-transit epoch (BJD).
        flux_err: Per-cadence uncertainties.
        duration_days: Transit duration for in-transit window.
        asymmetry_threshold_sigma: Sigma threshold for FAIL flag.

    Returns:
        :class:`OddEvenResult`.
    """
    if not time or period <= 0:
        return OddEvenResult(None, None, None, None, None, None, 0, 0, False, "INSUFFICIENT")

    half_dur = duration_days / 2.0
    t_arr = [float(t) for t in time]
    f_arr = [float(f) for f in flux]
    e_arr = ([float(e) for e in flux_err] if flux_err is not None else None)

    t_min, t_max = t_arr[0], t_arr[-1]
    n_start = math.ceil((t_min - epoch) / period)
    n_end = math.floor((t_max - epoch) / period)

    odd_depths: list[float] = []
    odd_errs: list[float] = []
    even_depths: list[float] = []
    even_errs: list[float] = []

    for n in range(int(n_start), int(n_end) + 1):
        t_mid = epoch + n * period
        if t_mid < t_min or t_mid > t_max:
            continue

        in_f = [f for t, f in zip(t_arr, f_arr, strict=False) if abs(t - t_mid) <= half_dur]
        in_e = (
            [e for t, e in zip(t_arr, e_arr, strict=False) if abs(t - t_mid) <= half_dur]
            if e_arr is not None else None
        )

        if len(in_f) < 2:
            continue

        oot_f = [f for t, f in zip(t_arr, f_arr, strict=False) if abs(t - t_mid) > half_dur * 2]
        baseline = sum(oot_f) / len(oot_f) if oot_f else 1.0
        depth = (baseline - sum(in_f) / len(in_f)) * 1e6

        if depth < 0:
            continue

        err_val = None
        if in_e is not None and len(in_e) > 0:
            err_val = (sum(e ** 2 for e in in_e) ** 0.5 / len(in_e)) * 1e6

        if n % 2 == 1:
            odd_depths.append(depth)
            if err_val is not None:
                odd_errs.append(err_val)
        else:
            even_depths.append(depth)
            if err_val is not None:
                even_errs.append(err_val)

    n_odd = len(odd_depths)
    n_even = len(even_depths)

    if n_odd < 2 or n_even < 2:
        return OddEvenResult(
            None, None, None, None, None, None, n_odd, n_even, False, "INSUFFICIENT"
        )

    d_odd, e_odd = _weighted_mean_err(odd_depths, odd_errs if odd_errs else None)
    d_even, e_even = _weighted_mean_err(even_depths, even_errs if even_errs else None)

    delta = abs(d_odd - d_even)
    combined_err = math.sqrt(e_odd ** 2 + e_even ** 2) if (e_odd and e_even) else 1.0
    sigma = delta / combined_err if combined_err > 0 else 0.0

    is_asym = sigma > asymmetry_threshold_sigma
    if is_asym:
        flag = "FAIL"
    elif sigma > asymmetry_threshold_sigma * 0.67:
        flag = "WARN"
    else:
        flag = "PASS"

    return OddEvenResult(
        depth_odd_ppm=round(d_odd, 2),
        depth_even_ppm=round(d_even, 2),
        err_odd_ppm=round(e_odd, 2),
        err_even_ppm=round(e_even, 2),
        delta_ppm=round(delta, 2),
        sigma_asymmetry=round(sigma, 3),
        n_odd=n_odd,
        n_even=n_even,
        is_asymmetric=is_asym,
        flag=flag,
    )


def format_odd_even_result(result: OddEvenResult) -> str:
    """Format odd/even result as Markdown."""
    lines = ["## Odd/Even Transit Depth Analysis", ""]
    if result.flag == "INSUFFICIENT":
        lines.append(f"- Insufficient transits (odd={result.n_odd}, even={result.n_even})")
    else:
        lines += [
            "| | Depth (ppm) | Error (ppm) | N transits |",
            "|---|---|---|---|",
            f"| Odd  | {result.depth_odd_ppm:.1f} | {result.err_odd_ppm:.1f} | {result.n_odd} |",
            f"| Even | {result.depth_even_ppm:.1f} | {result.err_even_ppm:.1f} | {result.n_even} |",
            "",
            f"- |Δdepth|: {result.delta_ppm:.1f} ppm",
            f"- Asymmetry significance: {result.sigma_asymmetry:.2f}σ",
            f"- Flag: **{result.flag}**",
        ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(
        prog="odd_even_analyzer",
        description="Analyze odd/even transit depth asymmetry.",
    )
    parser.add_argument("--lc", required=True, metavar="JSON")
    parser.add_argument("--period", type=float, required=True)
    parser.add_argument("--epoch", type=float, required=True)
    parser.add_argument("--duration", type=float, default=0.1)
    args = parser.parse_args(argv)

    lc = json.loads(Path(args.lc).read_text())
    result = analyze_odd_even(
        lc["time"], lc["flux"], args.period, args.epoch,
        duration_days=args.duration,
    )
    print(format_odd_even_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
