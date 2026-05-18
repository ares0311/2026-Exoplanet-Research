"""Combine ground-based multi-night photometry into a single phase-folded dataset.

Normalises each night separately (baseline = median of out-of-transit flux),
then phase-folds and merges all nights for depth measurement and transit
confirmation.

Public API
----------
NightResult(night_id, n_points, baseline, depth_ppm, has_transit)
CombinedPhotometryResult(nights, n_nights, combined_depth_ppm,
                          combined_depth_err_ppm, n_in_transit, flag)
combine_photometry_nights(nights, period_days, epoch_bjd, *,
                           duration_hours, min_in_transit) -> CombinedPhotometryResult
format_combined_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class NightResult:
    night_id: str
    n_points: int
    baseline: float
    depth_ppm: float | None
    has_transit: bool


@dataclass(frozen=True)
class CombinedPhotometryResult:
    nights: tuple[NightResult, ...]
    n_nights: int
    combined_depth_ppm: float | None
    combined_depth_err_ppm: float | None
    n_in_transit: int
    flag: str                   # "CONFIRMED", "MARGINAL", "NO_TRANSIT", "INSUFFICIENT"


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 1.0
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def combine_photometry_nights(
    nights: list[dict],
    period_days: float,
    epoch_bjd: float,
    *,
    duration_hours: float = 2.0,
    min_in_transit: int = 3,
    confirmed_snr: float = 5.0,
    marginal_snr: float = 3.0,
) -> CombinedPhotometryResult:
    """Combine multi-night photometry and measure transit depth.

    Args:
        nights: List of dicts, each with keys ``id``, ``time`` (list[float]),
            ``flux`` (list[float]).
        period_days: Orbital period in days.
        epoch_bjd: Transit epoch (BJD).
        duration_hours: Transit duration for in/out classification.
        min_in_transit: Minimum in-transit points to measure depth.
        confirmed_snr: SNR threshold for CONFIRMED flag.
        marginal_snr: SNR threshold for MARGINAL flag.

    Returns:
        :class:`CombinedPhotometryResult`.
    """
    if not nights or period_days <= 0:
        return CombinedPhotometryResult((), 0, None, None, 0, "INSUFFICIENT")

    half_dur_d = duration_hours / 24.0 / 2.0
    night_results: list[NightResult] = []
    all_in_flux: list[float] = []
    all_out_flux: list[float] = []

    for night in nights:
        nid = str(night.get("id", "?"))
        t_arr = [float(t) for t in night.get("time", [])]
        f_arr = [float(f) for f in night.get("flux", [])]
        if len(t_arr) < 2:
            night_results.append(NightResult(nid, 0, 1.0, None, False))
            continue

        # Phase relative to nearest transit
        phases_d = [((t - epoch_bjd) % period_days) for t in t_arr]
        phases_d = [p if p <= period_days / 2 else p - period_days for p in phases_d]

        out_f = [f for ph, f in zip(phases_d, f_arr, strict=False) if abs(ph) > half_dur_d * 2]

        baseline = _median(out_f) if out_f else _median(f_arr)
        if baseline == 0.0:
            baseline = 1.0

        # Normalise this night
        f_norm = [f / baseline for f in f_arr]
        in_f_norm = [
            f for ph, f in zip(phases_d, f_norm, strict=False) if abs(ph) <= half_dur_d
        ]
        out_f_norm = [
            f for ph, f in zip(phases_d, f_norm, strict=False) if abs(ph) > half_dur_d * 2
        ]

        has_transit = len(in_f_norm) >= min_in_transit
        depth_ppm: float | None = None
        if has_transit:
            mean_in = sum(in_f_norm) / len(in_f_norm)
            depth_ppm = (1.0 - mean_in) * 1e6

        all_in_flux.extend(in_f_norm)
        all_out_flux.extend(out_f_norm)

        night_results.append(NightResult(
            night_id=nid,
            n_points=len(t_arr),
            baseline=round(baseline, 6),
            depth_ppm=round(depth_ppm, 2) if depth_ppm is not None else None,
            has_transit=has_transit,
        ))

    n_in = len(all_in_flux)
    if n_in < min_in_transit:
        return CombinedPhotometryResult(
            tuple(night_results), len(night_results), None, None, n_in, "INSUFFICIENT"
        )

    # Combined depth
    mean_in = sum(all_in_flux) / n_in
    combined_depth = (1.0 - mean_in) * 1e6

    # Error from scatter
    if len(all_out_flux) > 1:
        mean_out = sum(all_out_flux) / len(all_out_flux)
        scatter_sq = sum((f - mean_out) ** 2 for f in all_out_flux)
        sigma_out = math.sqrt(scatter_sq / (len(all_out_flux) - 1))
        depth_err = sigma_out / math.sqrt(n_in) * 1e6
    else:
        depth_err = abs(combined_depth) * 0.1 or 100.0

    snr = combined_depth / depth_err if depth_err > 0 else 0.0

    if snr >= confirmed_snr:
        flag = "CONFIRMED"
    elif snr >= marginal_snr:
        flag = "MARGINAL"
    else:
        flag = "NO_TRANSIT"

    return CombinedPhotometryResult(
        nights=tuple(night_results),
        n_nights=len(night_results),
        combined_depth_ppm=round(combined_depth, 2),
        combined_depth_err_ppm=round(depth_err, 2),
        n_in_transit=n_in,
        flag=flag,
    )


def format_combined_result(result: CombinedPhotometryResult) -> str:
    """Format combined photometry result as Markdown."""
    lines = ["## Multi-Night Photometry Combination", ""]
    if result.flag == "INSUFFICIENT":
        lines.append(f"- Insufficient in-transit data (n={result.n_in_transit}).")
    else:
        err_str = (
            f" ± {result.combined_depth_err_ppm:.1f}"
            if result.combined_depth_err_ppm is not None else ""
        )
        lines += [
            f"- Nights combined: {result.n_nights}",
            f"- In-transit points: {result.n_in_transit}",
            f"- Combined depth: **{result.combined_depth_ppm:.1f}{err_str} ppm**",
            f"- Flag: **{result.flag}**",
            "",
            "| Night | N pts | Depth (ppm) | Transit? |",
            "|---|---|---|---|",
        ]
        for nr in result.nights:
            d = f"{nr.depth_ppm:.1f}" if nr.depth_ppm is not None else "—"
            t = "Yes" if nr.has_transit else "No"
            lines.append(f"| {nr.night_id} | {nr.n_points} | {d} | {t} |")
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(
        prog="multi_night_photometry_combiner",
        description="Combine multi-night ground photometry for transit confirmation.",
    )
    parser.add_argument("--nights", required=True, metavar="JSON")
    parser.add_argument("--period", type=float, required=True)
    parser.add_argument("--epoch", type=float, required=True)
    parser.add_argument("--duration-hours", type=float, default=2.0)
    args = parser.parse_args(argv)

    nights = json.loads(Path(args.nights).read_text())
    result = combine_photometry_nights(
        nights, args.period, args.epoch, duration_hours=args.duration_hours
    )
    print(format_combined_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
