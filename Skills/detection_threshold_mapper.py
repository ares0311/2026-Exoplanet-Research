"""Map minimum detectable transit depth across a period grid.

For each period, computes the minimum signal depth needed to achieve a
given SNR threshold given the white noise and number of transits available.

Public API
----------
ThresholdMapResult(periods_days, min_depths_ppm, n_transits, snr_threshold, flag)
map_detection_thresholds(periods_days, white_noise_ppm, n_transits_list, snr_threshold) -> ThresholdMapResult
format_threshold_map(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ThresholdMapResult:
    periods_days: tuple[float, ...]
    min_depths_ppm: tuple[float, ...]
    n_transits: tuple[int, ...]
    snr_threshold: float
    flag: str  # "OK" or "SHALLOW_COVERAGE"(any depth > 5000 ppm)


def map_detection_thresholds(
    periods_days: list[float],
    white_noise_ppm: float,
    n_transits_list: list[int],
    snr_threshold: float = 7.0,
) -> ThresholdMapResult:
    """Map minimum detectable depth across a period grid.

    Args:
        periods_days: List of orbital periods to evaluate [days].
        white_noise_ppm: Per-cadence white noise in ppm.
        n_transits_list: Number of transits expected at each period.
        snr_threshold: Required SNR for detection (default 7.0).

    Returns:
        ThresholdMapResult with per-period minimum detectable depths.
    """
    if len(periods_days) != len(n_transits_list):
        raise ValueError("periods_days and n_transits_list must have the same length")

    min_depths: list[float] = []
    for n in n_transits_list:
        n_safe = max(int(n), 1)
        depth = snr_threshold * white_noise_ppm / math.sqrt(n_safe)
        min_depths.append(round(depth, 2))

    flag = "SHALLOW_COVERAGE" if any(d > 5000.0 for d in min_depths) else "OK"

    return ThresholdMapResult(
        periods_days=tuple(float(p) for p in periods_days),
        min_depths_ppm=tuple(min_depths),
        n_transits=tuple(int(n) for n in n_transits_list),
        snr_threshold=float(snr_threshold),
        flag=flag,
    )


def format_threshold_map(result: ThresholdMapResult) -> str:
    """Format detection threshold map as Markdown.

    Args:
        result: ThresholdMapResult to format.

    Returns:
        Markdown string with a table of periods and min depths.
    """
    lines = [
        "## Detection Threshold Map",
        "",
        f"- SNR threshold: {result.snr_threshold:.1f}",
        f"- Status: `{result.flag}`",
        "",
        "| Period (d) | N transits | Min depth (ppm) |",
        "|---|---|---|",
    ]
    for p, n, d in zip(result.periods_days, result.n_transits, result.min_depths_ppm):
        lines.append(f"| {p:.3f} | {n} | {d:.1f} |")
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Map detection thresholds.")
    parser.add_argument("--noise-ppm", type=float, required=True,
                        help="White noise per cadence in ppm.")
    parser.add_argument("--snr", type=float, default=7.0)
    parser.add_argument("--periods", nargs="+", type=float, required=True)
    parser.add_argument("--n-transits", nargs="+", type=int, required=True)
    args = parser.parse_args(argv)

    result = map_detection_thresholds(
        args.periods, args.noise_ppm, args.n_transits, args.snr
    )
    print(format_threshold_map(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
