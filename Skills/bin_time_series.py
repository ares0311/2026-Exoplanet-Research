"""General-purpose time-series binner.

Given flux and time arrays, returns binned arrays at a target cadence with per-bin RMS.

Public API:
    BinResult  -- frozen dataclass
    bin_time_series(time, flux, bin_width_days) -> BinResult
    format_bin_result(result) -> str
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BinResult:
    n_input: int
    n_bins: int
    bin_width_days: float
    bin_centers: list[float]
    bin_flux: list[float]
    bin_rms: list[float]
    bin_counts: list[int]
    flag: str


def bin_time_series(
    time: list[float],
    flux: list[float],
    bin_width_days: float,
) -> BinResult:
    n = min(len(time), len(flux))
    if n < 2:
        return BinResult(
            n_input=n, n_bins=0, bin_width_days=bin_width_days,
            bin_centers=[], bin_flux=[], bin_rms=[], bin_counts=[], flag="INSUFFICIENT_DATA",
        )
    if bin_width_days <= 0:
        return BinResult(
            n_input=n, n_bins=0, bin_width_days=bin_width_days,
            bin_centers=[], bin_flux=[], bin_rms=[], bin_counts=[], flag="INVALID_BIN_WIDTH",
        )
    t = time[:n]
    f = flux[:n]
    t_min = min(t)
    t_max = max(t)
    if t_max <= t_min:
        return BinResult(
            n_input=n, n_bins=0, bin_width_days=bin_width_days,
            bin_centers=[], bin_flux=[], bin_rms=[], bin_counts=[], flag="ZERO_SPAN",
        )
    n_bins = math.ceil((t_max - t_min) / bin_width_days)
    bins: list[list[float]] = [[] for _ in range(n_bins)]
    for ti, fi in zip(t, f, strict=False):
        idx = min(int((ti - t_min) / bin_width_days), n_bins - 1)
        bins[idx].append(fi)
    centers: list[float] = []
    bin_flux_out: list[float] = []
    bin_rms_out: list[float] = []
    bin_counts_out: list[int] = []
    for i, b in enumerate(bins):
        if not b:
            continue
        center = t_min + (i + 0.5) * bin_width_days
        mean = sum(b) / len(b)
        rms = math.sqrt(sum((v - mean) ** 2 for v in b) / len(b)) if len(b) > 1 else 0.0
        centers.append(center)
        bin_flux_out.append(mean)
        bin_rms_out.append(rms)
        bin_counts_out.append(len(b))
    return BinResult(
        n_input=n,
        n_bins=len(centers),
        bin_width_days=bin_width_days,
        bin_centers=centers,
        bin_flux=bin_flux_out,
        bin_rms=bin_rms_out,
        bin_counts=bin_counts_out,
        flag="OK",
    )


def format_bin_result(result: BinResult) -> str:
    lines = [
        "## Time Series Binning",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Input Points | {result.n_input} |",
        f"| Output Bins | {result.n_bins} |",
        f"| Bin Width (days) | {result.bin_width_days:.6f} |",
        f"| Flag | {result.flag} |",
    ]
    return "\n".join(lines)


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Bin a time series at a target cadence.")
    parser.add_argument("time_file", help="JSON file of time values.")
    parser.add_argument("flux_file", help="JSON file of flux values.")
    parser.add_argument("bin_width_days", type=float)
    args = parser.parse_args()
    with open(args.time_file) as fh:
        time = json.load(fh)
    with open(args.flux_file) as fh:
        flux = json.load(fh)
    result = bin_time_series(time, flux, args.bin_width_days)
    print(format_bin_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
