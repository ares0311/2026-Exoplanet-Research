"""
Estimates the per-transit noise floor from out-of-transit (OOT) flux.

Public API:
    NoiseFloorResult     -- frozen dataclass holding noise floor diagnostics
    estimate_noise_floor(oot_flux, cadence_minutes, bin_minutes) -> NoiseFloorResult
    format_noise_floor(result) -> str
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class NoiseFloorResult:
    oot_rms_ppm: float
    n_points: int
    noise_floor_ppm: float
    cadence_minutes: float
    flag: str


def estimate_noise_floor(
    oot_flux: list[float],
    cadence_minutes: float = 2.0,
    bin_minutes: float = 30.0,
) -> NoiseFloorResult:
    n = len(oot_flux)
    if n == 0:
        return NoiseFloorResult(
            oot_rms_ppm=0.0,
            n_points=0,
            noise_floor_ppm=0.0,
            cadence_minutes=cadence_minutes,
            flag="NO_DATA",
        )

    mean_flux = sum(oot_flux) / n
    # Avoid division by zero for a flat or zero-mean series
    if mean_flux == 0.0:
        mean_flux = 1.0

    rms = math.sqrt(sum(((f / mean_flux - 1.0) * 1e6) ** 2 for f in oot_flux) / n)

    bin_factor = max(1, int(bin_minutes / cadence_minutes))
    noise_floor = rms / math.sqrt(bin_factor)

    flag = "HIGH_NOISE" if noise_floor > 500.0 else "OK"

    return NoiseFloorResult(
        oot_rms_ppm=rms,
        n_points=n,
        noise_floor_ppm=noise_floor,
        cadence_minutes=cadence_minutes,
        flag=flag,
    )


def format_noise_floor(result: NoiseFloorResult) -> str:
    lines = [
        "## Transit Noise Floor Estimate",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| OOT points | {result.n_points} |",
        f"| Cadence (min) | {result.cadence_minutes:.1f} |",
        f"| OOT RMS (ppm) | {result.oot_rms_ppm:.1f} |",
        f"| Noise floor (ppm) | {result.noise_floor_ppm:.1f} |",
        f"| Flag | {result.flag} |",
    ]
    return "\n".join(lines)


def _cli() -> int:
    parser = argparse.ArgumentParser(
        description="Estimate per-transit noise floor from OOT flux JSON file."
    )
    parser.add_argument("flux_file", help="JSON file containing list of flux values")
    parser.add_argument(
        "--cadence-minutes", type=float, default=2.0, help="Cadence in minutes (default 2.0)"
    )
    parser.add_argument(
        "--bin-minutes", type=float, default=30.0, help="Binning window in minutes (default 30.0)"
    )
    args = parser.parse_args()

    with open(args.flux_file) as fh:
        flux = json.load(fh)

    result = estimate_noise_floor(flux, args.cadence_minutes, args.bin_minutes)
    print(format_noise_floor(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
