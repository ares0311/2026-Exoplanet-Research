"""Compute total photometric noise budget for a TESS target.

Public API:
    NoiseBudgetResult  -- frozen dataclass
    compute_noise_budget(target_counts, sky_counts, n_pixels, read_noise_e,
                         n_cadences, systematic_ppm) -> NoiseBudgetResult
    format_noise_budget(result) -> str
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class NoiseBudgetResult:
    photon_noise_ppm: float
    sky_noise_ppm: float
    read_noise_ppm: float
    systematic_ppm: float
    total_noise_ppm: float
    flag: str


def compute_noise_budget(
    target_counts: float,
    sky_counts_per_pixel: float,
    n_pixels: int,
    read_noise_electrons: float,
    *,
    n_cadences: int = 1,
    systematic_ppm: float = 0.0,
) -> NoiseBudgetResult:
    if target_counts <= 0:
        return NoiseBudgetResult(
            photon_noise_ppm=0.0, sky_noise_ppm=0.0, read_noise_ppm=0.0,
            systematic_ppm=0.0, total_noise_ppm=0.0, flag="INVALID_TARGET_COUNTS",
        )
    if n_pixels <= 0:
        return NoiseBudgetResult(
            photon_noise_ppm=0.0, sky_noise_ppm=0.0, read_noise_ppm=0.0,
            systematic_ppm=0.0, total_noise_ppm=0.0, flag="INVALID_N_PIXELS",
        )
    if n_cadences <= 0:
        return NoiseBudgetResult(
            photon_noise_ppm=0.0, sky_noise_ppm=0.0, read_noise_ppm=0.0,
            systematic_ppm=0.0, total_noise_ppm=0.0, flag="INVALID_N_CADENCES",
        )
    photon_noise_ppm = 1e6 / math.sqrt(target_counts * n_cadences)
    sky_noise_ppm = (
        1e6 * math.sqrt(sky_counts_per_pixel * n_pixels * n_cadences)
        / (target_counts * n_cadences)
        if sky_counts_per_pixel >= 0 else 0.0
    )
    read_noise_ppm = (
        1e6 * math.sqrt(read_noise_electrons ** 2 * n_pixels * n_cadences)
        / (target_counts * n_cadences)
        if read_noise_electrons >= 0 else 0.0
    )
    total_noise_ppm = math.sqrt(
        photon_noise_ppm ** 2
        + sky_noise_ppm ** 2
        + read_noise_ppm ** 2
        + systematic_ppm ** 2
    )
    return NoiseBudgetResult(
        photon_noise_ppm=photon_noise_ppm,
        sky_noise_ppm=sky_noise_ppm,
        read_noise_ppm=read_noise_ppm,
        systematic_ppm=systematic_ppm,
        total_noise_ppm=total_noise_ppm,
        flag="OK",
    )


def format_noise_budget(result: NoiseBudgetResult) -> str:
    lines = [
        "## Stellar Noise Budget",
        "",
        "| Component | Value (ppm) |",
        "|-----------|-------------|",
        f"| Photon Noise | {result.photon_noise_ppm:.2f} |",
        f"| Sky Noise | {result.sky_noise_ppm:.2f} |",
        f"| Read Noise | {result.read_noise_ppm:.2f} |",
        f"| Systematic | {result.systematic_ppm:.2f} |",
        f"| **Total** | **{result.total_noise_ppm:.2f}** |",
        f"| Flag | {result.flag} |",
    ]
    return "\n".join(lines)


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Compute photometric noise budget.")
    parser.add_argument("target_counts", type=float)
    parser.add_argument("sky_counts_per_pixel", type=float)
    parser.add_argument("n_pixels", type=int)
    parser.add_argument("read_noise_electrons", type=float)
    parser.add_argument("--n-cadences", type=int, default=1)
    parser.add_argument("--systematic-ppm", type=float, default=0.0)
    args = parser.parse_args()
    result = compute_noise_budget(
        args.target_counts, args.sky_counts_per_pixel, args.n_pixels,
        args.read_noise_electrons, n_cadences=args.n_cadences,
        systematic_ppm=args.systematic_ppm,
    )
    print(format_noise_budget(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
