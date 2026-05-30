"""Simulate aperture photometry SNR for a target star.

Public API:
    ApertureSimResult  -- frozen dataclass
    simulate_aperture_photometry(target_counts, sky_counts_per_pixel, aperture_radius_pixels,
                                  read_noise_electrons, n_exposures) -> ApertureSimResult
    format_aperture_sim_result(result) -> str
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ApertureSimResult:
    target_counts: float
    sky_total_counts: float
    read_noise_total: float
    noise_total: float
    snr: float
    aperture_pixels: float
    flag: str


def simulate_aperture_photometry(
    target_counts: float,
    sky_counts_per_pixel: float,
    aperture_radius_pixels: float,
    read_noise_electrons: float = 10.0,
    n_exposures: int = 1,
) -> ApertureSimResult:
    if target_counts <= 0:
        return ApertureSimResult(
            target_counts=target_counts, sky_total_counts=0.0, read_noise_total=0.0,
            noise_total=0.0, snr=0.0, aperture_pixels=0.0, flag="INVALID_TARGET_COUNTS",
        )
    if aperture_radius_pixels <= 0:
        return ApertureSimResult(
            target_counts=target_counts, sky_total_counts=0.0, read_noise_total=0.0,
            noise_total=0.0, snr=0.0, aperture_pixels=0.0, flag="INVALID_APERTURE",
        )
    if n_exposures < 1:
        return ApertureSimResult(
            target_counts=target_counts, sky_total_counts=0.0, read_noise_total=0.0,
            noise_total=0.0, snr=0.0, aperture_pixels=0.0, flag="INVALID_N_EXPOSURES",
        )
    n_pix = math.pi * aperture_radius_pixels ** 2
    sky_total = sky_counts_per_pixel * n_pix * n_exposures
    signal = target_counts * n_exposures
    read_total = read_noise_electrons ** 2 * n_pix * n_exposures
    noise = math.sqrt(signal + sky_total + read_total)
    snr = signal / noise if noise > 0 else 0.0
    return ApertureSimResult(
        target_counts=target_counts,
        sky_total_counts=sky_total,
        read_noise_total=math.sqrt(read_total),
        noise_total=noise,
        snr=snr,
        aperture_pixels=n_pix,
        flag="OK",
    )


def format_aperture_sim_result(result: ApertureSimResult) -> str:
    lines = [
        "## Aperture Photometry Simulation",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Target Counts | {result.target_counts:.1f} |",
        f"| Sky Total Counts | {result.sky_total_counts:.1f} |",
        f"| Read Noise Total | {result.read_noise_total:.2f} |",
        f"| Total Noise | {result.noise_total:.2f} |",
        f"| SNR | {result.snr:.2f} |",
        f"| Aperture Pixels | {result.aperture_pixels:.2f} |",
        f"| Flag | {result.flag} |",
    ]
    return "\n".join(lines)


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Simulate aperture photometry SNR.")
    parser.add_argument("target_counts", type=float)
    parser.add_argument("sky_counts_per_pixel", type=float)
    parser.add_argument("aperture_radius_pixels", type=float)
    parser.add_argument("--read-noise", type=float, default=10.0)
    parser.add_argument("--n-exposures", type=int, default=1)
    args = parser.parse_args()
    result = simulate_aperture_photometry(
        args.target_counts, args.sky_counts_per_pixel, args.aperture_radius_pixels,
        args.read_noise, args.n_exposures,
    )
    print(format_aperture_sim_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
