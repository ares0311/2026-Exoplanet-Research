"""Estimate sky background flux contribution per pixel.

Public API:
    SkyBackgroundResult  -- frozen dataclass
    estimate_sky_background(sky_counts_per_pixel, n_pixels, target_counts) -> SkyBackgroundResult
    format_sky_background(result) -> str
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class SkyBackgroundResult:
    sky_counts_per_pixel: float
    n_pixels: int
    total_sky_counts: float
    target_counts: float
    sky_fraction: float
    flag: str


def estimate_sky_background(
    sky_counts_per_pixel: float,
    n_pixels: int,
    target_counts: float,
) -> SkyBackgroundResult:
    if sky_counts_per_pixel < 0:
        return SkyBackgroundResult(
            sky_counts_per_pixel=sky_counts_per_pixel,
            n_pixels=n_pixels,
            total_sky_counts=0.0,
            target_counts=target_counts,
            sky_fraction=0.0,
            flag="INVALID_SKY_COUNTS",
        )
    if n_pixels <= 0:
        return SkyBackgroundResult(
            sky_counts_per_pixel=sky_counts_per_pixel,
            n_pixels=n_pixels,
            total_sky_counts=0.0,
            target_counts=target_counts,
            sky_fraction=0.0,
            flag="INVALID_N_PIXELS",
        )
    if target_counts <= 0:
        return SkyBackgroundResult(
            sky_counts_per_pixel=sky_counts_per_pixel,
            n_pixels=n_pixels,
            total_sky_counts=0.0,
            target_counts=target_counts,
            sky_fraction=0.0,
            flag="INVALID_TARGET_COUNTS",
        )
    total_sky = sky_counts_per_pixel * n_pixels
    total = target_counts + total_sky
    sky_fraction = total_sky / total if total > 0 else 0.0
    return SkyBackgroundResult(
        sky_counts_per_pixel=sky_counts_per_pixel,
        n_pixels=n_pixels,
        total_sky_counts=total_sky,
        target_counts=target_counts,
        sky_fraction=sky_fraction,
        flag="OK",
    )


def format_sky_background(result: SkyBackgroundResult) -> str:
    lines = [
        "## Sky Background Estimate",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Sky Counts/Pixel | {result.sky_counts_per_pixel:.2f} |",
        f"| N Pixels | {result.n_pixels} |",
        f"| Total Sky Counts | {result.total_sky_counts:.2f} |",
        f"| Target Counts | {result.target_counts:.2f} |",
        f"| Sky Fraction | {result.sky_fraction:.4f} |",
        f"| Flag | {result.flag} |",
    ]
    return "\n".join(lines)


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Estimate sky background contribution.")
    parser.add_argument("sky_counts_per_pixel", type=float)
    parser.add_argument("n_pixels", type=int)
    parser.add_argument("target_counts", type=float)
    args = parser.parse_args()
    result = estimate_sky_background(
        args.sky_counts_per_pixel, args.n_pixels, args.target_counts
    )
    print(format_sky_background(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
