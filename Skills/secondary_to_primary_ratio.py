"""Compute secondary/primary eclipse depth ratio and planet-model consistency.

Public API:
    SecondaryPrimaryResult  -- frozen dataclass
    compute_secondary_to_primary_ratio(primary_depth_ppm, secondary_depth_ppm,
                                        albedo_max) -> SecondaryPrimaryResult
    format_ratio_result(result) -> str
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class SecondaryPrimaryResult:
    primary_depth_ppm: float
    secondary_depth_ppm: float
    ratio: float
    planet_consistent: bool
    albedo_implied: float
    flag: str


def compute_secondary_to_primary_ratio(
    primary_depth_ppm: float,
    secondary_depth_ppm: float,
    albedo_max: float = 0.60,
) -> SecondaryPrimaryResult:
    if primary_depth_ppm <= 0:
        return SecondaryPrimaryResult(
            primary_depth_ppm=primary_depth_ppm, secondary_depth_ppm=secondary_depth_ppm,
            ratio=0.0, planet_consistent=False, albedo_implied=0.0, flag="INVALID_PRIMARY_DEPTH",
        )
    if secondary_depth_ppm < 0:
        return SecondaryPrimaryResult(
            primary_depth_ppm=primary_depth_ppm, secondary_depth_ppm=secondary_depth_ppm,
            ratio=0.0, planet_consistent=False, albedo_implied=0.0, flag="INVALID_SECONDARY_DEPTH",
        )
    ratio = secondary_depth_ppm / primary_depth_ppm
    # For a planet, secondary depth ~ Ag * (Rp/a)^2; primary depth ~ (Rp/R*)^2
    # ratio = secondary/primary = Ag * (R*/a)^2
    # For a typical hot Jupiter a/R* ~ 5-10: (R*/a)^2 ~ 0.01-0.04
    # Practical upper limit: ratio should be < albedo_max (any reasonable reflected light)
    # If ratio > albedo_max it implies thermal emission or EB
    planet_consistent = ratio <= albedo_max
    albedo_implied = ratio  # approximate: ratio ≈ Ag * (R*/a)^2 but we store ratio as proxy
    flag = "PLANET_CONSISTENT" if planet_consistent else "EB_SUSPECTED"
    return SecondaryPrimaryResult(
        primary_depth_ppm=primary_depth_ppm,
        secondary_depth_ppm=secondary_depth_ppm,
        ratio=ratio,
        planet_consistent=planet_consistent,
        albedo_implied=albedo_implied,
        flag=flag,
    )


def format_ratio_result(result: SecondaryPrimaryResult) -> str:
    lines = [
        "## Secondary/Primary Depth Ratio",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Primary Depth (ppm) | {result.primary_depth_ppm:.1f} |",
        f"| Secondary Depth (ppm) | {result.secondary_depth_ppm:.1f} |",
        f"| Ratio (sec/pri) | {result.ratio:.4f} |",
        f"| Planet Consistent | {result.planet_consistent} |",
        f"| Flag | {result.flag} |",
    ]
    return "\n".join(lines)


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Compute secondary/primary eclipse depth ratio.")
    parser.add_argument("primary_depth_ppm", type=float)
    parser.add_argument("secondary_depth_ppm", type=float)
    parser.add_argument("--albedo-max", type=float, default=0.60)
    args = parser.parse_args()
    result = compute_secondary_to_primary_ratio(
        args.primary_depth_ppm, args.secondary_depth_ppm, args.albedo_max,
    )
    print(format_ratio_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
