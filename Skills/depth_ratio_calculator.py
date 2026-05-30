"""
Computes the secondary-to-primary depth ratio and classifies the signal.

Public API:
    DepthRatioResult      -- frozen dataclass holding depth ratio classification
    compute_depth_ratio(primary_depth_ppm, secondary_depth_ppm) -> DepthRatioResult
    format_depth_ratio(result) -> str
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class DepthRatioResult:
    primary_depth_ppm: float
    secondary_depth_ppm: float
    depth_ratio: float
    classification: str
    flag: str


def compute_depth_ratio(
    primary_depth_ppm: float,
    secondary_depth_ppm: float,
) -> DepthRatioResult:
    if primary_depth_ppm <= 0.0:
        return DepthRatioResult(
            primary_depth_ppm=primary_depth_ppm,
            secondary_depth_ppm=secondary_depth_ppm,
            depth_ratio=0.0,
            classification="UNDEFINED",
            flag="INVALID_PRIMARY_DEPTH",
        )

    ratio = secondary_depth_ppm / primary_depth_ppm

    if ratio < 0.1:
        classification = "PLANET_LIKE"
    elif ratio < 0.5:
        classification = "EB_LIKELY"
    else:
        classification = "SYMMETRIC_EB"

    flag = "EB_FLAG" if ratio >= 0.1 else "OK"

    return DepthRatioResult(
        primary_depth_ppm=primary_depth_ppm,
        secondary_depth_ppm=secondary_depth_ppm,
        depth_ratio=ratio,
        classification=classification,
        flag=flag,
    )


def format_depth_ratio(result: DepthRatioResult) -> str:
    lines = [
        "## Depth Ratio Calculator",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Primary depth (ppm) | {result.primary_depth_ppm:.1f} |",
        f"| Secondary depth (ppm) | {result.secondary_depth_ppm:.1f} |",
        f"| Depth ratio (sec/pri) | {result.depth_ratio:.4f} |",
        f"| Classification | {result.classification} |",
        f"| Flag | {result.flag} |",
    ]
    return "\n".join(lines)


def _cli() -> int:
    parser = argparse.ArgumentParser(
        description="Compute secondary-to-primary depth ratio and classify signal."
    )
    parser.add_argument("primary_depth_ppm", type=float, help="Primary transit depth in ppm")
    parser.add_argument("secondary_depth_ppm", type=float, help="Secondary eclipse depth in ppm")
    args = parser.parse_args()

    result = compute_depth_ratio(args.primary_depth_ppm, args.secondary_depth_ppm)
    print(format_depth_ratio(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
