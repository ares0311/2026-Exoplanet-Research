"""Correct observed transit depth for quadratic limb darkening.

Public API:
    LimbDarkeningDepthResult  -- frozen dataclass
    correct_depth_for_ld(observed_depth_ppm, u1, u2, impact_parameter) -> LimbDarkeningDepthResult
    format_ld_depth_result(result) -> str
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class LimbDarkeningDepthResult:
    observed_depth_ppm: float
    corrected_depth_ppm: float
    ld_correction_factor: float
    u1: float
    u2: float
    flag: str


def correct_depth_for_ld(
    observed_depth_ppm: float,
    u1: float,
    u2: float,
    impact_parameter: float = 0.0,
) -> LimbDarkeningDepthResult:
    if observed_depth_ppm <= 0:
        return LimbDarkeningDepthResult(
            observed_depth_ppm=observed_depth_ppm, corrected_depth_ppm=0.0,
            ld_correction_factor=1.0, u1=u1, u2=u2, flag="INVALID_DEPTH",
        )
    if not (0.0 <= u1 <= 1.0 and 0.0 <= u2 <= 1.0 and u1 + u2 <= 1.0):
        return LimbDarkeningDepthResult(
            observed_depth_ppm=observed_depth_ppm, corrected_depth_ppm=0.0,
            ld_correction_factor=1.0, u1=u1, u2=u2, flag="INVALID_LD_COEFFS",
        )
    if not 0.0 <= impact_parameter < 1.0:
        return LimbDarkeningDepthResult(
            observed_depth_ppm=observed_depth_ppm, corrected_depth_ppm=0.0,
            ld_correction_factor=1.0, u1=u1, u2=u2, flag="INVALID_IMPACT_PARAMETER",
        )
    mu = (1.0 - impact_parameter ** 2) ** 0.5
    i_mu = 1.0 - u1 * (1.0 - mu) - u2 * (1.0 - mu) ** 2
    i_center = 1.0
    ld_factor = i_mu / i_center if i_center > 0 else 1.0
    corrected = observed_depth_ppm / ld_factor if ld_factor > 0 else observed_depth_ppm
    return LimbDarkeningDepthResult(
        observed_depth_ppm=observed_depth_ppm,
        corrected_depth_ppm=corrected,
        ld_correction_factor=ld_factor,
        u1=u1,
        u2=u2,
        flag="OK",
    )


def format_ld_depth_result(result: LimbDarkeningDepthResult) -> str:
    lines = [
        "## Limb Darkening Depth Correction",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Observed Depth (ppm) | {result.observed_depth_ppm:.1f} |",
        f"| Corrected Depth (ppm) | {result.corrected_depth_ppm:.1f} |",
        f"| LD Correction Factor | {result.ld_correction_factor:.4f} |",
        f"| u1 | {result.u1:.4f} |",
        f"| u2 | {result.u2:.4f} |",
        f"| Flag | {result.flag} |",
    ]
    return "\n".join(lines)


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Correct transit depth for limb darkening.")
    parser.add_argument("observed_depth_ppm", type=float)
    parser.add_argument("u1", type=float)
    parser.add_argument("u2", type=float)
    parser.add_argument("--impact-parameter", type=float, default=0.0)
    args = parser.parse_args()
    result = correct_depth_for_ld(
        args.observed_depth_ppm, args.u1, args.u2,
        impact_parameter=args.impact_parameter,
    )
    print(format_ld_depth_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
