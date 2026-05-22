"""Compute flux ratios between a target star and nearby field stars.

Flux ratios are used to estimate dilution factors, contamination scores,
and background-EB false-positive probabilities.

Public API
----------
FluxRatioResult(target_mag, neighbour_mags, flux_ratios, total_contamination,
                dilution_factor, flag)
compute_flux_ratios(target_mag, neighbour_mags, *, zero_point) -> FluxRatioResult
format_flux_ratio_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FluxRatioResult:
    target_mag: float
    neighbour_mags: tuple[float, ...]
    flux_ratios: tuple[float, ...]          # F_i / F_target for each neighbour
    total_contamination: float              # sum(F_i) / F_target
    dilution_factor: float                  # F_target / (F_target + sum(F_i))
    flag: str  # "OK" | "NO_NEIGHBOURS" | "INVALID"


def compute_flux_ratios(
    target_mag: float,
    neighbour_mags: list[float],
    *,
    zero_point: float = 0.0,
) -> FluxRatioResult:
    """Compute flux ratios from magnitudes.

    Args:
        target_mag: Target star magnitude (TESS T-mag, Kepler Kp, etc.).
        neighbour_mags: List of neighbour star magnitudes.
        zero_point: Magnitude zero-point offset (default 0 = same band).

    Returns:
        :class:`FluxRatioResult`.
    """
    if not math.isfinite(target_mag):
        return FluxRatioResult(target_mag, (), (), 0.0, 1.0, "INVALID")

    if not neighbour_mags:
        return FluxRatioResult(target_mag, (), (), 0.0, 1.0, "NO_NEIGHBOURS")

    flux_target = 10.0 ** (-(target_mag - zero_point) / 2.5)
    ratios: list[float] = []
    for nm in neighbour_mags:
        if not math.isfinite(nm):
            ratios.append(0.0)
            continue
        flux_n = 10.0 ** (-(nm - zero_point) / 2.5)
        ratios.append(round(flux_n / flux_target, 6))

    total_contam = sum(ratios)
    dilution = 1.0 / (1.0 + total_contam) if total_contam >= 0 else 1.0

    return FluxRatioResult(
        target_mag=target_mag,
        neighbour_mags=tuple(neighbour_mags),
        flux_ratios=tuple(ratios),
        total_contamination=round(total_contam, 6),
        dilution_factor=round(dilution, 6),
        flag="OK",
    )


def format_flux_ratio_result(result: FluxRatioResult) -> str:
    """Format flux ratio result as Markdown."""
    lines = [
        "## Flux Ratio Calculator",
        "",
        f"- Target magnitude: {result.target_mag}",
        f"- Neighbours: {len(result.neighbour_mags)}",
        f"- Total contamination (ΣF_i/F_t): {result.total_contamination:.4f}",
        f"- Dilution factor: {result.dilution_factor:.4f}",
        f"- **Flag: {result.flag}**",
    ]
    if result.flux_ratios:
        lines.append("")
        lines.append("| Neighbour mag | Flux ratio |")
        lines.append("|---|---|")
        for mag, ratio in zip(result.neighbour_mags, result.flux_ratios, strict=False):
            lines.append(f"| {mag:.2f} | {ratio:.5f} |")
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="flux_ratio_calculator",
        description="Compute flux ratios from target and neighbour magnitudes.",
    )
    parser.add_argument("target_mag", type=float)
    parser.add_argument("neighbour_mags", nargs="*", type=float)
    args = parser.parse_args(argv)

    result = compute_flux_ratios(args.target_mag, args.neighbour_mags)
    print(format_flux_ratio_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
