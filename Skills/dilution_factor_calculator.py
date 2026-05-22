"""Compute the flux dilution factor from nearby contaminating stars.

When a target star is blended with one or more neighbours inside the
photometric aperture, the observed transit depth is shallower than the true
depth by the dilution factor D = F_target / F_total.  This module computes D
from a list of neighbour flux ratios and also derives the corrected depth.

Public API
----------
DilutionResult(target_flux_frac, dilution_factor, corrected_depth_ppm,
               n_contaminants, flag)
compute_dilution_factor(neighbour_flux_ratios, observed_depth_ppm, *,
                        target_flux) -> DilutionResult
format_dilution_result(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DilutionResult:
    target_flux_frac: float        # F_target / F_total in [0, 1]
    dilution_factor: float         # same as target_flux_frac
    corrected_depth_ppm: float | None  # observed_depth / dilution_factor
    n_contaminants: int
    flag: str  # "OK" | "NO_CONTAMINANTS" | "INVALID"


def compute_dilution_factor(
    neighbour_flux_ratios: list[float],
    observed_depth_ppm: float | None = None,
    *,
    target_flux: float = 1.0,
) -> DilutionResult:
    """Compute dilution factor from neighbour flux ratios.

    Args:
        neighbour_flux_ratios: List of flux ratios F_neighbour / F_target for
            each contaminating source inside the aperture.  Values must be
            non-negative.
        observed_depth_ppm: Observed transit depth in ppm (optional).  If
            provided, returns the dilution-corrected true depth.
        target_flux: Flux of the target star in arbitrary units (default 1.0).
            Neighbours are expressed as ratios relative to this value.

    Returns:
        :class:`DilutionResult`.
    """
    if target_flux <= 0:
        return DilutionResult(0.0, 0.0, None, 0, "INVALID")

    if any(r < 0 for r in neighbour_flux_ratios):
        return DilutionResult(0.0, 0.0, None, 0, "INVALID")

    if not neighbour_flux_ratios:
        return DilutionResult(
            target_flux_frac=1.0,
            dilution_factor=1.0,
            corrected_depth_ppm=observed_depth_ppm,
            n_contaminants=0,
            flag="NO_CONTAMINANTS",
        )

    total_contaminant_flux = target_flux * sum(neighbour_flux_ratios)
    f_total = target_flux + total_contaminant_flux
    d = target_flux / f_total  # dilution factor

    corrected: float | None = None
    if observed_depth_ppm is not None and d > 1e-12:
        corrected = round(observed_depth_ppm / d, 2)

    return DilutionResult(
        target_flux_frac=round(d, 6),
        dilution_factor=round(d, 6),
        corrected_depth_ppm=corrected,
        n_contaminants=len(neighbour_flux_ratios),
        flag="OK",
    )


def format_dilution_result(result: DilutionResult) -> str:
    """Format dilution result as Markdown."""
    lines = [
        "## Flux Dilution Factor",
        "",
        f"- Contaminants: {result.n_contaminants}",
        f"- Target flux fraction: {result.target_flux_frac:.4f}",
        f"- Dilution factor: {result.dilution_factor:.4f}",
        "- Observed depth (ppm): —",
        f"- Corrected depth (ppm): {result.corrected_depth_ppm}",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="dilution_factor_calculator",
        description="Compute flux dilution factor from neighbour flux ratios.",
    )
    parser.add_argument("ratios", nargs="*", type=float,
                        help="Flux ratios F_neighbour / F_target")
    parser.add_argument("--depth-ppm", type=float, default=None)
    args = parser.parse_args(argv)

    result = compute_dilution_factor(args.ratios, args.depth_ppm)
    print(format_dilution_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
