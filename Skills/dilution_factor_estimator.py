from __future__ import annotations

import argparse
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class DilutionResult:
    dilution_factor: float
    corrected_depth_ppm: float
    n_neighbours: int
    flag: str


def estimate_dilution_factor(
    target_flux: float,
    neighbour_fluxes: list[float],
) -> DilutionResult:
    """Estimate the dilution factor from target and neighbour flux contributions.

    dilution_factor = target_flux / (target_flux + sum(neighbour_fluxes))
    """
    if target_flux <= 0:
        return DilutionResult(
            dilution_factor=0.0,
            corrected_depth_ppm=0.0,
            n_neighbours=len(neighbour_fluxes),
            flag="INVALID_TARGET_FLUX",
        )

    total = target_flux + sum(neighbour_fluxes)
    dilution_factor = target_flux / total if total > 0 else 1.0

    return DilutionResult(
        dilution_factor=dilution_factor,
        corrected_depth_ppm=0.0,
        n_neighbours=len(neighbour_fluxes),
        flag="OK",
    )


def correct_diluted_depth(
    observed_depth_ppm: float,
    target_flux: float,
    neighbour_fluxes: list[float],
) -> DilutionResult:
    """Correct an observed transit depth for flux dilution from neighbours."""
    if observed_depth_ppm <= 0:
        return DilutionResult(
            dilution_factor=0.0,
            corrected_depth_ppm=0.0,
            n_neighbours=len(neighbour_fluxes),
            flag="INVALID_DEPTH",
        )

    if target_flux <= 0:
        return DilutionResult(
            dilution_factor=0.0,
            corrected_depth_ppm=0.0,
            n_neighbours=len(neighbour_fluxes),
            flag="INVALID_TARGET_FLUX",
        )

    total = target_flux + sum(neighbour_fluxes)
    dilution_factor = target_flux / total if total > 0 else 1.0
    corrected_depth_ppm = observed_depth_ppm / dilution_factor if dilution_factor > 0 else 0.0

    return DilutionResult(
        dilution_factor=dilution_factor,
        corrected_depth_ppm=corrected_depth_ppm,
        n_neighbours=len(neighbour_fluxes),
        flag="OK",
    )


def format_dilution_result(result: DilutionResult) -> str:
    """Return a Markdown table summarising the dilution result."""
    lines = [
        "| Field | Value |",
        "| --- | --- |",
        f"| Dilution Factor | {result.dilution_factor:.6f} |",
        f"| Corrected Depth (ppm) | {result.corrected_depth_ppm:.2f} |",
        f"| N Neighbours | {result.n_neighbours} |",
        f"| Flag | {result.flag} |",
    ]
    return "\n".join(lines)


def _cli() -> int:
    parser = argparse.ArgumentParser(
        description="Estimate flux dilution factor from target and neighbour fluxes."
    )
    parser.add_argument("target_flux", type=float, help="Target star flux.")
    parser.add_argument(
        "--neighbours",
        nargs="*",
        type=float,
        default=[],
        help="Neighbour star flux values.",
    )
    parser.add_argument(
        "--observed-depth",
        type=float,
        default=None,
        help="Observed transit depth in ppm (enables depth correction).",
    )
    parser.add_argument("--json", action="store_true", help="Output raw JSON.")
    args = parser.parse_args()

    if args.observed_depth is not None:
        result = correct_diluted_depth(
            args.observed_depth, args.target_flux, args.neighbours
        )
    else:
        result = estimate_dilution_factor(args.target_flux, args.neighbours)

    if args.json:
        import dataclasses
        print(json.dumps(dataclasses.asdict(result), indent=2))
    else:
        print(format_dilution_result(result))

    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
