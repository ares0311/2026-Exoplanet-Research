"""Estimate TESS pixel contamination from CROWDSAP and neighbour fluxes."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PixelContaminationResult:
    crowdsap: float
    neighbour_dilution: float
    corrected_depth_factor: float
    contamination_fraction: float
    flag: str


def estimate_pixel_contamination(
    crowdsap: float,
    depth_ppm: float = 1000.0,
    neighbour_flux_ratio: float = 0.0,
) -> PixelContaminationResult:
    """
    Estimate TESS pixel-level contamination and corrected transit depth.

    CROWDSAP = fraction of flux from the target star in the aperture.
    contamination_fraction = 1 - CROWDSAP.

    Additional neighbour_flux_ratio (from catalog neighbours) is added
    to the contamination estimate.

    corrected_depth_factor = 1 / (CROWDSAP - neighbour_dilution)
    True depth = measured_depth * corrected_depth_factor.
    """
    if not math.isfinite(crowdsap) or not (0.0 < crowdsap <= 1.0):
        return PixelContaminationResult(
            crowdsap=crowdsap,
            neighbour_dilution=neighbour_flux_ratio,
            corrected_depth_factor=float("nan"),
            contamination_fraction=float("nan"),
            flag="INVALID_CROWDSAP",
        )
    if not math.isfinite(neighbour_flux_ratio) or neighbour_flux_ratio < 0.0:
        return PixelContaminationResult(
            crowdsap=crowdsap,
            neighbour_dilution=neighbour_flux_ratio,
            corrected_depth_factor=float("nan"),
            contamination_fraction=float("nan"),
            flag="INVALID_NEIGHBOUR_FLUX",
        )
    if not math.isfinite(depth_ppm) or depth_ppm < 0.0:
        return PixelContaminationResult(
            crowdsap=crowdsap,
            neighbour_dilution=neighbour_flux_ratio,
            corrected_depth_factor=float("nan"),
            contamination_fraction=float("nan"),
            flag="INVALID_DEPTH",
        )

    contamination_frac = 1.0 - crowdsap + neighbour_flux_ratio
    contamination_frac = min(contamination_frac, 0.9999)

    effective_target_frac = crowdsap - neighbour_flux_ratio
    if effective_target_frac <= 0.0:
        effective_target_frac = crowdsap

    depth_factor = 1.0 / effective_target_frac

    flag = "HIGH_CONTAMINATION" if contamination_frac > 0.5 else "OK"

    return PixelContaminationResult(
        crowdsap=crowdsap,
        neighbour_dilution=round(neighbour_flux_ratio, 5),
        corrected_depth_factor=round(depth_factor, 4),
        contamination_fraction=round(contamination_frac, 4),
        flag=flag,
    )


def format_pixel_contamination(r: PixelContaminationResult) -> str:
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| CROWDSAP | {r.crowdsap:.4f} |\n"
        f"| Neighbour dilution | {r.neighbour_dilution:.5f} |\n"
        f"| Contamination fraction | {r.contamination_fraction:.4f} |\n"
        f"| Depth correction factor | {r.corrected_depth_factor:.4f} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Estimate TESS pixel contamination.")
    p.add_argument("crowdsap", type=float)
    p.add_argument("--depth-ppm", type=float, default=1000.0)
    p.add_argument("--neighbour-flux-ratio", type=float, default=0.0)
    args = p.parse_args()
    r = estimate_pixel_contamination(
        args.crowdsap, args.depth_ppm, args.neighbour_flux_ratio
    )
    print(format_pixel_contamination(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
