"""Estimate aperture crowding from nearby star count and magnitude sum.

Computes flux ratio and crowding metric from neighbor magnitudes within a
given aperture radius, flagging heavily contaminated targets.

Public API
----------
CrowdingResult
estimate_crowding(target_mag, neighbor_mags, neighbor_separations_arcsec, ...) -> CrowdingResult
format_crowding(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CrowdingResult:
    target_mag: float
    n_neighbors: int
    neighbor_mags: tuple[float, ...]
    neighbor_separations_arcsec: tuple[float, ...]
    flux_ratio: float
    crowding_metric: float
    contamination_fraction: float
    aperture_radius_arcsec: float
    flag: str  # "OK" | "CROWDED" | "INVALID"


def estimate_crowding(
    target_mag: float,
    neighbor_mags: list[float],
    neighbor_separations_arcsec: list[float],
    *,
    aperture_radius_arcsec: float = 21.0,
    crowded_threshold: float = 0.10,
) -> CrowdingResult:
    """Estimate aperture crowding from nearby star magnitudes.

    flux_ratio = sum(10^(-0.4*(m_i - target_mag))) for neighbors within aperture.
    crowding_metric = 1 / (1 + flux_ratio)  — 1.0 means no contamination.
    contamination_fraction = flux_ratio / (1 + flux_ratio)

    Args:
        target_mag: Target star magnitude.
        neighbor_mags: List of neighbor magnitudes.
        neighbor_separations_arcsec: Separation of each neighbor in arcsec.
        aperture_radius_arcsec: Aperture radius; neighbors beyond this are excluded.
        crowded_threshold: contamination_fraction > this → CROWDED.

    Returns:
        :class:`CrowdingResult`.
    """
    if not math.isfinite(target_mag):
        return CrowdingResult(
            target_mag=target_mag,
            n_neighbors=0,
            neighbor_mags=(),
            neighbor_separations_arcsec=(),
            flux_ratio=0.0,
            crowding_metric=1.0,
            contamination_fraction=0.0,
            aperture_radius_arcsec=aperture_radius_arcsec,
            flag="INVALID",
        )

    if len(neighbor_mags) != len(neighbor_separations_arcsec):
        return CrowdingResult(
            target_mag=target_mag,
            n_neighbors=0,
            neighbor_mags=tuple(neighbor_mags),
            neighbor_separations_arcsec=tuple(neighbor_separations_arcsec),
            flux_ratio=0.0,
            crowding_metric=1.0,
            contamination_fraction=0.0,
            aperture_radius_arcsec=aperture_radius_arcsec,
            flag="INVALID",
        )

    # Filter neighbors within aperture
    inside_mags: list[float] = []
    inside_seps: list[float] = []
    for mag, sep in zip(neighbor_mags, neighbor_separations_arcsec, strict=False):
        if math.isfinite(sep) and sep <= aperture_radius_arcsec and math.isfinite(mag):
            inside_mags.append(mag)
            inside_seps.append(sep)

    # Compute flux ratio
    flux_ratio = sum(10.0 ** (-0.4 * (m - target_mag)) for m in inside_mags)

    crowding_metric = 1.0 / (1.0 + flux_ratio) if (1.0 + flux_ratio) > 0 else 0.0
    contamination_fraction = flux_ratio / (1.0 + flux_ratio) if (1.0 + flux_ratio) > 0 else 0.0

    flag = "CROWDED" if contamination_fraction > crowded_threshold else "OK"

    return CrowdingResult(
        target_mag=target_mag,
        n_neighbors=len(inside_mags),
        neighbor_mags=tuple(round(m, 4) for m in inside_mags),
        neighbor_separations_arcsec=tuple(round(s, 4) for s in inside_seps),
        flux_ratio=round(flux_ratio, 6),
        crowding_metric=round(crowding_metric, 6),
        contamination_fraction=round(contamination_fraction, 6),
        aperture_radius_arcsec=aperture_radius_arcsec,
        flag=flag,
    )


def format_crowding(result: CrowdingResult) -> str:
    """Format crowding estimation result as Markdown."""
    lines = [
        "## Target Crowding Estimator",
        "",
        f"- Target magnitude: {result.target_mag}",
        f"- Aperture radius: {result.aperture_radius_arcsec} arcsec",
        f"- Neighbors within aperture: {result.n_neighbors}",
        f"- **Flux ratio: {result.flux_ratio:.6f}**",
        f"- **Crowding metric: {result.crowding_metric:.6f}**",
        f"- **Contamination fraction: {result.contamination_fraction:.6f}**",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="target_crowding_estimator",
        description="Estimate aperture crowding from nearby star magnitudes.",
    )
    parser.add_argument("target_mag", type=float)
    parser.add_argument("neighbor_mags_json", help="JSON array of neighbor magnitudes")
    parser.add_argument("neighbor_separations_json", help="JSON array of separations (arcsec)")
    parser.add_argument("--aperture-radius-arcsec", type=float, default=21.0)
    parser.add_argument("--crowded-threshold", type=float, default=0.10)
    args = parser.parse_args(argv)

    neighbor_mags = json.loads(args.neighbor_mags_json)
    neighbor_seps = json.loads(args.neighbor_separations_json)

    result = estimate_crowding(
        args.target_mag,
        neighbor_mags,
        neighbor_seps,
        aperture_radius_arcsec=args.aperture_radius_arcsec,
        crowded_threshold=args.crowded_threshold,
    )
    print(format_crowding(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
