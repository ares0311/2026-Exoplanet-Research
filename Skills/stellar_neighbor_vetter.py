"""Vet nearby stars for potential contamination of a transit signal.

Checks whether any catalogue neighbour is bright enough and close enough
to the target aperture to dilute or mimic the observed transit depth.

Public API
----------
NeighborCandidate(catalog_id, separation_arcsec, delta_mag, flux_fraction,
                  diluted_depth_ppm, contaminant_flag)
NeighborVettingResult(tic_id, target_depth_ppm, aperture_radius_arcsec,
                      n_neighbors, n_contaminants, max_dilution_fraction,
                      neighbors, flag)
vet_stellar_neighbors(tic_id, target_depth_ppm, neighbors, *,
                      aperture_radius_arcsec) -> NeighborVettingResult
format_neighbor_vetting(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NeighborCandidate:
    catalog_id: str
    separation_arcsec: float
    delta_mag: float              # mag difference (neighbor - target), positive = fainter
    flux_fraction: float          # neighbor flux / target flux
    diluted_depth_ppm: float      # if the transit were on the neighbor, diluted depth seen
    contaminant_flag: bool        # True if separation < aperture AND flux_fraction significant


@dataclass(frozen=True)
class NeighborVettingResult:
    tic_id: int | None
    target_depth_ppm: float
    aperture_radius_arcsec: float
    n_neighbors: int
    n_contaminants: int
    max_dilution_fraction: float
    neighbors: tuple[NeighborCandidate, ...]
    flag: str  # "CLEAN" | "CONTAMINATED" | "SEVERE_CONTAMINATION" | "NO_NEIGHBORS"


def vet_stellar_neighbors(
    tic_id: int | None,
    target_depth_ppm: float,
    neighbors: list[dict],
    *,
    aperture_radius_arcsec: float = 21.0,
    min_flux_fraction_concern: float = 0.01,
    severe_fraction: float = 0.10,
) -> NeighborVettingResult:
    """Vet neighboring stars for contamination.

    Args:
        tic_id: TIC ID of the primary target.
        target_depth_ppm: Observed transit depth in ppm.
        neighbors: List of dicts with keys: catalog_id, separation_arcsec, delta_mag.
        aperture_radius_arcsec: Photometric aperture radius in arcsec.
        min_flux_fraction_concern: Flux fraction threshold for flagging a contaminant.
        severe_fraction: Flag SEVERE if any neighbour contributes above this fraction.

    Returns:
        NeighborVettingResult with per-neighbor analysis.
    """
    processed: list[NeighborCandidate] = []
    for n in neighbors:
        raw_sep = n.get("separation_arcsec") if n.get("separation_arcsec") is not None \
            else n.get("separation")
        sep = float(raw_sep) if raw_sep is not None else 0.0
        raw_dmag = n.get("delta_mag") if n.get("delta_mag") is not None \
            else n.get("magnitude_difference")
        dmag = float(raw_dmag) if raw_dmag is not None else 99.0
        cat_id = str(n.get("catalog_id") or n.get("tic_id") or "")

        flux_fraction = 10 ** (-dmag / 2.5)
        # Diluted depth if transit were entirely on the neighbor
        diluted = target_depth_ppm * flux_fraction / (1.0 + flux_fraction)

        within_aperture = sep < aperture_radius_arcsec
        contaminant = within_aperture and flux_fraction >= min_flux_fraction_concern

        processed.append(NeighborCandidate(
            catalog_id=cat_id,
            separation_arcsec=round(sep, 2),
            delta_mag=round(dmag, 3),
            flux_fraction=round(flux_fraction, 6),
            diluted_depth_ppm=round(diluted, 2),
            contaminant_flag=contaminant,
        ))

    processed.sort(key=lambda x: x.separation_arcsec)
    n_contaminants = sum(1 for c in processed if c.contaminant_flag)
    max_dil = max((c.flux_fraction for c in processed if c.contaminant_flag), default=0.0)

    if not processed:
        flag = "NO_NEIGHBORS"
    elif n_contaminants == 0:
        flag = "CLEAN"
    elif max_dil >= severe_fraction:
        flag = "SEVERE_CONTAMINATION"
    else:
        flag = "CONTAMINATED"

    return NeighborVettingResult(
        tic_id=tic_id,
        target_depth_ppm=target_depth_ppm,
        aperture_radius_arcsec=aperture_radius_arcsec,
        n_neighbors=len(processed),
        n_contaminants=n_contaminants,
        max_dilution_fraction=round(max_dil, 6),
        neighbors=tuple(processed),
        flag=flag,
    )


def format_neighbor_vetting(result: NeighborVettingResult) -> str:
    """Format neighbor vetting result as Markdown.

    Args:
        result: NeighborVettingResult to format.

    Returns:
        Markdown string.
    """
    tic_str = str(result.tic_id) if result.tic_id is not None else "Unknown"
    lines = [
        f"## Stellar Neighbor Vetting — TIC {tic_str}\n",
        f"**Status**: `{result.flag}` | "
        f"Neighbors: {result.n_neighbors} | Contaminants: {result.n_contaminants} | "
        f"Max dilution: {result.max_dilution_fraction:.4f}\n",
    ]
    if not result.neighbors:
        lines.append("\n_No neighbors in catalog._")
        return "\n".join(lines)

    lines += [
        "",
        "| ID | Sep (arcsec) | Δmag | Flux fraction | Diluted depth (ppm) | Contaminant? |",
        "|---|---|---|---|---|---|",
    ]
    for n in result.neighbors:
        cont_str = "⚠ YES" if n.contaminant_flag else "no"
        lines.append(
            f"| {n.catalog_id} | {n.separation_arcsec:.1f} | {n.delta_mag:.2f} | "
            f"{n.flux_fraction:.4f} | {n.diluted_depth_ppm:.1f} | {cont_str} |"
        )
    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Vet stellar neighbors for contamination.")
    parser.add_argument("--tic-id", type=int, default=None)
    parser.add_argument("--depth-ppm", type=float, required=True)
    parser.add_argument("--neighbors", required=True, help="JSON file of neighbor dicts.")
    parser.add_argument("--aperture", type=float, default=21.0)
    args = parser.parse_args(argv)

    from pathlib import Path
    neighbors = json.loads(Path(args.neighbors).read_text())
    result = vet_stellar_neighbors(
        args.tic_id, args.depth_ppm, neighbors,
        aperture_radius_arcsec=args.aperture,
    )
    print(format_neighbor_vetting(result))
    return 0 if result.flag in ("CLEAN", "NO_NEIGHBORS") else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
