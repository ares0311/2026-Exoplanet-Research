"""Compute the aperture crowding metric from nearby-star magnitudes.

Derives the fraction of aperture flux contributed by the target star versus
all sources (the TESS ``CROWDSAP`` equivalent, but computed analytically from
catalog positions and magnitudes).  Distinct from ``flux_contamination_corrector``
(applies the correction) and ``dilution_factor_calculator`` (uses pre-computed ratios).

Public API
----------
CrowdingResult(target_flux, total_flux, crowding_metric, contamination_ratio,
               dilution_ppm_per_neighbor, n_neighbors, flag)
compute_crowding_metric(target_mag, neighbor_mags, neighbor_separations_arcsec,
                        *, aperture_radius_arcsec,
                        psf_fwhm_arcsec) -> CrowdingResult
format_crowding_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CrowdingResult:
    target_flux: float               # relative flux of target (= 1.0 reference)
    total_flux: float                # total flux inside aperture (target + neighbors)
    crowding_metric: float | None    # target_flux / total_flux ∈ (0, 1]
    contamination_ratio: float | None  # 1 - crowding_metric
    dilution_ppm_per_neighbor: float | None  # average dilution per neighbor (ppm)
    n_neighbors: int
    flag: str  # "OK" | "INVALID"


def _flux_in_aperture(separation_arcsec: float, aperture_radius_arcsec: float,
                      psf_fwhm_arcsec: float) -> float:
    """Fraction of a Gaussian PSF falling inside a circular aperture.

    Uses the analytic approximation: for a source at ``separation``, the
    fraction inside radius ``r`` of a Gaussian PSF with sigma = FWHM/2.355
    is approximated as the integral of the 2-D Gaussian from 0 to the
    effective radial distance. Here we use a simple enclosed-energy formula.
    """
    sigma = psf_fwhm_arcsec / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    if sigma <= 0:
        return 1.0 if separation_arcsec < aperture_radius_arcsec else 0.0
    # Fraction of a 2-D Gaussian inside a circle of radius r at origin
    # centred on (separation, 0): approximation via mean radial distance
    effective_r = max(0.0, aperture_radius_arcsec - separation_arcsec)
    if effective_r <= 0:
        return 0.0
    frac = 1.0 - math.exp(-0.5 * (effective_r / sigma) ** 2)
    return max(0.0, min(1.0, frac))


def compute_crowding_metric(
    target_mag: float,
    neighbor_mags: list[float],
    neighbor_separations_arcsec: list[float],
    *,
    aperture_radius_arcsec: float = 21.0,   # TESS pixel ≈ 21 arcsec
    psf_fwhm_arcsec: float = 21.0,
) -> CrowdingResult:
    """Compute the aperture crowding metric.

    Args:
        target_mag: Magnitude of the target star (TESS band).
        neighbor_mags: Magnitudes of nearby stars.
        neighbor_separations_arcsec: Angular separations (arcsec) from the
            target; must have same length as *neighbor_mags*.
        aperture_radius_arcsec: Photometric aperture radius (arcsec).
        psf_fwhm_arcsec: PSF full-width at half-maximum (arcsec).

    Returns:
        :class:`CrowdingResult`.
    """
    if len(neighbor_mags) != len(neighbor_separations_arcsec):
        return CrowdingResult(0.0, 0.0, None, None, None, 0, "INVALID")
    if aperture_radius_arcsec <= 0 or psf_fwhm_arcsec <= 0:
        return CrowdingResult(0.0, 0.0, None, None, None, 0, "INVALID")

    # Target always fully inside aperture (separation = 0)
    target_flux = 10.0 ** (-0.4 * target_mag)

    neighbor_fluxes_in_ap: list[float] = []
    for mag, sep in zip(neighbor_mags, neighbor_separations_arcsec, strict=False):
        f = 10.0 ** (-0.4 * mag)
        frac = _flux_in_aperture(sep, aperture_radius_arcsec, psf_fwhm_arcsec)
        neighbor_fluxes_in_ap.append(f * frac)

    total_neighbor = sum(neighbor_fluxes_in_ap)
    total_flux = target_flux + total_neighbor

    if total_flux <= 0:
        return CrowdingResult(
            target_flux, total_flux, None, None, None, len(neighbor_mags), "INVALID"
        )

    crowding = round(target_flux / total_flux, 6)
    contamination = round(1.0 - crowding, 6)

    dil_per_neighbor = None
    if neighbor_fluxes_in_ap:
        avg_flux = total_neighbor / len(neighbor_fluxes_in_ap)
        dil_per_neighbor = round(avg_flux / target_flux * 1e6, 2)

    return CrowdingResult(
        target_flux=round(target_flux, 8),
        total_flux=round(total_flux, 8),
        crowding_metric=crowding,
        contamination_ratio=contamination,
        dilution_ppm_per_neighbor=dil_per_neighbor,
        n_neighbors=len(neighbor_mags),
        flag="OK",
    )


def format_crowding_result(result: CrowdingResult) -> str:
    """Format crowding metric result as Markdown."""
    lines = [
        "## Crowding Metric Calculator",
        "",
        f"- Neighbors in aperture: {result.n_neighbors}",
        f"- **Crowding metric (CROWDSAP equivalent): {result.crowding_metric}**",
        f"- Contamination ratio: {result.contamination_ratio}",
        f"- Avg dilution per neighbor: {result.dilution_ppm_per_neighbor} ppm",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="crowding_metric_calculator",
        description="Compute aperture crowding metric from nearby-star catalog.",
    )
    parser.add_argument("target_mag", type=float)
    parser.add_argument("--aperture-radius", type=float, default=21.0)
    args = parser.parse_args(argv)

    result = compute_crowding_metric(
        args.target_mag, [], [], aperture_radius_arcsec=args.aperture_radius
    )
    print(format_crowding_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
