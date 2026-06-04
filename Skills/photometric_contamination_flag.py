"""Flag photometric contamination level from a nearby source magnitude difference."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ContaminationFlagResult:
    delta_mag: float
    flux_ratio: float
    dilution_factor: float
    depth_correction_factor: float
    contamination_level: str    # NEGLIGIBLE / LOW / MODERATE / SEVERE / EXTREME
    depth_bias_ppm_per_1000ppm: float
    flag: str


def compute_photometric_contamination_flag(
    target_magnitude: float,
    neighbour_magnitude: float,
    separation_arcsec: float | None = None,
    aperture_radius_arcsec: float | None = None,
) -> ContaminationFlagResult:
    """Estimate photometric contamination from a neighbouring source.

    Computes the flux ratio and dilution factor assuming the neighbour is fully
    within the photometric aperture. If aperture/separation are provided, applies
    a Gaussian PSF fraction correction.

    Args:
        target_magnitude: apparent magnitude of target star
        neighbour_magnitude: apparent magnitude of neighbour
        separation_arcsec: angular separation from target (arcsec); optional
        aperture_radius_arcsec: photometric aperture radius (arcsec); optional
    """
    delta_mag = neighbour_magnitude - target_magnitude

    # Flux ratio: neighbour / target
    flux_ratio = 10.0 ** (-0.4 * delta_mag)

    # PSF fraction in aperture (Gaussian approximation)
    psf_fraction = 1.0
    if separation_arcsec is not None and aperture_radius_arcsec is not None:
        if aperture_radius_arcsec <= 0.0:
            return ContaminationFlagResult(
                delta_mag=delta_mag, flux_ratio=flux_ratio,
                dilution_factor=float("nan"), depth_correction_factor=float("nan"),
                contamination_level="UNKNOWN", depth_bias_ppm_per_1000ppm=float("nan"),
                flag="INVALID_APERTURE",
            )
        # Fraction of Gaussian PSF (sigma = aperture/2) within aperture at given separation
        sigma = aperture_radius_arcsec / 2.0
        if separation_arcsec > 0.0:
            psf_fraction = math.exp(-0.5 * (separation_arcsec / sigma) ** 2)
        # else fully centred = 1.0

    effective_flux_ratio = flux_ratio * psf_fraction
    dilution = effective_flux_ratio / (1.0 + effective_flux_ratio)
    depth_correction = 1.0 + effective_flux_ratio

    bias_per_1000 = dilution * 1000.0

    if dilution < 0.005:
        level = "NEGLIGIBLE"
    elif dilution < 0.02:
        level = "LOW"
    elif dilution < 0.05:
        level = "MODERATE"
    elif dilution < 0.15:
        level = "SEVERE"
    else:
        level = "EXTREME"

    return ContaminationFlagResult(
        delta_mag=delta_mag,
        flux_ratio=effective_flux_ratio,
        dilution_factor=dilution,
        depth_correction_factor=depth_correction,
        contamination_level=level,
        depth_bias_ppm_per_1000ppm=bias_per_1000,
        flag="OK",
    )


def format_contamination_flag_result(r: ContaminationFlagResult) -> str:
    if r.flag != "OK":
        return f"ContaminationFlag | flag={r.flag}"
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Δmag (neighbour − target) | {r.delta_mag:+.2f} |\n"
        f"| Flux ratio (neighbour/target) | {r.flux_ratio:.4f} |\n"
        f"| Dilution factor | {r.dilution_factor:.4f} |\n"
        f"| Depth correction factor | {r.depth_correction_factor:.4f} |\n"
        f"| Depth bias | {r.depth_bias_ppm_per_1000ppm:.1f} ppm per 1000 ppm |\n"
        f"| Contamination level | {r.contamination_level} |\n"
        f"| Flag | {r.flag} |"
    )


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Photometric contamination flag")
    p.add_argument("target_mag", type=float, help="Target apparent magnitude")
    p.add_argument("neighbour_mag", type=float, help="Neighbour apparent magnitude")
    p.add_argument("--sep", type=float, default=None, help="Separation (arcsec)")
    p.add_argument("--aperture", type=float, default=None, help="Aperture radius (arcsec)")
    args = p.parse_args()
    r = compute_photometric_contamination_flag(
        args.target_mag, args.neighbour_mag,
        separation_arcsec=args.sep, aperture_radius_arcsec=args.aperture,
    )
    print(format_contamination_flag_result(r))


if __name__ == "__main__":
    _cli()
