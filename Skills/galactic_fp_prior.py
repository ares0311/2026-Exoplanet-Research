"""Estimate galactic background false-positive prior from sky coordinates."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class GalacticFPPriorResult:
    stellar_density_per_sq_deg: float   # background star density (stars/deg²)
    bgeb_prior: float                   # background EB prior probability
    blend_probability: float            # probability of chance alignment
    galactic_latitude_deg: float
    fp_prior_class: str                 # LOW / MODERATE / HIGH / VERY_HIGH
    flag: str


def estimate_galactic_fp_prior(
    galactic_latitude_deg: float,
    galactic_longitude_deg: float = 0.0,
    aperture_radius_arcsec: float = 10.0,
    limiting_magnitude: float = 20.0,
    target_magnitude: float = 12.0,
) -> GalacticFPPriorResult:
    """Estimate background eclipsing binary (BEB) false-positive prior.

    Uses simplified Galactic stellar density model (Robin+2003 approximation):
      N(b) ≈ N_0 × exp(-|b| / b_scale)  stars/deg²
    where N_0 ~ 10^4 at plane and b_scale ~ 20 degrees.

    BEB prior (Morton & Johnson 2011):
      P_BEB = N_stars_in_aperture × P_EB × (ΔF/F)²
    where P_EB ~ 0.5% eclipsing binary fraction.

    Args:
        galactic_latitude_deg: galactic latitude b (degrees)
        galactic_longitude_deg: galactic longitude l (degrees)
        aperture_radius_arcsec: photometric aperture radius (arcsec)
        limiting_magnitude: depth limit for background stars
        target_magnitude: target star magnitude
    """
    if not (-90.0 <= galactic_latitude_deg <= 90.0):
        return GalacticFPPriorResult(float("nan"), float("nan"), float("nan"),
                                      galactic_latitude_deg, "UNKNOWN",
                                      "INVALID_LATITUDE")
    if aperture_radius_arcsec <= 0.0:
        return GalacticFPPriorResult(float("nan"), float("nan"), float("nan"),
                                      galactic_latitude_deg, "UNKNOWN",
                                      "INVALID_APERTURE")

    b = abs(galactic_latitude_deg)

    # Simplified Robin+2003: stellar density per deg² to limiting magnitude
    b_scale = 20.0  # degrees
    n_plane = 1e4   # stars/deg² at b=0 to V<20
    n_density = n_plane * math.exp(-b / b_scale)

    # Number of background stars in aperture
    aperture_sq_deg = math.pi * (aperture_radius_arcsec / 3600.0)**2
    n_in_aperture = n_density * aperture_sq_deg

    # Magnitude range of relevant background stars
    delta_mag = limiting_magnitude - target_magnitude
    frac_bright = min(delta_mag / 10.0, 1.0)  # rough fraction at relevant depth
    n_relevant = max(n_in_aperture * frac_bright, 0.0)

    # Background EB prior
    p_eb_fraction = 0.005   # 0.5% EB fraction
    # Dilution: BEB at delta_m would have depth inflated by ~10^(delta_m/2.5)
    bgeb_prior = n_relevant * p_eb_fraction

    # Chance blend probability (any background star)
    blend_prob = min(1.0 - math.exp(-n_in_aperture * 0.001), 0.5)

    # FP prior class
    if bgeb_prior < 1e-4:
        fp_class = "LOW"
    elif bgeb_prior < 1e-3:
        fp_class = "MODERATE"
    elif bgeb_prior < 1e-2:
        fp_class = "HIGH"
    else:
        fp_class = "VERY_HIGH"

    return GalacticFPPriorResult(
        stellar_density_per_sq_deg=n_density,
        bgeb_prior=bgeb_prior,
        blend_probability=blend_prob,
        galactic_latitude_deg=galactic_latitude_deg,
        fp_prior_class=fp_class,
        flag="OK",
    )


def format_galactic_fp_prior_result(r: GalacticFPPriorResult) -> str:
    if r.flag != "OK":
        return f"GalacticFPPrior | flag={r.flag}"
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Galactic latitude | {r.galactic_latitude_deg:.1f} ° |\n"
        f"| Stellar density | {r.stellar_density_per_sq_deg:.1f} /deg² |\n"
        f"| BEB prior | {r.bgeb_prior:.2e} |\n"
        f"| Blend probability | {r.blend_probability:.4f} |\n"
        f"| FP prior class | {r.fp_prior_class} |\n"
        f"| Flag | {r.flag} |"
    )


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Galactic false-positive prior estimator")
    p.add_argument("galactic_latitude_deg", type=float)
    p.add_argument("--aperture-arcsec", type=float, default=10.0)
    p.add_argument("--target-mag", type=float, default=12.0)
    args = p.parse_args()
    r = estimate_galactic_fp_prior(args.galactic_latitude_deg,
                                    aperture_radius_arcsec=args.aperture_arcsec,
                                    target_magnitude=args.target_mag)
    print(format_galactic_fp_prior_result(r))


if __name__ == "__main__":
    _cli()
