"""Estimate angular separation and contrast ratio for direct imaging of exoplanets."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

_AU_M = 1.496e11
_R_EARTH_M = 6.371e6
_PC_M = 3.086e16   # parsec in metres


@dataclass(frozen=True)
class DirectImagingResult:
    orbital_distance_au: float
    stellar_distance_pc: float
    planet_radius_rearth: float
    geometric_albedo: float
    angular_separation_arcsec: float
    contrast_ratio: float            # reflected light at quadrature
    contrast_mag_diff: float         # delta-magnitude
    detectable_gpi: bool             # GPI/SPHERE inner-working-angle ~0.1 arcsec, contrast ~1e-6
    detectable_jwst_nircam: bool     # JWST NIRCam ~0.1 arcsec, contrast ~1e-7
    detectable_roman: bool           # Roman CGI ~0.1 arcsec, contrast ~1e-9
    flag: str


def compute_direct_imaging_contrast(
    orbital_distance_au: float,
    stellar_distance_pc: float,
    planet_radius_rearth: float,
    geometric_albedo: float = 0.3,
) -> DirectImagingResult:
    """
    Compute angular separation and reflected-light contrast ratio.

    Angular separation: θ = a / d  [arcsec]  (at quadrature, max elongation)
    Contrast ratio at quadrature: C = Ag * (Rp / (2a))²

    Instrument thresholds (approximate):
      GPI/SPHERE: IWA 0.1", contrast ~1e-6
      JWST NIRCam coronagraph: IWA ~0.1", contrast ~1e-7
      Roman Space Telescope CGI: IWA ~0.1", contrast ~1e-9

    Parameters
    ----------
    orbital_distance_au:  Semi-major axis in AU.
    stellar_distance_pc:  Distance to star in parsecs.
    planet_radius_rearth: Planet radius in Earth radii.
    geometric_albedo:     Geometric albedo (default 0.3).
    """
    if not math.isfinite(orbital_distance_au) or orbital_distance_au <= 0:
        return DirectImagingResult(
            orbital_distance_au, stellar_distance_pc, planet_radius_rearth,
            geometric_albedo, float("nan"), float("nan"), float("nan"),
            False, False, False, "INVALID_DISTANCE",
        )
    if not math.isfinite(stellar_distance_pc) or stellar_distance_pc <= 0:
        return DirectImagingResult(
            orbital_distance_au, stellar_distance_pc, planet_radius_rearth,
            geometric_albedo, float("nan"), float("nan"), float("nan"),
            False, False, False, "INVALID_STELLAR_DISTANCE",
        )
    if not math.isfinite(planet_radius_rearth) or planet_radius_rearth <= 0:
        return DirectImagingResult(
            orbital_distance_au, stellar_distance_pc, planet_radius_rearth,
            geometric_albedo, float("nan"), float("nan"), float("nan"),
            False, False, False, "INVALID_RADIUS",
        )

    ang_sep_arcsec = orbital_distance_au / stellar_distance_pc

    rp_m = planet_radius_rearth * _R_EARTH_M
    a_m = orbital_distance_au * _AU_M
    contrast = geometric_albedo * (rp_m / (2.0 * a_m)) ** 2

    delta_mag = -2.5 * math.log10(contrast) if contrast > 0 else float("inf")

    detectable_gpi = ang_sep_arcsec >= 0.1 and contrast >= 1e-6
    detectable_jwst = ang_sep_arcsec >= 0.1 and contrast >= 1e-7
    detectable_roman = ang_sep_arcsec >= 0.1 and contrast >= 1e-9

    return DirectImagingResult(
        orbital_distance_au=orbital_distance_au,
        stellar_distance_pc=stellar_distance_pc,
        planet_radius_rearth=planet_radius_rearth,
        geometric_albedo=geometric_albedo,
        angular_separation_arcsec=round(ang_sep_arcsec, 8),
        contrast_ratio=contrast,
        contrast_mag_diff=round(delta_mag, 2),
        detectable_gpi=detectable_gpi,
        detectable_jwst_nircam=detectable_jwst,
        detectable_roman=detectable_roman,
        flag="OK",
    )


def format_direct_imaging_result(r: DirectImagingResult) -> str:
    def _f(v: float, fmt: str = ".6f") -> str:
        return format(v, fmt) if math.isfinite(v) else "N/A"

    return (
        f"| Parameter | Value |\n|---|---|\n"
        f"| Angular separation (arcsec) | {_f(r.angular_separation_arcsec)} |\n"
        f"| Contrast ratio | {_f(r.contrast_ratio, '.4e')} |\n"
        f"| Contrast (Δmag) | {_f(r.contrast_mag_diff, '.2f')} |\n"
        f"| Detectable (GPI/SPHERE) | {r.detectable_gpi} |\n"
        f"| Detectable (JWST NIRCam) | {r.detectable_jwst_nircam} |\n"
        f"| Detectable (Roman CGI) | {r.detectable_roman} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Estimate direct imaging contrast and separation.")
    p.add_argument("orbital_distance_au", type=float)
    p.add_argument("stellar_distance_pc", type=float)
    p.add_argument("planet_radius_rearth", type=float)
    p.add_argument("--geometric-albedo", type=float, default=0.3)
    args = p.parse_args()
    r = compute_direct_imaging_contrast(
        args.orbital_distance_au, args.stellar_distance_pc,
        args.planet_radius_rearth, args.geometric_albedo,
    )
    print(format_direct_imaging_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
