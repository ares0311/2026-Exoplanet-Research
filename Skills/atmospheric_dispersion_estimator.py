"""Estimate differential atmospheric refraction between two wavelengths.

Refractive index formula (Edlen 1953 / standard form):
  n(lambda) = 1 + (64.328 + 29498.1/(146 - sigma^2) + 255.4/(41 - sigma^2)) * 1e-6
where sigma = 1 / lambda_um  (lambda in micrometres).

Differential dispersion:
  delta_r = 206265 * |n(lambda1) - n(lambda2)| * tan(zenith_angle)  [arcsec]

Public API
----------
DispersionResult(dispersion_arcsec, airmass, wavelength1_nm, wavelength2_nm,
                 zenith_angle_deg, flag)
estimate_atmospheric_dispersion(wavelength1_nm, wavelength2_nm, airmass)
    -> DispersionResult
format_dispersion_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DispersionResult:
    dispersion_arcsec: float
    airmass: float
    wavelength1_nm: float
    wavelength2_nm: float
    zenith_angle_deg: float
    flag: str  # "OK", "LARGE_DISPERSION"


def _refraction_index(wavelength_nm: float) -> float:
    """Compute refractive index using Edlen formula at STP."""
    lambda_um = wavelength_nm / 1000.0
    sigma = 1.0 / lambda_um  # sigma = 1/lambda in um^-1
    n_minus_1 = (
        64.328
        + 29498.1 / (146.0 - sigma**2)
        + 255.4 / (41.0 - sigma**2)
    ) * 1e-6
    return 1.0 + n_minus_1


def estimate_atmospheric_dispersion(
    wavelength1_nm: float,
    wavelength2_nm: float,
    airmass: float = 1.5,
) -> DispersionResult:
    """Estimate differential atmospheric dispersion.

    Args:
        wavelength1_nm: First wavelength in nanometres.
        wavelength2_nm: Second wavelength in nanometres.
        airmass: Observation airmass (>= 1.0).

    Returns:
        :class:`DispersionResult`.
    """
    airmass = max(1.0, airmass)

    # Zenith angle from airmass (plane-parallel approximation)
    cos_z = 1.0 / airmass
    cos_z = max(-1.0, min(1.0, cos_z))
    zenith_deg = math.degrees(math.acos(cos_z))
    tan_z = math.tan(math.radians(zenith_deg))

    n1 = _refraction_index(wavelength1_nm)
    n2 = _refraction_index(wavelength2_nm)

    dispersion_arcsec = 206265.0 * abs(n1 - n2) * tan_z

    flag = "LARGE_DISPERSION" if dispersion_arcsec > 1.0 else "OK"

    return DispersionResult(
        dispersion_arcsec=round(dispersion_arcsec, 4),
        airmass=round(airmass, 3),
        wavelength1_nm=wavelength1_nm,
        wavelength2_nm=wavelength2_nm,
        zenith_angle_deg=round(zenith_deg, 3),
        flag=flag,
    )


def format_dispersion_result(result: DispersionResult) -> str:
    """Format dispersion result as Markdown."""
    lines = [
        "## Atmospheric Dispersion Estimate",
        "",
        f"- Wavelength 1: {result.wavelength1_nm:.1f} nm",
        f"- Wavelength 2: {result.wavelength2_nm:.1f} nm",
        f"- Airmass: {result.airmass:.3f}",
        f"- Zenith angle: {result.zenith_angle_deg:.2f}°",
        f"- Differential dispersion: {result.dispersion_arcsec:.4f} arcsec",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wave1", type=float, default=400.0, help="Wavelength 1 (nm)")
    p.add_argument("--wave2", type=float, default=700.0, help="Wavelength 2 (nm)")
    p.add_argument("--airmass", type=float, default=1.5)
    args = p.parse_args(argv)
    r = estimate_atmospheric_dispersion(args.wave1, args.wave2, args.airmass)
    print(format_dispersion_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
