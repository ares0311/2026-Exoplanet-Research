"""Compute a photon noise budget for ground-based transit photometry.

Noise components:
  - Photon noise: 1e6 / sqrt(signal_e)  [ppm]
  - Sky noise: 1e6 * sqrt(sky_e) / signal_e  [ppm]
  - Scintillation: Young (1967) formula  [ppm]
  - Read noise: 1e6 * sqrt(n_pix) * read_noise_e / signal_e  [ppm]
  - Total: quadrature sum

Public API
----------
PrecisionBudget(photon_noise_ppm, sky_noise_ppm, scint_noise_ppm, read_noise_ppm,
                total_noise_ppm, flag)
compute_precision_budget(flux_adu, exptime_s, ...) -> PrecisionBudget
format_precision_budget(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PrecisionBudget:
    photon_noise_ppm: float
    sky_noise_ppm: float
    scint_noise_ppm: float
    read_noise_ppm: float
    total_noise_ppm: float
    flag: str  # "OK", "SKY_LIMITED", "SCINT_LIMITED", "READ_LIMITED"


def compute_precision_budget(
    flux_adu: float,
    exptime_s: float,
    gain: float = 1.5,
    n_pix_aperture: int = 50,
    sky_adu_pix: float = 100.0,
    read_noise_e: float = 10.0,
    aperture_cm: float = 25.0,
    airmass: float = 1.2,
    elevation_m: float = 500.0,
) -> PrecisionBudget:
    """Compute the photometric noise budget.

    Args:
        flux_adu: Target flux in ADU per second.
        exptime_s: Exposure time in seconds.
        gain: CCD gain in e⁻/ADU.
        n_pix_aperture: Number of pixels in the aperture.
        sky_adu_pix: Sky background in ADU per pixel per second.
        read_noise_e: Read noise in electrons per pixel.
        aperture_cm: Telescope aperture diameter in cm.
        airmass: Observation airmass.
        elevation_m: Site elevation in metres.

    Returns:
        :class:`PrecisionBudget`.
    """
    # Signal in electrons
    signal_e = flux_adu * gain * exptime_s
    signal_e = max(signal_e, 1.0)  # avoid division by zero

    # Photon noise [ppm]
    photon_noise_ppm = 1e6 / math.sqrt(signal_e)

    # Sky noise [ppm]
    sky_e = sky_adu_pix * n_pix_aperture * gain * exptime_s
    sky_noise_ppm = 1e6 * math.sqrt(sky_e) / signal_e

    # Scintillation noise — Young (1967) approximation [ppm]
    # sigma_scint = 0.09 * D_cm^(-2/3) * X^(7/4) * exp(-h/8000) / sqrt(2*t)
    scint_frac = (
        0.09
        * aperture_cm ** (-2.0 / 3.0)
        * airmass ** (7.0 / 4.0)
        * math.exp(-elevation_m / 8000.0)
        / math.sqrt(2.0 * exptime_s)
    )
    scint_noise_ppm = 1e6 * scint_frac

    # Read noise [ppm]
    read_noise_ppm = 1e6 * math.sqrt(n_pix_aperture) * read_noise_e / signal_e

    # Total quadrature sum [ppm]
    total_noise_ppm = math.sqrt(
        photon_noise_ppm**2
        + sky_noise_ppm**2
        + scint_noise_ppm**2
        + read_noise_ppm**2
    )

    # Determine dominant noise source
    components = {
        "SKY_LIMITED": sky_noise_ppm,
        "SCINT_LIMITED": scint_noise_ppm,
        "READ_LIMITED": read_noise_ppm,
    }
    dominant = max(components, key=lambda k: components[k])
    # Only flag non-photon noise sources if they exceed photon noise
    max_other = max(sky_noise_ppm, scint_noise_ppm, read_noise_ppm)
    flag = dominant if max_other > photon_noise_ppm else "OK"

    return PrecisionBudget(
        photon_noise_ppm=round(photon_noise_ppm, 2),
        sky_noise_ppm=round(sky_noise_ppm, 2),
        scint_noise_ppm=round(scint_noise_ppm, 2),
        read_noise_ppm=round(read_noise_ppm, 2),
        total_noise_ppm=round(total_noise_ppm, 2),
        flag=flag,
    )


def format_precision_budget(result: PrecisionBudget) -> str:
    """Format precision budget as Markdown."""
    lines = [
        "## Photometry Precision Budget",
        "",
        "| Noise Source    | ppm      |",
        "|-----------------|----------|",
        f"| Photon noise    | {result.photon_noise_ppm:.2f} |",
        f"| Sky noise       | {result.sky_noise_ppm:.2f} |",
        f"| Scintillation   | {result.scint_noise_ppm:.2f} |",
        f"| Read noise      | {result.read_noise_ppm:.2f} |",
        f"| **Total**       | **{result.total_noise_ppm:.2f}** |",
        "",
        f"Flag: **{result.flag}**",
    ]
    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--flux-adu", type=float, default=10000.0)
    p.add_argument("--exptime-s", type=float, default=60.0)
    p.add_argument("--gain", type=float, default=1.5)
    p.add_argument("--n-pix", type=int, default=50)
    p.add_argument("--sky-adu-pix", type=float, default=100.0)
    p.add_argument("--read-noise-e", type=float, default=10.0)
    p.add_argument("--aperture-cm", type=float, default=25.0)
    p.add_argument("--airmass", type=float, default=1.2)
    p.add_argument("--elevation-m", type=float, default=500.0)
    args = p.parse_args(argv)
    r = compute_precision_budget(
        args.flux_adu, args.exptime_s, args.gain, args.n_pix, args.sky_adu_pix,
        args.read_noise_e, args.aperture_cm, args.airmass, args.elevation_m,
    )
    print(format_precision_budget(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
