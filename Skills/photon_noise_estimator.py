"""Estimate photon noise floor for a TESS target.

Converts TESS magnitude to an expected per-cadence photon noise estimate
using the TESS noise model (Sullivan et al. 2015 approximation).

Public API
----------
PhotonNoiseResult(tmag, cadence_sec, n_pixels, systematic_floor_ppm,
                  photon_noise_ppm, total_noise_ppm, cdpp_ppm_hr, flag)
estimate_photon_noise(tmag, *, cadence_sec, n_pixels,
                      systematic_floor_ppm) -> PhotonNoiseResult
format_photon_noise_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# TESS photon collection rate: ~1.57×10⁸ e⁻/s at Tmag=0 (approximate)
_TESS_ZERO_FLUX_E_PER_S = 1.57e8
_READ_NOISE_E = 10.0          # e⁻ per pixel per read
_DARK_CURRENT_E_PER_S = 0.0   # TESS dark current is negligible


@dataclass(frozen=True)
class PhotonNoiseResult:
    tmag: float
    cadence_sec: float
    n_pixels: int
    systematic_floor_ppm: float
    photon_noise_ppm: float
    total_noise_ppm: float
    cdpp_ppm_hr: float          # combined noise scaled to 1-hour baseline
    flag: str                   # "OK", "BRIGHT", "FAINT", "INVALID"


def estimate_photon_noise(
    tmag: float,
    *,
    cadence_sec: float = 120.0,
    n_pixels: int = 4,
    systematic_floor_ppm: float = 60.0,
) -> PhotonNoiseResult:
    """Estimate per-cadence photon noise for a TESS target.

    Args:
        tmag: TESS magnitude.
        cadence_sec: Integration time in seconds (default 120 s).
        n_pixels: Number of pixels in the photometric aperture.
        systematic_floor_ppm: Irreducible systematic noise floor in ppm.

    Returns:
        :class:`PhotonNoiseResult`.
    """
    if cadence_sec <= 0 or n_pixels <= 0:
        return PhotonNoiseResult(
            tmag, cadence_sec, n_pixels, systematic_floor_ppm,
            0.0, 0.0, 0.0, "INVALID",
        )

    # Source flux in electrons per cadence
    flux_e = _TESS_ZERO_FLUX_E_PER_S * 10 ** (-0.4 * tmag) * cadence_sec
    # Shot noise
    shot_noise_e = math.sqrt(flux_e)
    # Read noise per cadence (one read per cadence assumed)
    read_noise_e = math.sqrt(n_pixels) * _READ_NOISE_E

    if flux_e <= 0:
        photon_noise_ppm = 0.0
        total_noise_ppm = 0.0
    else:
        photon_noise_ppm = (shot_noise_e / flux_e) * 1e6
        noise_from_read = (read_noise_e / flux_e) * 1e6
        quadrature_ppm = math.sqrt(photon_noise_ppm ** 2 + noise_from_read ** 2)
        total_noise_ppm = math.sqrt(quadrature_ppm ** 2 + systematic_floor_ppm ** 2)

    # CDPP: noise per sqrt(N_cadences in 1 hr)
    n_cadences_per_hr = 3600.0 / cadence_sec
    cdpp_ppm_hr = total_noise_ppm / math.sqrt(n_cadences_per_hr)

    if tmag < 6:
        flag = "BRIGHT"
    elif tmag > 16:
        flag = "FAINT"
    else:
        flag = "OK"

    return PhotonNoiseResult(
        tmag=tmag,
        cadence_sec=cadence_sec,
        n_pixels=n_pixels,
        systematic_floor_ppm=systematic_floor_ppm,
        photon_noise_ppm=round(photon_noise_ppm, 2),
        total_noise_ppm=round(total_noise_ppm, 2),
        cdpp_ppm_hr=round(cdpp_ppm_hr, 2),
        flag=flag,
    )


def format_photon_noise_result(result: PhotonNoiseResult) -> str:
    """Format photon noise result as Markdown."""
    lines = [
        "## Photon Noise Estimate",
        "",
        f"- Tmag: {result.tmag:.2f}",
        f"- Cadence: {result.cadence_sec:.0f} s",
        f"- Aperture pixels: {result.n_pixels}",
        f"- Systematic floor: {result.systematic_floor_ppm:.1f} ppm",
        f"- Photon noise: {result.photon_noise_ppm:.2f} ppm",
        f"- Total noise: {result.total_noise_ppm:.2f} ppm",
        f"- CDPP (1 hr): {result.cdpp_ppm_hr:.2f} ppm",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="photon_noise_estimator",
        description="Estimate photon noise for a TESS target.",
    )
    parser.add_argument("tmag", type=float)
    parser.add_argument("--cadence-sec", type=float, default=120.0)
    parser.add_argument("--n-pixels", type=int, default=4)
    parser.add_argument("--systematic-floor-ppm", type=float, default=60.0)
    args = parser.parse_args(argv)

    result = estimate_photon_noise(
        args.tmag,
        cadence_sec=args.cadence_sec,
        n_pixels=args.n_pixels,
        systematic_floor_ppm=args.systematic_floor_ppm,
    )
    print(format_photon_noise_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
