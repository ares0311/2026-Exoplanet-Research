"""Estimate SNR for ground-based photometric follow-up of a transit."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

# TESS magnitude → flux conversion baseline: Tmag=10 gives ~1e6 e-/s on a 1-m telescope
_FLUX_REF_EPER_S = 1.0e6   # e-/s at Tmag=10 for 1-m aperture
_TMAG_REF = 10.0


@dataclass(frozen=True)
class GroundSnrResult:
    depth_ppm: float
    tmag: float
    aperture_m: float
    exptime_s: float
    n_transits: int
    signal_ppm: float
    noise_ppm: float
    snr_single: float
    snr_stacked: float
    flag: str


def estimate_ground_snr(
    depth_ppm: float,
    tmag: float,
    aperture_m: float = 1.0,
    exptime_s: float = 60.0,
    n_transits: int = 1,
    sky_background_eper_s: float = 500.0,
    read_noise_e: float = 10.0,
    scintillation_frac: float = 0.001,
    n_comparison_stars: int = 3,
) -> GroundSnrResult:
    """
    Estimate differential-photometry SNR for a ground-based transit detection.

    Noise sources: photon noise (target + sky), read noise, scintillation,
    and comparison-star photon noise (divided by sqrt(n_comp)).
    """
    for name, val in [
        ("depth_ppm", depth_ppm),
        ("tmag", tmag),
        ("aperture_m", aperture_m),
        ("exptime_s", exptime_s),
    ]:
        if not math.isfinite(val) or val <= 0.0:
            return GroundSnrResult(
                depth_ppm=depth_ppm, tmag=tmag, aperture_m=aperture_m,
                exptime_s=exptime_s, n_transits=n_transits,
                signal_ppm=float("nan"), noise_ppm=float("nan"),
                snr_single=float("nan"), snr_stacked=float("nan"),
                flag=f"INVALID_{name.upper()}",
            )
    if n_transits < 1:
        return GroundSnrResult(
            depth_ppm=depth_ppm, tmag=tmag, aperture_m=aperture_m,
            exptime_s=exptime_s, n_transits=n_transits,
            signal_ppm=float("nan"), noise_ppm=float("nan"),
            snr_single=float("nan"), snr_stacked=float("nan"),
            flag="INVALID_N_TRANSITS",
        )

    # Flux scales with aperture area and exposure time, and with 10^(0.4*(Tmag_ref - Tmag))
    area_ratio = (aperture_m / 1.0) ** 2
    flux_ratio = 10.0 ** (0.4 * (_TMAG_REF - tmag))
    n_electrons = _FLUX_REF_EPER_S * area_ratio * flux_ratio * exptime_s

    # Individual noise components (in e-)
    photon_noise_e2 = n_electrons
    sky_noise_e2 = sky_background_eper_s * area_ratio * exptime_s
    read_noise_e2 = read_noise_e**2
    scint_noise_e2 = (scintillation_frac * n_electrons) ** 2
    comp_noise_e2 = (n_electrons / n_comparison_stars) if n_comparison_stars > 0 else 0.0

    total_noise_e = math.sqrt(
        photon_noise_e2 + sky_noise_e2 + read_noise_e2 + scint_noise_e2 + comp_noise_e2
    )

    noise_frac = total_noise_e / n_electrons if n_electrons > 0 else float("inf")
    noise_ppm = noise_frac * 1e6
    signal_ppm = depth_ppm

    snr_single = signal_ppm / noise_ppm if noise_ppm > 0 else 0.0
    snr_stacked = snr_single * math.sqrt(n_transits)

    return GroundSnrResult(
        depth_ppm=depth_ppm,
        tmag=tmag,
        aperture_m=aperture_m,
        exptime_s=exptime_s,
        n_transits=n_transits,
        signal_ppm=round(signal_ppm, 2),
        noise_ppm=round(noise_ppm, 2),
        snr_single=round(snr_single, 3),
        snr_stacked=round(snr_stacked, 3),
        flag="OK",
    )


def format_ground_snr(r: GroundSnrResult) -> str:
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Depth (ppm) | {r.depth_ppm:.1f} |\n"
        f"| T mag | {r.tmag:.2f} |\n"
        f"| Aperture (m) | {r.aperture_m:.2f} |\n"
        f"| Exp time (s) | {r.exptime_s:.1f} |\n"
        f"| N transits stacked | {r.n_transits} |\n"
        f"| Noise (ppm/exp) | {r.noise_ppm:.2f} |\n"
        f"| SNR (single transit) | {r.snr_single:.3f} |\n"
        f"| SNR (stacked) | {r.snr_stacked:.3f} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Estimate ground-based photometry SNR.")
    p.add_argument("depth_ppm", type=float)
    p.add_argument("tmag", type=float)
    p.add_argument("--aperture-m", type=float, default=1.0)
    p.add_argument("--exptime-s", type=float, default=60.0)
    p.add_argument("--n-transits", type=int, default=1)
    args = p.parse_args()
    r = estimate_ground_snr(
        args.depth_ppm, args.tmag,
        aperture_m=args.aperture_m,
        exptime_s=args.exptime_s,
        n_transits=args.n_transits,
    )
    print(format_ground_snr(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
