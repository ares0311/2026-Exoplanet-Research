"""Estimate achievable radial velocity precision from stellar and instrument properties."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RvPrecisionResult:
    photon_noise_ms: float          # photon-noise RV uncertainty (m/s)
    stellar_jitter_ms: float        # activity jitter estimate (m/s)
    total_precision_ms: float       # quadrature sum
    n_photons_per_pixel: float      # estimated photons in CCF
    flag: str


def estimate_rv_precision(
    stellar_teff_k: float,
    stellar_vsini_kms: float,
    stellar_vmag: float,
    instrument_resolution: float = 115000.0,
    exposure_time_min: float = 15.0,
    telescope_diameter_m: float = 3.6,
    read_noise_e: float = 3.0,
    log_rhk: float = -4.9,
) -> RvPrecisionResult:
    """Estimate RV precision using Bouchy et al. (2001) photon noise formalism.

    σ_RV ≈ (c / R) × (vsini / sqrt(N_lines × SNR))
    Simplified: σ_RV ≈ FWHM_CCF / (SNR_CCF × Q_factor)
    where Q = 1 / (FWHM_CCF / c) and FWHM includes vsini broadening.

    Args:
        stellar_teff_k: stellar effective temperature (K)
        stellar_vsini_kms: projected rotational velocity (km/s)
        stellar_vmag: V-band magnitude
        instrument_resolution: spectrograph resolving power R = λ/Δλ
        exposure_time_min: exposure time (minutes)
        telescope_diameter_m: telescope primary diameter (m)
        read_noise_e: CCD read noise (electrons)
        log_rhk: log R'HK for jitter estimate (default: quiet Sun)
    """
    if stellar_teff_k <= 0.0:
        return RvPrecisionResult(float("nan"), float("nan"),
                                  float("nan"), float("nan"), "INVALID_TEFF")
    if stellar_vsini_kms < 0.0:
        return RvPrecisionResult(float("nan"), float("nan"),
                                  float("nan"), float("nan"), "INVALID_VSINI")
    if instrument_resolution <= 0.0:
        return RvPrecisionResult(float("nan"), float("nan"),
                                  float("nan"), float("nan"), "INVALID_RESOLUTION")
    if exposure_time_min <= 0.0:
        return RvPrecisionResult(float("nan"), float("nan"),
                                  float("nan"), float("nan"), "INVALID_EXPOSURE")

    # Instrument FWHM in km/s
    c_kms = 2.998e5
    fwhm_inst_kms = c_kms / instrument_resolution

    # vsini broadening adds in quadrature
    fwhm_total_kms = math.sqrt(fwhm_inst_kms**2 + stellar_vsini_kms**2)

    # Photon count: simplified V-band, zero-point ~3640 Jy, Δλ/λ = 1/R
    # Flux ~ 3640 × 10^(-vmag/2.5) Jy; 1 Jy = 1e-26 W/m²/Hz
    flux_jy = 3640.0 * 10.0 ** (-stellar_vmag / 2.5)
    area_m2 = math.pi * (telescope_diameter_m / 2.0) ** 2
    delta_nu_hz = c_kms * 1e3 / (550e-9) / instrument_resolution  # Hz per pixel
    t_s = exposure_time_min * 60.0
    n_photons = (flux_jy * 1e-26 * area_m2 * delta_nu_hz * t_s /
                 (6.626e-34 * c_kms * 1e3 / 550e-9))
    n_photons = max(n_photons, 1.0)

    snr_per_pixel = math.sqrt(n_photons) / math.sqrt(1.0 + read_noise_e**2 / n_photons)

    # RV quality factor Q ≈ c / (sqrt(8*ln2) * FWHM_total)
    q_factor = c_kms / (2.355 * fwhm_total_kms)

    # N_eff spectral lines contributing
    n_lines_eff = max(1000.0 * (stellar_teff_k / 5000.0) ** (-1.5), 10.0)
    n_lines_eff = min(n_lines_eff, 1e4)

    sigma_rv_kms = 1.0 / (q_factor * snr_per_pixel * math.sqrt(n_lines_eff))
    sigma_rv_ms = sigma_rv_kms * 1e3

    # Stellar jitter from Isaacson & Fischer (2010)
    log_jitter = 0.51 * log_rhk + 4.72
    jitter_ms = 10.0 ** log_jitter if -6.0 <= log_rhk <= -3.5 else 2.0

    total_ms = math.sqrt(sigma_rv_ms**2 + jitter_ms**2)

    return RvPrecisionResult(
        photon_noise_ms=sigma_rv_ms,
        stellar_jitter_ms=jitter_ms,
        total_precision_ms=total_ms,
        n_photons_per_pixel=n_photons,
        flag="OK",
    )


def format_rv_precision_result(r: RvPrecisionResult) -> str:
    if r.flag != "OK":
        return f"RvPrecision | flag={r.flag}"
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Photon noise | {r.photon_noise_ms:.2f} m/s |\n"
        f"| Stellar jitter | {r.stellar_jitter_ms:.2f} m/s |\n"
        f"| Total precision | {r.total_precision_ms:.2f} m/s |\n"
        f"| Photons per pixel | {r.n_photons_per_pixel:.0f} |\n"
        f"| Flag | {r.flag} |"
    )


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="RV precision estimator")
    p.add_argument("teff_k", type=float)
    p.add_argument("vsini_kms", type=float)
    p.add_argument("vmag", type=float)
    p.add_argument("--resolution", type=float, default=115000.0)
    p.add_argument("--texp", type=float, default=15.0)
    args = p.parse_args()
    r = estimate_rv_precision(args.teff_k, args.vsini_kms, args.vmag,
                               instrument_resolution=args.resolution,
                               exposure_time_min=args.texp)
    print(format_rv_precision_result(r))


if __name__ == "__main__":
    _cli()
