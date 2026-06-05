"""Correct for geometric and photometric detection bias in transit survey occurrence rates."""
from __future__ import annotations

import math
from dataclasses import dataclass

_G = 6.674e-11
_MSUN_KG = 1.989e30
_RSUN_M = 6.957e8
_AU_M = 1.495978707e11


@dataclass(frozen=True)
class DetectionBiasResult:
    geometric_probability: float
    photometric_completeness: float
    combined_efficiency: float
    occurrence_rate_correction: float   # = 1 / combined_efficiency
    n_expected_per_star: float
    flag: str


def compute_detection_bias(
    period_days: float,
    planet_radius_rearth: float,
    stellar_mass_msun: float = 1.0,
    stellar_radius_rsun: float = 1.0,
    cdpp_ppm_per_hour: float = 100.0,
    n_transits_observed: float = 3.0,
    snr_threshold: float = 7.5,
    n_stars_surveyed: int = 1,
    n_detections: int = 1,
) -> DetectionBiasResult:
    """Compute geometric + photometric efficiency for transit survey occurrence rates.

    Geometric probability: p_geom = (R★ + Rp) / a
    Photometric completeness: p_photo = P(SNR > threshold | N_tr, CDPP)
      SNR = (Rp/Rs)² × sqrt(N_tr × T_tr / τ_CDPP) / CDPP

    Occurrence rate = N_det / (N_stars × p_geom × p_photo)

    Args:
        period_days: orbital period (days)
        planet_radius_rearth: planet radius (Earth radii)
        stellar_mass_msun: stellar mass (solar masses)
        stellar_radius_rsun: stellar radius (solar radii)
        cdpp_ppm_per_hour: combined differential photometric precision (ppm/√hr)
        n_transits_observed: number of observed transits
        snr_threshold: detection SNR threshold
        n_stars_surveyed: number of stars in survey
        n_detections: number of planets detected
    """
    _REARTH_M = 6.371e6

    if period_days <= 0.0:
        return DetectionBiasResult(float("nan"), float("nan"), float("nan"),
                                    float("nan"), float("nan"), "INVALID_PERIOD")
    if planet_radius_rearth <= 0.0:
        return DetectionBiasResult(float("nan"), float("nan"), float("nan"),
                                    float("nan"), float("nan"), "INVALID_RADIUS")

    ms_kg = stellar_mass_msun * _MSUN_KG
    rs_m = stellar_radius_rsun * _RSUN_M
    rp_m = planet_radius_rearth * _REARTH_M
    p_s = period_days * 86400.0

    a_m = (_G * ms_kg * p_s**2 / (4.0 * math.pi**2)) ** (1.0 / 3.0)

    p_geom = min((rs_m + rp_m) / a_m, 1.0)

    depth_ppm = (rp_m / rs_m)**2 * 1e6
    t_tr_hr = (period_days / math.pi) * math.asin(rs_m / a_m) * 24.0
    t_tr_hr = max(t_tr_hr, 0.5)

    snr = depth_ppm * math.sqrt(n_transits_observed * t_tr_hr) / max(cdpp_ppm_per_hour, 1.0)
    p_photo = min(max((snr - snr_threshold) / snr_threshold, 0.0), 1.0) if snr > 0 else 0.0
    # Step function approx: fully detectable above threshold
    p_photo = 1.0 if snr >= snr_threshold else max(snr / snr_threshold - 0.5, 0.0)

    combined = p_geom * p_photo
    correction = 1.0 / combined if combined > 0.0 else float("inf")
    n_expected = n_detections * correction / max(n_stars_surveyed, 1)

    return DetectionBiasResult(
        geometric_probability=p_geom,
        photometric_completeness=p_photo,
        combined_efficiency=combined,
        occurrence_rate_correction=correction,
        n_expected_per_star=n_expected,
        flag="OK",
    )


def format_detection_bias_result(r: DetectionBiasResult) -> str:
    if r.flag != "OK":
        return f"DetectionBias | flag={r.flag}"
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Geometric probability | {r.geometric_probability:.4f} |\n"
        f"| Photometric completeness | {r.photometric_completeness:.4f} |\n"
        f"| Combined efficiency | {r.combined_efficiency:.4f} |\n"
        f"| Occurrence rate correction | {r.occurrence_rate_correction:.2f}× |\n"
        f"| Expected planets per star | {r.n_expected_per_star:.4f} |\n"
        f"| Flag | {r.flag} |"
    )


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Transit detection bias corrector")
    p.add_argument("period_days", type=float)
    p.add_argument("planet_radius_rearth", type=float)
    p.add_argument("--cdpp", type=float, default=100.0)
    p.add_argument("--ntr", type=float, default=3.0)
    args = p.parse_args()
    r = compute_detection_bias(args.period_days, args.planet_radius_rearth,
                                cdpp_ppm_per_hour=args.cdpp, n_transits_observed=args.ntr)
    print(format_detection_bias_result(r))


if __name__ == "__main__":
    _cli()
