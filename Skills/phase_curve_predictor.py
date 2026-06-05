"""Predict reflected-light phase curve amplitude and thermal component."""
from __future__ import annotations

import math
from dataclasses import dataclass

_G = 6.674e-11
_MSUN_KG = 1.989e30
_RJUP_M = 7.1492e7
_RSUN_M = 6.957e8
_AU_M = 1.495978707e11
_SIGMA_SB = 5.6704e-8


@dataclass(frozen=True)
class PhaseCurveResult:
    reflected_amplitude_ppm: float   # reflected-light semi-amplitude
    thermal_amplitude_ppm: float     # thermal emission semi-amplitude
    total_amplitude_ppm: float       # combined
    secondary_eclipse_depth_ppm: float
    phase_offset_deg: float          # predicted hot-spot offset
    flag: str


def predict_phase_curve(
    period_days: float,
    stellar_teff_k: float,
    stellar_radius_rsun: float,
    planet_radius_rjup: float,
    geometric_albedo: float = 0.1,
    heat_redistribution: float = 0.5,
    stellar_mass_msun: float = 1.0,
    eccentricity: float = 0.0,
) -> PhaseCurveResult:
    """Predict optical phase curve amplitude for a close-in planet.

    Reflected light semi-amplitude (Winn 2010):
      A_ref = (Ag/2) * (Rp/a)²  [ppm, full amplitude = 2 * A_ref]

    Thermal emission (Cowan & Agol 2011):
      T_day = Teq * (1 - f_redist)^(1/4), T_night = Teq * f_redist^(1/4)
      A_therm ≈ (B(T_day) - B(T_night)) / B(T★) * (Rp/Rs)²

    Args:
        period_days: orbital period (days)
        stellar_teff_k: stellar effective temperature (K)
        stellar_radius_rsun: stellar radius (solar radii)
        planet_radius_rjup: planet radius (Jupiter radii)
        geometric_albedo: geometric albedo Ag (0–1)
        heat_redistribution: heat redistribution factor f (0=no, 0.5=full)
        stellar_mass_msun: stellar mass (solar masses)
        eccentricity: orbital eccentricity
    """
    if period_days <= 0.0:
        return PhaseCurveResult(float("nan"), float("nan"), float("nan"),
                                 float("nan"), float("nan"), "INVALID_PERIOD")
    if stellar_teff_k <= 0.0:
        return PhaseCurveResult(float("nan"), float("nan"), float("nan"),
                                 float("nan"), float("nan"), "INVALID_TEFF")
    if not (0.0 <= geometric_albedo <= 1.0):
        return PhaseCurveResult(float("nan"), float("nan"), float("nan"),
                                 float("nan"), float("nan"), "INVALID_ALBEDO")

    p_s = period_days * 86400.0
    ms_kg = stellar_mass_msun * _MSUN_KG
    a_m = (_G * ms_kg * p_s**2 / (4.0 * math.pi**2)) ** (1.0 / 3.0)
    rs_m = stellar_radius_rsun * _RSUN_M
    rp_m = planet_radius_rjup * _RJUP_M

    # Reflected light semi-amplitude
    a_ref_ppm = 0.5 * geometric_albedo * (rp_m / a_m)**2 * 1e6

    # Equilibrium temperature
    t_eq = stellar_teff_k * (rs_m / (2.0 * a_m)) ** 0.5
    f = heat_redistribution
    t_day = t_eq * (2.0 - 2.0 * f) ** 0.25 if f < 1.0 else t_eq
    t_night = t_eq * (2.0 * f) ** 0.25 if f > 0.0 else 0.0

    # Thermal contrast in stellar bandpass (optical; approximate as Rayleigh-Jeans)
    # More precisely: B_nu ~ T, so ratio ~ T_day/T★ - T_night/T★
    rp_rs_sq = (rp_m / rs_m)**2
    therm_amp_ppm = rp_rs_sq * abs(t_day - t_night) / stellar_teff_k * 1e6

    total_ppm = math.sqrt(a_ref_ppm**2 + therm_amp_ppm**2)

    sec_eclipse_ppm = (geometric_albedo * (rp_m / a_m)**2 +
                       rp_rs_sq * t_day / stellar_teff_k) * 1e6

    # Hot-spot offset: varies with heat redistribution (Showman & Guillot 2002)
    # For tidally locked: offset ~ 0–20° east, parameterize by redistribution
    phase_offset = (1.0 - heat_redistribution) * 20.0  # degrees east

    return PhaseCurveResult(
        reflected_amplitude_ppm=a_ref_ppm,
        thermal_amplitude_ppm=therm_amp_ppm,
        total_amplitude_ppm=total_ppm,
        secondary_eclipse_depth_ppm=sec_eclipse_ppm,
        phase_offset_deg=phase_offset,
        flag="OK",
    )


def format_phase_curve_result(r: PhaseCurveResult) -> str:
    if r.flag != "OK":
        return f"PhaseCurve | flag={r.flag}"
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Reflected amplitude | {r.reflected_amplitude_ppm:.2f} ppm |\n"
        f"| Thermal amplitude | {r.thermal_amplitude_ppm:.2f} ppm |\n"
        f"| Total amplitude | {r.total_amplitude_ppm:.2f} ppm |\n"
        f"| Secondary eclipse depth | {r.secondary_eclipse_depth_ppm:.2f} ppm |\n"
        f"| Hot-spot offset | {r.phase_offset_deg:.1f} ° |\n"
        f"| Flag | {r.flag} |"
    )


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Phase curve amplitude predictor")
    p.add_argument("period_days", type=float)
    p.add_argument("stellar_teff_k", type=float)
    p.add_argument("stellar_radius_rsun", type=float)
    p.add_argument("planet_radius_rjup", type=float)
    p.add_argument("--albedo", type=float, default=0.1)
    p.add_argument("--redistribution", type=float, default=0.5)
    args = p.parse_args()
    r = predict_phase_curve(args.period_days, args.stellar_teff_k,
                             args.stellar_radius_rsun, args.planet_radius_rjup,
                             geometric_albedo=args.albedo,
                             heat_redistribution=args.redistribution)
    print(format_phase_curve_result(r))


if __name__ == "__main__":
    _cli()
