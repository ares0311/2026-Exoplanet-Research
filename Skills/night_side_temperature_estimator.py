"""Estimate night-side temperature from phase curve amplitude and heat redistribution."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NightSideTemperatureResult:
    day_side_temp_k: float
    night_side_temp_k: float
    heat_redistribution_efficiency: float
    day_night_contrast_k: float
    day_night_flux_ratio: float
    circulation_class: str   # EFFICIENT / MODERATE / POOR / VERY_POOR
    flag: str


def compute_night_side_temperature(
    equilibrium_temp_k: float,
    phase_curve_amplitude_ppm: float,
    planet_radius_rjup: float,
    stellar_radius_rsun: float,
) -> NightSideTemperatureResult:
    """Estimate night-side temperature from thermal phase curve amplitude.

    Day-night contrast observed as phase curve amplitude A:
      A ≈ (Rp/Rs)² × (F_day - F_night) / F_star

    With Stefan-Boltzmann: F ∝ T⁴:
      T_day = Teq × (1 + f)^(1/4), T_night = Teq × (1 - f)^(1/4)
    where f is solved from A = (Rp/Rs)² × (T_day⁴ - T_night⁴) / T_star⁴.

    For a practical approach, we use the phase amplitude directly:
      ΔT⁴ = T_day⁴ - T_night⁴ ∝ A / (Rp/Rs)²
      T_day = Teq × 2^(1/4) [no redistribution upper bound]
      T_night estimated from energy balance.

    Args:
        equilibrium_temp_k: equilibrium temperature (K)
        phase_curve_amplitude_ppm: peak-to-trough thermal phase curve amplitude (ppm)
        planet_radius_rjup: planet radius (Jupiter radii)
        stellar_radius_rsun: stellar radius (solar radii)
    """
    _RJUP_M = 7.1492e7
    _RSUN_M = 6.957e8

    if equilibrium_temp_k <= 0.0:
        return NightSideTemperatureResult(float("nan"), float("nan"), float("nan"),
                                           float("nan"), float("nan"), "UNKNOWN", "INVALID_TEQ")
    if phase_curve_amplitude_ppm < 0.0:
        return NightSideTemperatureResult(float("nan"), float("nan"), float("nan"),
                                           float("nan"), float("nan"), "UNKNOWN",
                                           "INVALID_AMPLITUDE")
    if planet_radius_rjup <= 0.0 or stellar_radius_rsun <= 0.0:
        return NightSideTemperatureResult(float("nan"), float("nan"), float("nan"),
                                           float("nan"), float("nan"), "UNKNOWN",
                                           "INVALID_RADII")

    rp_rs = (planet_radius_rjup * _RJUP_M) / (stellar_radius_rsun * _RSUN_M)
    depth = rp_rs**2

    # ΔT⁴ from amplitude: A = depth × (T_day⁴ - T_night⁴) / T_star⁴
    # But T_star cancels if we work in units of Teq
    # Use: A_thermal = depth × (T_day⁴ - T_night⁴) / T_eq⁴ × (T_eq/T_star)⁴
    # Simplified: treat Teq as proxy; derive contrast
    a_frac = phase_curve_amplitude_ppm * 1e-6 / depth if depth > 0 else 0.0
    delta_t4_norm = a_frac * equilibrium_temp_k**4

    # T_day: assume T_day⁴ = T_eq⁴ × (1 + x), T_night⁴ = T_eq⁴ × (1 - x)
    # δT⁴ = 2x × T_eq⁴ → x = δT⁴ / (2 T_eq⁴)
    x = min(delta_t4_norm / (2.0 * equilibrium_temp_k**4), 1.0)
    x = max(x, 0.0)

    t_day = equilibrium_temp_k * (1.0 + x) ** 0.25
    t_night = equilibrium_temp_k * max(1.0 - x, 0.0) ** 0.25
    contrast_k = t_day - t_night
    flux_ratio = (t_day / t_night) ** 4 if t_night > 0 else float("inf")

    efficiency = 1.0 - x
    if efficiency > 0.85:
        circ_class = "EFFICIENT"
    elif efficiency > 0.60:
        circ_class = "MODERATE"
    elif efficiency > 0.30:
        circ_class = "POOR"
    else:
        circ_class = "VERY_POOR"

    return NightSideTemperatureResult(
        day_side_temp_k=t_day,
        night_side_temp_k=t_night,
        heat_redistribution_efficiency=efficiency,
        day_night_contrast_k=contrast_k,
        day_night_flux_ratio=flux_ratio,
        circulation_class=circ_class,
        flag="OK",
    )


def format_night_side_temperature_result(r: NightSideTemperatureResult) -> str:
    if r.flag != "OK":
        return f"NightSideTemperature | flag={r.flag}"
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Day-side temperature | {r.day_side_temp_k:.0f} K |\n"
        f"| Night-side temperature | {r.night_side_temp_k:.0f} K |\n"
        f"| Day-night contrast | {r.day_night_contrast_k:.0f} K |\n"
        f"| Day/night flux ratio | {r.day_night_flux_ratio:.2f} |\n"
        f"| Heat redistribution efficiency | {r.heat_redistribution_efficiency:.3f} |\n"
        f"| Circulation class | {r.circulation_class} |\n"
        f"| Flag | {r.flag} |"
    )


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Night-side temperature estimator")
    p.add_argument("teq_k", type=float)
    p.add_argument("amplitude_ppm", type=float)
    p.add_argument("rp_rjup", type=float)
    p.add_argument("rs_rsun", type=float)
    args = p.parse_args()
    r = compute_night_side_temperature(args.teq_k, args.amplitude_ppm,
                                        args.rp_rjup, args.rs_rsun)
    print(format_night_side_temperature_result(r))


if __name__ == "__main__":
    _cli()
