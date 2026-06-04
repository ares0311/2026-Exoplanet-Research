"""Classify atmospheric retention via Jeans escape parameter."""
from __future__ import annotations

import math
from dataclasses import dataclass

_K_B = 1.380649e-23    # Boltzmann constant J/K
_G = 6.674e-11          # gravitational constant
_REARTH_M = 6.371e6
_MEARTH_KG = 5.972e24
_AMU_KG = 1.6605390666e-27


@dataclass(frozen=True)
class JeansEscapeResult:
    surface_gravity_ms2: float
    thermal_velocity_ms: float
    jeans_parameter: float
    retention_class: str   # STABLE / MARGINAL / RAPID_ESCAPE
    mass_loss_regime: str  # JEANS / HYDRODYNAMIC / NONE
    flag: str


def compute_jeans_escape(
    planet_mass_mearth: float,
    planet_radius_rearth: float,
    atmosphere_temp_k: float,
    mean_molecular_weight_amu: float = 2.0,
) -> JeansEscapeResult:
    """Compute Jeans escape parameter Λ = G Mp μ / (k_B T Rp).

    Λ > 20: atmosphere stable against Jeans escape
    6 < Λ ≤ 20: marginal; slow hydrodynamic winds possible
    Λ ≤ 6: rapid escape; atmosphere unlikely to be retained

    Args:
        planet_mass_mearth: planet mass (Earth masses)
        planet_radius_rearth: planet radius (Earth radii)
        atmosphere_temp_k: upper atmosphere temperature (K); ~Teq for simple estimate
        mean_molecular_weight_amu: mean molecular weight (2=H2, 4=He, 18=H2O, 28=N2/CO, 44=CO2)
    """
    if planet_mass_mearth <= 0.0:
        return JeansEscapeResult(float("nan"), float("nan"), float("nan"),
                                  "UNKNOWN", "UNKNOWN", "INVALID_MASS")
    if planet_radius_rearth <= 0.0:
        return JeansEscapeResult(float("nan"), float("nan"), float("nan"),
                                  "UNKNOWN", "UNKNOWN", "INVALID_RADIUS")
    if atmosphere_temp_k <= 0.0:
        return JeansEscapeResult(float("nan"), float("nan"), float("nan"),
                                  "UNKNOWN", "UNKNOWN", "INVALID_TEMP")
    if mean_molecular_weight_amu <= 0.0:
        return JeansEscapeResult(float("nan"), float("nan"), float("nan"),
                                  "UNKNOWN", "UNKNOWN", "INVALID_MEAN_MOLECULAR_WEIGHT")

    mp_kg = planet_mass_mearth * _MEARTH_KG
    rp_m = planet_radius_rearth * _REARTH_M
    mu_kg = mean_molecular_weight_amu * _AMU_KG

    g_surface = _G * mp_kg / rp_m**2
    v_thermal = math.sqrt(2.0 * _K_B * atmosphere_temp_k / mu_kg)
    lambda_j = _G * mp_kg * mu_kg / (_K_B * atmosphere_temp_k * rp_m)

    if lambda_j > 20.0:
        retention = "STABLE"
        regime = "NONE"
    elif lambda_j > 6.0:
        retention = "MARGINAL"
        regime = "JEANS"
    else:
        retention = "RAPID_ESCAPE"
        regime = "HYDRODYNAMIC"

    return JeansEscapeResult(
        surface_gravity_ms2=g_surface,
        thermal_velocity_ms=v_thermal,
        jeans_parameter=lambda_j,
        retention_class=retention,
        mass_loss_regime=regime,
        flag="OK",
    )


def format_jeans_escape_result(r: JeansEscapeResult) -> str:
    if r.flag != "OK":
        return f"JeansEscape | flag={r.flag}"
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Surface gravity | {r.surface_gravity_ms2:.2f} m/s² |\n"
        f"| Thermal velocity | {r.thermal_velocity_ms:.0f} m/s |\n"
        f"| Jeans parameter (Λ) | {r.jeans_parameter:.2f} |\n"
        f"| Retention class | {r.retention_class} |\n"
        f"| Mass-loss regime | {r.mass_loss_regime} |\n"
        f"| Flag | {r.flag} |"
    )


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Jeans escape classifier")
    p.add_argument("mass_mearth", type=float, help="Planet mass (Earth masses)")
    p.add_argument("radius_rearth", type=float, help="Planet radius (Earth radii)")
    p.add_argument("temp_k", type=float, help="Upper atmosphere temperature (K)")
    p.add_argument("--mu", type=float, default=2.0, help="Mean molecular weight (amu)")
    args = p.parse_args()
    r = compute_jeans_escape(args.mass_mearth, args.radius_rearth, args.temp_k,
                              mean_molecular_weight_amu=args.mu)
    print(format_jeans_escape_result(r))


if __name__ == "__main__":
    _cli()
