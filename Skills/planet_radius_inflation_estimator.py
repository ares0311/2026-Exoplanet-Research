"""Estimate anomalous radius inflation of hot Jupiters via Ohmic dissipation."""
from __future__ import annotations

from dataclasses import dataclass

_G = 6.674e-11
_MJUP_KG = 1.898e27
_RJUP_M = 7.1492e7
_L_SUN = 3.828e26
_SEC_PER_GYR = 3.156e16


@dataclass(frozen=True)
class RadiusInflationResult:
    stellar_flux_wm2: float
    ohmic_power_fraction: float        # L_ohm / L_irr
    expected_radius_rjup: float        # Baraffe+2003 cooling-track radius
    inflation_rjup: float              # anomalous extra radius
    inflated_radius_rjup: float        # expected + inflation
    inflation_class: str               # NONE / MILD / MODERATE / SEVERE
    flag: str


_AU_M = 1.495978707e11
_RSUN_M = 6.957e8
_SIGMA = 5.67e-8


def compute_radius_inflation(
    planet_mass_mjup: float,
    planet_radius_rjup: float,
    orbital_distance_au: float,
    stellar_teff_k: float = 5778.0,
    stellar_radius_rsun: float = 1.0,
    planet_age_gyr: float = 5.0,
    ohmic_efficiency: float = 0.01,
) -> RadiusInflationResult:
    """Estimate hot Jupiter radius inflation via Ohmic dissipation.

    Batygin & Stevenson (2010): Ohmic power fraction P_ohm / P_irr drives inflation.
    Empirical Thorngren & Fortney (2018) relation:
      ΔR/R ≈ 0.35 × (F_irr / F_ref)^0.6  [F_ref = 10^9 erg/cm²/s = 10^5 W/m²]

    Expected cooling-track radius from Baraffe et al. (2003) approx:
      R_cool ≈ 1.0 × (Mp/Mjup)^(-0.04) [weak mass dependence for 0.5–10 Mjup]

    Args:
        planet_mass_mjup: planet mass (Jupiter masses)
        planet_radius_rjup: observed planet radius (Jupiter radii)
        orbital_distance_au: orbital distance (AU)
        stellar_teff_k: stellar effective temperature (K)
        stellar_radius_rsun: stellar radius (solar radii)
        planet_age_gyr: planet age (Gyr)
        ohmic_efficiency: fraction of irradiation going to Ohmic heating
    """
    if planet_mass_mjup <= 0.0:
        return RadiusInflationResult(float("nan"), float("nan"), float("nan"),
                                      float("nan"), float("nan"), "UNKNOWN", "INVALID_MASS")
    if planet_radius_rjup <= 0.0:
        return RadiusInflationResult(float("nan"), float("nan"), float("nan"),
                                      float("nan"), float("nan"), "UNKNOWN", "INVALID_RADIUS")
    if orbital_distance_au <= 0.0:
        return RadiusInflationResult(float("nan"), float("nan"), float("nan"),
                                      float("nan"), float("nan"), "UNKNOWN", "INVALID_DISTANCE")

    r_star_m = stellar_radius_rsun * _RSUN_M
    a_m = orbital_distance_au * _AU_M
    f_irr = _SIGMA * stellar_teff_k**4 * (r_star_m / a_m)**2  # W/m²

    f_ref = 1e5  # W/m² reference flux
    ohmic_frac = ohmic_efficiency

    delta_r_frac = (
        0.35 * (f_irr / f_ref) ** 0.6 * ohmic_efficiency / 0.01 if f_irr > f_ref else 0.0
    )

    r_cool = 1.0 * planet_mass_mjup ** (-0.04)
    delta_r = r_cool * delta_r_frac
    r_inflated = r_cool + delta_r

    if delta_r_frac < 0.02:
        infl_class = "NONE"
    elif delta_r_frac < 0.08:
        infl_class = "MILD"
    elif delta_r_frac < 0.20:
        infl_class = "MODERATE"
    else:
        infl_class = "SEVERE"

    return RadiusInflationResult(
        stellar_flux_wm2=f_irr,
        ohmic_power_fraction=ohmic_frac,
        expected_radius_rjup=r_cool,
        inflation_rjup=delta_r,
        inflated_radius_rjup=r_inflated,
        inflation_class=infl_class,
        flag="OK",
    )


def format_radius_inflation_result(r: RadiusInflationResult) -> str:
    if r.flag != "OK":
        return f"RadiusInflation | flag={r.flag}"
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Stellar irradiation flux | {r.stellar_flux_wm2:.2e} W/m² |\n"
        f"| Expected radius (cooling) | {r.expected_radius_rjup:.3f} Rjup |\n"
        f"| Ohmic inflation | {r.inflation_rjup:.3f} Rjup |\n"
        f"| Predicted inflated radius | {r.inflated_radius_rjup:.3f} Rjup |\n"
        f"| Inflation class | {r.inflation_class} |\n"
        f"| Flag | {r.flag} |"
    )


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Hot Jupiter radius inflation estimator")
    p.add_argument("mass_mjup", type=float)
    p.add_argument("radius_rjup", type=float)
    p.add_argument("distance_au", type=float)
    p.add_argument("--teff", type=float, default=5778.0)
    p.add_argument("--rstar", type=float, default=1.0)
    args = p.parse_args()
    r = compute_radius_inflation(args.mass_mjup, args.radius_rjup, args.distance_au,
                                  stellar_teff_k=args.teff, stellar_radius_rsun=args.rstar)
    print(format_radius_inflation_result(r))


if __name__ == "__main__":
    _cli()
