"""Estimate planetary Bond albedo from equilibrium temperature and stellar irradiation."""
from __future__ import annotations

import math
from dataclasses import dataclass

_SIGMA = 5.670374419e-8  # Stefan-Boltzmann constant W m^-2 K^-4
_AU_M = 1.495978707e11   # AU in metres
_RSUN_M = 6.957e8        # Solar radius in metres


@dataclass(frozen=True)
class BondAlbedoResult:
    stellar_flux_wm2: float
    equilibrium_temp_k: float
    bond_albedo: float
    geometric_albedo_estimate: float
    irradiation_class: str
    flag: str


def compute_bond_albedo(
    equilibrium_temp_k: float,
    orbital_distance_au: float,
    stellar_teff_k: float = 5778.0,
    stellar_radius_rsun: float = 1.0,
    heat_redistribution_f: float = 0.25,
) -> BondAlbedoResult:
    """Estimate Bond albedo from equilibrium temperature.

    T_eq = Teff * sqrt(R*/2a) * (1 - AB)^0.25 * f^0.25
    → AB = 1 - (T_eq / (Teff * sqrt(R*/2a) * f^0.25))^4

    Args:
        equilibrium_temp_k: observed/derived equilibrium temperature (K)
        orbital_distance_au: semi-major axis (AU)
        stellar_teff_k: stellar effective temperature (K)
        stellar_radius_rsun: stellar radius (solar radii)
        heat_redistribution_f: heat redistribution factor (0.25 = uniform, 0.667 = instant re-rad)
    """
    if equilibrium_temp_k <= 0.0:
        return BondAlbedoResult(
            stellar_flux_wm2=float("nan"),
            equilibrium_temp_k=equilibrium_temp_k,
            bond_albedo=float("nan"),
            geometric_albedo_estimate=float("nan"),
            irradiation_class="UNKNOWN",
            flag="INVALID_TEQ",
        )
    if orbital_distance_au <= 0.0:
        return BondAlbedoResult(
            stellar_flux_wm2=float("nan"),
            equilibrium_temp_k=equilibrium_temp_k,
            bond_albedo=float("nan"),
            geometric_albedo_estimate=float("nan"),
            irradiation_class="UNKNOWN",
            flag="INVALID_DISTANCE",
        )
    if stellar_teff_k <= 0.0 or stellar_radius_rsun <= 0.0:
        return BondAlbedoResult(
            stellar_flux_wm2=float("nan"),
            equilibrium_temp_k=equilibrium_temp_k,
            bond_albedo=float("nan"),
            geometric_albedo_estimate=float("nan"),
            irradiation_class="UNKNOWN",
            flag="INVALID_STELLAR_PARAMS",
        )

    r_star_m = stellar_radius_rsun * _RSUN_M
    a_m = orbital_distance_au * _AU_M
    stellar_flux = _SIGMA * stellar_teff_k**4 * (r_star_m / a_m) ** 2

    # AB = 1 - (T_eq / (T_eq_zero_albedo))^4 where T_eq_zero_albedo uses AB=0
    t_eq_zero = stellar_teff_k * math.sqrt(r_star_m / (2.0 * a_m)) * heat_redistribution_f**0.25
    if t_eq_zero <= 0.0:
        return BondAlbedoResult(
            stellar_flux_wm2=stellar_flux,
            equilibrium_temp_k=equilibrium_temp_k,
            bond_albedo=float("nan"),
            geometric_albedo_estimate=float("nan"),
            irradiation_class="UNKNOWN",
            flag="NUMERICAL_ERROR",
        )

    ab = 1.0 - (equilibrium_temp_k / t_eq_zero) ** 4
    ab = max(0.0, min(0.999, ab))
    # Rowe+2006: Ag ≈ 3/2 * AB for Lambertian sphere
    ag_estimate = min(1.0, 1.5 * ab)

    flux_earth = stellar_flux / 1361.0
    if flux_earth >= 100.0:
        irr_class = "ULTRA_HOT"
    elif flux_earth >= 10.0:
        irr_class = "HOT"
    elif flux_earth >= 1.0:
        irr_class = "WARM"
    else:
        irr_class = "COLD"

    return BondAlbedoResult(
        stellar_flux_wm2=stellar_flux,
        equilibrium_temp_k=equilibrium_temp_k,
        bond_albedo=ab,
        geometric_albedo_estimate=ag_estimate,
        irradiation_class=irr_class,
        flag="OK",
    )


def format_bond_albedo_result(r: BondAlbedoResult) -> str:
    if r.flag != "OK":
        return f"BondAlbedo | flag={r.flag}"
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Stellar flux at planet | {r.stellar_flux_wm2:.2e} W/m² |\n"
        f"| Equilibrium temperature | {r.equilibrium_temp_k:.0f} K |\n"
        f"| Bond albedo | {r.bond_albedo:.3f} |\n"
        f"| Geometric albedo (est.) | {r.geometric_albedo_estimate:.3f} |\n"
        f"| Irradiation class | {r.irradiation_class} |\n"
        f"| Flag | {r.flag} |"
    )


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Planet Bond albedo estimator")
    p.add_argument("teq", type=float, help="Equilibrium temperature (K)")
    p.add_argument("a_au", type=float, help="Orbital distance (AU)")
    p.add_argument("--teff", type=float, default=5778.0, help="Stellar Teff (K)")
    p.add_argument("--rstar", type=float, default=1.0, help="Stellar radius (Rsun)")
    args = p.parse_args()
    r = compute_bond_albedo(args.teq, args.a_au, stellar_teff_k=args.teff,
                            stellar_radius_rsun=args.rstar)
    print(format_bond_albedo_result(r))


if __name__ == "__main__":
    _cli()
