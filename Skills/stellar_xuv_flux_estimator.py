"""Estimate stellar XUV flux at a planet using rotation-period or age scaling."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

_L_SUN_ERG_S = 3.828e33   # erg/s
_AU_CM = 1.496e13          # AU in cm


@dataclass(frozen=True)
class StellarXuvFluxResult:
    stellar_mass_msun: float
    rotation_period_days: float | None
    stellar_age_gyr: float | None
    lx_erg_s: float                # X-ray luminosity
    leuv_erg_s: float              # EUV luminosity (scaling from L_X)
    lxuv_erg_s: float              # combined XUV
    xuv_flux_at_planet_wm2: float  # XUV flux at specified orbital distance
    saturation_regime: bool
    flag: str


# Wright et al. (2011) Rossby-number power-law scaling
_LOG_RX_SAT = -3.13       # log10(Rx_sat) saturation value
_BETA_ACT = -2.70         # power-law slope in unsaturated regime
_PROT_SAT_DAYS = 1.5      # approximate saturation period (varies by spectral type)


def compute_stellar_xuv_flux(
    stellar_mass_msun: float,
    orbital_distance_au: float,
    rotation_period_days: float | None = None,
    stellar_age_gyr: float | None = None,
    stellar_lbol_lsun: float | None = None,
) -> StellarXuvFluxResult:
    """
    Estimate stellar XUV luminosity and flux at a planet.

    Uses Wright et al. (2011) X-ray / bolometric luminosity ratio scaling:
    - Saturation regime (fast rotators): log(Lx/Lbol) ≈ -3.13
    - Unsaturated regime: log(Lx/Lbol) ≈ -3.13 + β * log(Prot / P_sat)  with β ≈ -2.7
    EUV from Sanz-Forcada et al. (2011): log(L_EUV) ≈ log(L_X) * 0.860 + 4.80

    If rotation_period_days is None and stellar_age_gyr is given, Skumanich (1972)
    scaling is used: Prot ∝ age^0.5 (calibrated: Prot(1 Gyr) ≈ 10 d for solar-type).

    Parameters
    ----------
    stellar_mass_msun:      Stellar mass in solar masses.
    orbital_distance_au:    Orbital distance in AU (for flux calculation).
    rotation_period_days:   Stellar rotation period (optional).
    stellar_age_gyr:        Stellar age in Gyr (used if Prot not given).
    stellar_lbol_lsun:      Bolometric luminosity in solar units (optional; derived from
                            mass if None).
    """
    if not math.isfinite(stellar_mass_msun) or stellar_mass_msun <= 0:
        return StellarXuvFluxResult(
            stellar_mass_msun, rotation_period_days, stellar_age_gyr,
            float("nan"), float("nan"), float("nan"), float("nan"), False, "INVALID_MASS",
        )
    if not math.isfinite(orbital_distance_au) or orbital_distance_au <= 0:
        return StellarXuvFluxResult(
            stellar_mass_msun, rotation_period_days, stellar_age_gyr,
            float("nan"), float("nan"), float("nan"), float("nan"), False, "INVALID_DISTANCE",
        )

    if stellar_lbol_lsun is None:
        stellar_lbol_lsun = stellar_mass_msun ** 4.0

    if rotation_period_days is None:
        if stellar_age_gyr is not None and stellar_age_gyr > 0:
            rotation_period_days = 10.0 * math.sqrt(stellar_age_gyr)
        else:
            rotation_period_days = 25.4 * stellar_mass_msun ** (-0.5)

    saturation = rotation_period_days <= _PROT_SAT_DAYS
    if saturation:
        log_rx = _LOG_RX_SAT
    else:
        log_rx = _LOG_RX_SAT + _BETA_ACT * math.log10(rotation_period_days / _PROT_SAT_DAYS)

    lbol_erg_s = stellar_lbol_lsun * _L_SUN_ERG_S
    lx_erg_s = 10.0 ** log_rx * lbol_erg_s

    # Sanz-Forcada et al. (2011) EUV scaling
    log_leuv = 0.860 * math.log10(lx_erg_s) + 4.80
    leuv_erg_s = 10.0 ** log_leuv

    lxuv_erg_s = lx_erg_s + leuv_erg_s

    a_cm = orbital_distance_au * _AU_CM
    xuv_flux_erg_cm2_s = lxuv_erg_s / (4.0 * math.pi * a_cm ** 2)
    xuv_flux_wm2 = xuv_flux_erg_cm2_s * 1e-3  # erg/cm²/s → W/m²

    return StellarXuvFluxResult(
        stellar_mass_msun=stellar_mass_msun,
        rotation_period_days=rotation_period_days,
        stellar_age_gyr=stellar_age_gyr,
        lx_erg_s=round(lx_erg_s, 4),
        leuv_erg_s=round(leuv_erg_s, 4),
        lxuv_erg_s=round(lxuv_erg_s, 4),
        xuv_flux_at_planet_wm2=round(xuv_flux_wm2, 6),
        saturation_regime=saturation,
        flag="OK",
    )


def format_stellar_xuv_flux_result(r: StellarXuvFluxResult) -> str:
    def _f(v: float, fmt: str = ".4e") -> str:
        return format(v, fmt) if math.isfinite(v) else "N/A"

    prot = f"{r.rotation_period_days:.2f}" if r.rotation_period_days is not None else "N/A"
    return (
        f"| Parameter | Value |\n|---|---|\n"
        f"| Rotation period (days) | {prot} |\n"
        f"| L_X (erg/s) | {_f(r.lx_erg_s)} |\n"
        f"| L_EUV (erg/s) | {_f(r.leuv_erg_s)} |\n"
        f"| L_XUV (erg/s) | {_f(r.lxuv_erg_s)} |\n"
        f"| XUV flux at planet (W/m²) | {_f(r.xuv_flux_at_planet_wm2)} |\n"
        f"| Saturation regime | {r.saturation_regime} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Estimate stellar XUV flux at a planet.")
    p.add_argument("stellar_mass_msun", type=float)
    p.add_argument("orbital_distance_au", type=float)
    p.add_argument("--rotation-period-days", type=float, default=None)
    p.add_argument("--stellar-age-gyr", type=float, default=None)
    p.add_argument("--stellar-lbol-lsun", type=float, default=None)
    args = p.parse_args()
    r = compute_stellar_xuv_flux(
        args.stellar_mass_msun, args.orbital_distance_au,
        rotation_period_days=args.rotation_period_days,
        stellar_age_gyr=args.stellar_age_gyr,
        stellar_lbol_lsun=args.stellar_lbol_lsun,
    )
    print(format_stellar_xuv_flux_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
