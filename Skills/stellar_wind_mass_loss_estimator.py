"""Estimate stellar wind mass loss rate from rotation and X-ray luminosity."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

# Wood et al. (2005) astrospheric Ly-alpha detection scaling:
# Mdot / Mdot_sun ~ (Lx / Lx_sun)^1.34  for F-K dwarfs with Lx/Lsun < 10^-3.8
# Mdot_sun = 2e-14 Msun/yr
# Cranmer & Saar (2011) rotation-based scaling:
# Mdot ~ Prot^(-1.33) * Mstar^0.5  (simplified, solar-calibrated)

_MDOT_SUN_MSUN_PER_YR = 2.0e-14    # Msun/yr
_LX_SUN_ERGS = 2.0e27              # erg/s (solar mean cycle X-ray)
_LSUN_ERGS = 3.828e33              # erg/s


@dataclass(frozen=True)
class StellarWindResult:
    mass_loss_msun_per_yr: float
    mass_loss_relative_to_sun: float
    method: str   # XRAY / ROTATION / DEFAULT
    reliable: bool
    flag: str


def estimate_mass_loss_rate(
    prot_days: float | None = None,
    lx_ergs: float | None = None,
    mass_msun: float = 1.0,
    teff_k: float | None = None,
) -> StellarWindResult:
    """
    Estimate stellar wind mass loss rate.

    Methods (in order of preference):
    1. X-ray luminosity: Wood et al. (2005) Lx scaling (valid Lx/Lsun < 10^-3.8)
    2. Rotation period: Cranmer & Saar (2011) Prot scaling
    3. Fallback: constant solar-scaled estimate

    Returns mass loss in Msun/yr and relative to solar.
    reliable = False for extrapolations outside calibration range.
    """
    if not math.isfinite(mass_msun) or mass_msun <= 0.0:
        return StellarWindResult(
            mass_loss_msun_per_yr=float("nan"), mass_loss_relative_to_sun=float("nan"),
            method="INVALID", reliable=False, flag="INVALID_MASS",
        )

    method = "DEFAULT"
    reliable = False
    mdot = _MDOT_SUN_MSUN_PER_YR * mass_msun  # fallback

    if lx_ergs is not None and math.isfinite(lx_ergs) and lx_ergs > 0:
        lx_ratio = lx_ergs / _LX_SUN_ERGS
        # Wood et al. (2005) valid for log(Lx/Lsun) < -3.8 (log Lx < 29.6 erg/s)
        log_lx_lsun = math.log10(lx_ergs / _LSUN_ERGS)
        if log_lx_lsun < -3.8:
            mdot = _MDOT_SUN_MSUN_PER_YR * (lx_ratio ** 1.34) * mass_msun ** 0.5
            method = "XRAY"
            reliable = True
        else:
            # Saturated regime — Wood et al. relation breaks down
            mdot = _MDOT_SUN_MSUN_PER_YR * (10 ** (-3.8 / 1.34)) ** 1.34 * mass_msun ** 0.5
            method = "XRAY"
            reliable = False

    elif prot_days is not None and math.isfinite(prot_days) and prot_days > 0:
        # Cranmer & Saar (2011) simplified: Mdot ~ (25/Prot)^1.33 * Mstar^0.5 * Mdot_sun
        mdot = _MDOT_SUN_MSUN_PER_YR * (25.0 / prot_days) ** 1.33 * mass_msun ** 0.5
        method = "ROTATION"
        reliable = 5.0 <= prot_days <= 50.0

    relative = mdot / _MDOT_SUN_MSUN_PER_YR

    return StellarWindResult(
        mass_loss_msun_per_yr=mdot,
        mass_loss_relative_to_sun=round(relative, 3),
        method=method,
        reliable=reliable,
        flag="OK",
    )


def format_wind_result(r: StellarWindResult) -> str:
    mdot_str = f"{r.mass_loss_msun_per_yr:.3e}" if math.isfinite(r.mass_loss_msun_per_yr) else "N/A"
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Mass loss (Msun/yr) | {mdot_str} |\n"
        f"| Relative to solar | {r.mass_loss_relative_to_sun:.3f} |\n"
        f"| Method | {r.method} |\n"
        f"| Reliable | {r.reliable} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Estimate stellar wind mass loss rate.")
    p.add_argument("--prot-days", type=float, default=None)
    p.add_argument("--lx-ergs", type=float, default=None)
    p.add_argument("--mass-msun", type=float, default=1.0)
    p.add_argument("--teff-k", type=float, default=None)
    args = p.parse_args()
    r = estimate_mass_loss_rate(args.prot_days, args.lx_ergs, args.mass_msun, args.teff_k)
    print(format_wind_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
