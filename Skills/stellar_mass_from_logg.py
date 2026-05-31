"""Estimate stellar mass from surface gravity and radius."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

_G_CGS = 6.674e-8  # cm^3 g^-1 s^-2
_RSUN_CM = 6.957e10
_MSUN_G = 1.989e33


@dataclass(frozen=True)
class StellarMassResult:
    logg: float
    radius_rsun: float
    mass_msun: float
    mass_msun_err: float | None
    log_mass: float
    flag: str


def compute_stellar_mass_from_logg(
    logg: float,
    radius_rsun: float,
    logg_err: float | None = None,
    radius_err_rsun: float | None = None,
) -> StellarMassResult:
    """Compute M★ = g·R²/G from log g (cgs) and radius."""
    if not math.isfinite(logg) or logg < 0.0 or logg > 6.0:
        return StellarMassResult(
            logg=logg, radius_rsun=radius_rsun,
            mass_msun=float("nan"), mass_msun_err=None,
            log_mass=float("nan"), flag="INVALID_LOGG",
        )
    if not math.isfinite(radius_rsun) or radius_rsun <= 0.0:
        return StellarMassResult(
            logg=logg, radius_rsun=radius_rsun,
            mass_msun=float("nan"), mass_msun_err=None,
            log_mass=float("nan"), flag="INVALID_RADIUS",
        )

    g_cgs = 10.0 ** logg
    r_cm = radius_rsun * _RSUN_CM
    mass_g = g_cgs * r_cm**2 / _G_CGS
    mass_msun = mass_g / _MSUN_G
    log_mass = math.log10(mass_msun)

    mass_err: float | None = None
    if logg_err is not None and radius_err_rsun is not None:
        dm_dlogg = mass_msun * math.log(10.0)
        dm_dr = 2.0 * mass_msun / radius_rsun
        mass_err = math.sqrt((dm_dlogg * logg_err) ** 2 + (dm_dr * radius_err_rsun) ** 2)

    flag = "OFF_MAIN_SEQUENCE" if mass_msun < 0.05 or mass_msun > 150.0 else "OK"

    return StellarMassResult(
        logg=logg,
        radius_rsun=radius_rsun,
        mass_msun=round(mass_msun, 6),
        mass_msun_err=round(mass_err, 6) if mass_err is not None else None,
        log_mass=round(log_mass, 4),
        flag=flag,
    )


def format_stellar_mass_result(r: StellarMassResult) -> str:
    err_str = f" ± {r.mass_msun_err:.4f}" if r.mass_msun_err is not None else ""
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| log g (cgs) | {r.logg:.3f} |\n"
        f"| Radius (R☉) | {r.radius_rsun:.4f} |\n"
        f"| Mass (M☉) | {r.mass_msun:.4f}{err_str} |\n"
        f"| log(M/M☉) | {r.log_mass:.4f} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Estimate stellar mass from log g and radius.")
    p.add_argument("logg", type=float, help="Surface gravity log g (cgs)")
    p.add_argument("radius_rsun", type=float, help="Stellar radius in solar radii")
    p.add_argument("--logg-err", type=float, default=None)
    p.add_argument("--radius-err", type=float, default=None)
    args = p.parse_args()
    r = compute_stellar_mass_from_logg(
        args.logg, args.radius_rsun, args.logg_err, args.radius_err
    )
    print(format_stellar_mass_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
