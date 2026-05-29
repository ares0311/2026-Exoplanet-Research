"""Compute orbital distance from stellar density and period.

Given stellar density rho (g/cm^3) and orbital period P (days):
    (a/R*)^3 = G * rho * P^2 / (3 * pi)

where G = 6.674e-8 cm^3 g^-1 s^-2, P converted to seconds.

Then a_au = a_rstar * rstar_rsun * R_SUN_AU

Public API
----------
OrbitalPeriodResult(a_rstar, a_au, period_days, flag)
compute_orbital_distance(rho_gcc, period_days, rstar_rsun) -> OrbitalPeriodResult
format_orbital_period(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass

_G_CGS = 6.674e-8        # cm^3 g^-1 s^-2
_SEC_PER_DAY = 86400.0
_R_SUN_AU = 0.00465047   # 1 solar radius in AU


@dataclass(frozen=True)
class OrbitalPeriodResult:
    a_rstar: float      # semi-major axis in stellar radii
    a_au: float         # semi-major axis in AU
    period_days: float  # period used
    flag: str           # "OK" or "LONG_PERIOD"


def compute_orbital_distance(
    rho_gcc: float,
    period_days: float,
    rstar_rsun: float = 1.0,
) -> OrbitalPeriodResult:
    """Compute semi-major axis from stellar density and orbital period.

    Args:
        rho_gcc: Mean stellar density in g/cm^3.
        period_days: Orbital period in days.
        rstar_rsun: Stellar radius in solar radii (default 1.0).

    Returns:
        :class:`OrbitalPeriodResult`.
    """
    if rho_gcc <= 0 or period_days <= 0 or rstar_rsun <= 0:
        return OrbitalPeriodResult(
            a_rstar=0.0, a_au=0.0, period_days=period_days, flag="ERROR",
        )

    p_sec = period_days * _SEC_PER_DAY
    # (a/R*)^3 = G * rho * P^2 / (3*pi)
    a_rstar_cubed = _G_CGS * rho_gcc * p_sec**2 / (3.0 * math.pi)
    a_rstar = a_rstar_cubed ** (1.0 / 3.0)
    a_au = a_rstar * float(rstar_rsun) * _R_SUN_AU

    flag = "LONG_PERIOD" if period_days > 365.0 else "OK"

    return OrbitalPeriodResult(
        a_rstar=round(a_rstar, 4),
        a_au=round(a_au, 6),
        period_days=period_days,
        flag=flag,
    )


def format_orbital_period(result: OrbitalPeriodResult) -> str:
    """Format orbital distance result as Markdown."""
    lines = [
        "## Orbital Distance from Stellar Density",
        "",
        f"- Period: {result.period_days:.3f} days",
        f"- a/R★: **{result.a_rstar:.3f}**",
        f"- a: **{result.a_au:.5f} AU**",
        f"- Flag: {result.flag}",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="orbital_period_from_density",
        description=__doc__,
    )
    p.add_argument("rho_gcc", type=float, help="Stellar density in g/cm^3")
    p.add_argument("period_days", type=float, help="Orbital period in days")
    p.add_argument("--rstar-rsun", type=float, default=1.0, help="Stellar radius in R_sun")
    args = p.parse_args(argv)
    r = compute_orbital_distance(args.rho_gcc, args.period_days, args.rstar_rsun)
    print(format_orbital_period(r))
    return 0 if r.flag in ("OK", "LONG_PERIOD") else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
