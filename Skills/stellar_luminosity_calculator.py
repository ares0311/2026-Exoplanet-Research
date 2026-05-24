"""Compute stellar luminosity from effective temperature and radius.

Uses the Stefan-Boltzmann law to derive L/L_sun from Teff and R/R_sun.
Complements ``stellar_density_calculator`` and ``kopparapu_hz_calculator``
which both need luminosity but do not expose it as a standalone output.

Public API
----------
LuminosityResult(teff_k, radius_rsun, luminosity_lsun, luminosity_log10,
                 radius_au, flag)
compute_stellar_luminosity(teff_k, radius_rsun) -> LuminosityResult
format_luminosity_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass

_TEFF_SUN = 5778.0   # K
_RSUN_TO_AU = 0.00465047


@dataclass(frozen=True)
class LuminosityResult:
    teff_k: float
    radius_rsun: float
    luminosity_lsun: float | None
    luminosity_log10: float | None   # log10(L/L_sun)
    radius_au: float | None
    flag: str  # "OK" | "INVALID"


def compute_stellar_luminosity(
    teff_k: float,
    radius_rsun: float,
) -> LuminosityResult:
    """Compute stellar luminosity via the Stefan-Boltzmann law.

    L/L_sun = (R/R_sun)^2 × (Teff/Teff_sun)^4

    Args:
        teff_k: Effective temperature (K).
        radius_rsun: Stellar radius in solar radii.

    Returns:
        :class:`LuminosityResult`.
    """
    if not (math.isfinite(teff_k) and math.isfinite(radius_rsun)):
        return LuminosityResult(teff_k, radius_rsun, None, None, None, "INVALID")
    if teff_k <= 0 or radius_rsun <= 0:
        return LuminosityResult(teff_k, radius_rsun, None, None, None, "INVALID")

    lum = (radius_rsun ** 2) * ((teff_k / _TEFF_SUN) ** 4)
    log_lum = math.log10(lum)
    radius_au = radius_rsun * _RSUN_TO_AU

    return LuminosityResult(
        teff_k=teff_k,
        radius_rsun=radius_rsun,
        luminosity_lsun=round(lum, 6),
        luminosity_log10=round(log_lum, 6),
        radius_au=round(radius_au, 8),
        flag="OK",
    )


def format_luminosity_result(result: LuminosityResult) -> str:
    """Format luminosity result as Markdown."""
    lines = [
        "## Stellar Luminosity Calculator",
        "",
        f"- Teff: {result.teff_k} K",
        f"- Radius: {result.radius_rsun} R_sun",
        f"- **Luminosity: {result.luminosity_lsun} L_sun**",
        f"- log10(L/L_sun): {result.luminosity_log10}",
        f"- Radius: {result.radius_au} AU",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="stellar_luminosity_calculator",
        description="Compute stellar luminosity from Teff and radius.",
    )
    parser.add_argument("teff_k", type=float)
    parser.add_argument("radius_rsun", type=float)
    args = parser.parse_args(argv)

    result = compute_stellar_luminosity(args.teff_k, args.radius_rsun)
    print(format_luminosity_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
