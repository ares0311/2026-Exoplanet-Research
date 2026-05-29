"""Format stellar properties into a structured Markdown report.

Combines TIC parameters with derived quantities (luminosity, habitable zone,
spectral classification) for a given target.

Public API
----------
StellarReport(tic_id, teff_k, radius_rsun, mass_msun, logg, metallicity_dex,
              luminosity_lsun, spectral_type, hz_inner_au, hz_outer_au, flag)
build_stellar_report(tic_id, *, teff_k, radius_rsun, mass_msun,
                     logg, metallicity_dex) -> StellarReport
format_stellar_report(report) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class StellarReport:
    tic_id: int
    teff_k: float | None
    radius_rsun: float | None
    mass_msun: float | None
    logg: float | None
    metallicity_dex: float | None
    luminosity_lsun: float | None
    spectral_type: str | None
    hz_inner_au: float | None
    hz_outer_au: float | None
    flag: str  # "OK" | "INCOMPLETE" | "INVALID"


def _luminosity(teff_k: float, radius_rsun: float) -> float:
    """Stefan-Boltzmann luminosity in L_sun units."""
    teff_sun = 5778.0
    return (radius_rsun ** 2) * ((teff_k / teff_sun) ** 4)


def _spectral_type(teff_k: float) -> str:
    if teff_k >= 30000:
        return "O"
    if teff_k >= 10000:
        return "B"
    if teff_k >= 7500:
        return "A"
    if teff_k >= 6000:
        return "F"
    if teff_k >= 5200:
        return "G"
    if teff_k >= 3700:
        return "K"
    return "M"


def _hz_boundaries(luminosity_lsun: float) -> tuple[float, float]:
    """Conservative HZ boundaries (Kopparapu 2013 simplified)."""
    hz_inner = math.sqrt(luminosity_lsun / 1.1)
    hz_outer = math.sqrt(luminosity_lsun / 0.356)
    return round(hz_inner, 4), round(hz_outer, 4)


def build_stellar_report(
    tic_id: int,
    *,
    teff_k: float | None = None,
    radius_rsun: float | None = None,
    mass_msun: float | None = None,
    logg: float | None = None,
    metallicity_dex: float | None = None,
) -> StellarReport:
    """Build a stellar properties report with derived quantities.

    Args:
        tic_id: TIC identifier.
        teff_k: Effective temperature in Kelvin.
        radius_rsun: Stellar radius in solar radii.
        mass_msun: Stellar mass in solar masses.
        logg: Surface gravity (log10 cgs).
        metallicity_dex: Metallicity [Fe/H] in dex.

    Returns:
        StellarReport with derived luminosity, spectral type, and HZ.
    """
    if teff_k is not None and teff_k <= 0:
        return StellarReport(
            tic_id=tic_id, teff_k=teff_k, radius_rsun=radius_rsun,
            mass_msun=mass_msun, logg=logg, metallicity_dex=metallicity_dex,
            luminosity_lsun=None, spectral_type=None,
            hz_inner_au=None, hz_outer_au=None, flag="INVALID",
        )

    lum: float | None = None
    hz_inner: float | None = None
    hz_outer: float | None = None
    sp_type: str | None = None

    if teff_k is not None and radius_rsun is not None and radius_rsun > 0:
        lum = round(_luminosity(teff_k, radius_rsun), 4)
        hz_inner, hz_outer = _hz_boundaries(lum)

    if teff_k is not None:
        sp_type = _spectral_type(teff_k)

    n_missing = sum(
        1 for v in [teff_k, radius_rsun, mass_msun] if v is None
    )
    flag = "INCOMPLETE" if n_missing > 0 else "OK"

    return StellarReport(
        tic_id=tic_id,
        teff_k=teff_k,
        radius_rsun=radius_rsun,
        mass_msun=mass_msun,
        logg=logg,
        metallicity_dex=metallicity_dex,
        luminosity_lsun=lum,
        spectral_type=sp_type,
        hz_inner_au=hz_inner,
        hz_outer_au=hz_outer,
        flag=flag,
    )


def format_stellar_report(report: StellarReport) -> str:
    """Format stellar properties report as Markdown.

    Args:
        report: StellarReport to format.

    Returns:
        Markdown string.
    """
    def _fmtf(v: float | None, fmt: str = ".3f", suffix: str = "") -> str:
        return f"{v:{fmt}}{suffix}" if v is not None else "—"

    lines = [
        f"## Stellar Properties — TIC {report.tic_id}\n",
        f"**Status**: `{report.flag}` | "
        f"Spectral type: {report.spectral_type or '—'}\n",
        "",
        "| Parameter | Value |",
        "|---|---|",
        f"| T_eff | {_fmtf(report.teff_k, '.0f', ' K')} |",
        f"| Radius | {_fmtf(report.radius_rsun, '.3f', ' R☉')} |",
        f"| Mass | {_fmtf(report.mass_msun, '.3f', ' M☉')} |",
        f"| log g | {_fmtf(report.logg, '.2f')} |",
        f"| [Fe/H] | {_fmtf(report.metallicity_dex, '.2f', ' dex')} |",
        f"| Luminosity | {_fmtf(report.luminosity_lsun, '.4f', ' L☉')} |",
        f"| HZ inner | {_fmtf(report.hz_inner_au, '.4f', ' AU')} |",
        f"| HZ outer | {_fmtf(report.hz_outer_au, '.4f', ' AU')} |",
    ]
    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Report stellar properties.")
    parser.add_argument("tic_id", type=int)
    parser.add_argument("--teff", type=float, default=None)
    parser.add_argument("--radius", type=float, default=None)
    parser.add_argument("--mass", type=float, default=None)
    parser.add_argument("--logg", type=float, default=None)
    parser.add_argument("--feh", type=float, default=None)
    args = parser.parse_args(argv)

    report = build_stellar_report(
        args.tic_id,
        teff_k=args.teff,
        radius_rsun=args.radius,
        mass_msun=args.mass,
        logg=args.logg,
        metallicity_dex=args.feh,
    )
    print(format_stellar_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
