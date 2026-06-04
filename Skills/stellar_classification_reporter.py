"""Build a unified stellar classification report from Teff, logg, and optional inputs."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class StellarClassificationReport:
    teff_k: float
    logg: float
    spectral_class: str
    luminosity_class: str
    evolution_stage: str
    luminosity_lsun: float | None
    radius_rsun: float | None
    flag: str


def _spectral_class(teff: float) -> str:
    if teff >= 30000:
        return "O"
    if teff >= 10000:
        return "B"
    if teff >= 7500:
        return "A"
    if teff >= 6000:
        return "F"
    if teff >= 5200:
        return "G"
    if teff >= 3700:
        return "K"
    return "M"


def _evolution_stage(logg: float) -> str:
    if logg >= 7.0:
        return "white_dwarf"
    if logg >= 4.2:
        return "main_sequence"
    if logg >= 3.5:
        return "subgiant"
    if logg >= 2.0:
        return "giant"
    return "supergiant"


def _luminosity_class(logg: float) -> str:
    if logg >= 7.0:
        return "VII"
    if logg >= 4.2:
        return "V"
    if logg >= 3.5:
        return "IV"
    if logg >= 2.0:
        return "III"
    if logg >= 0.5:
        return "II"
    return "I"


def _luminosity_from_teff_logg(teff: float, logg: float) -> float:
    """Estimate L/L_sun via Stefan-Boltzmann: L ∝ R² T⁴, with R from logg."""
    # g = GM/R² → R ∝ M^0.5 / g^0.5; assume M~1 Msun for simplicity
    _G = 6.674e-11
    _MSUN = 1.989e30
    _RSUN = 6.957e8
    _LSUN = 3.828e26
    _SIGMA = 5.6704e-8
    _TSUN = 5778.0

    g_cgs = 10 ** (logg - 2.0)  # logg in cgs (cm/s²), convert to SI m/s²: g_si = g_cgs / 100
    g_si = g_cgs / 100.0
    r_m = math.sqrt(_G * _MSUN / g_si)
    lum = 4.0 * math.pi * r_m**2 * _SIGMA * teff**4
    return lum / _LSUN


def build_stellar_classification_report(
    teff_k: float,
    logg: float,
    radius_rsun: float | None = None,
) -> StellarClassificationReport:
    """
    Build unified stellar classification from Teff and log g.

    Derives spectral class, luminosity class, evolution stage,
    and estimated luminosity. Radius is accepted externally if known.
    """
    if not math.isfinite(teff_k) or teff_k <= 0.0:
        return StellarClassificationReport(
            teff_k=teff_k, logg=logg, spectral_class="", luminosity_class="",
            evolution_stage="", luminosity_lsun=None, radius_rsun=radius_rsun,
            flag="INVALID_TEFF",
        )
    if not math.isfinite(logg):
        return StellarClassificationReport(
            teff_k=teff_k, logg=logg, spectral_class="", luminosity_class="",
            evolution_stage="", luminosity_lsun=None, radius_rsun=radius_rsun,
            flag="INVALID_LOGG",
        )

    spec = _spectral_class(teff_k)
    lum_cls = _luminosity_class(logg)
    stage = _evolution_stage(logg)
    lum = _luminosity_from_teff_logg(teff_k, logg)

    return StellarClassificationReport(
        teff_k=teff_k,
        logg=logg,
        spectral_class=spec,
        luminosity_class=lum_cls,
        evolution_stage=stage,
        luminosity_lsun=round(lum, 3),
        radius_rsun=radius_rsun,
        flag="OK",
    )


def format_classification_report(r: StellarClassificationReport) -> str:
    rad_str = f"{r.radius_rsun:.3f}" if r.radius_rsun is not None else "N/A"
    lum_str = f"{r.luminosity_lsun:.3f}" if r.luminosity_lsun is not None else "N/A"
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Teff (K) | {r.teff_k:.0f} |\n"
        f"| log g | {r.logg:.2f} |\n"
        f"| Spectral class | {r.spectral_class} |\n"
        f"| Luminosity class | {r.luminosity_class} |\n"
        f"| Evolution stage | {r.evolution_stage} |\n"
        f"| Luminosity (L☉) | {lum_str} |\n"
        f"| Radius (R☉) | {rad_str} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Build stellar classification report.")
    p.add_argument("teff_k", type=float)
    p.add_argument("logg", type=float)
    p.add_argument("--radius-rsun", type=float, default=None)
    args = p.parse_args()
    r = build_stellar_classification_report(args.teff_k, args.logg, args.radius_rsun)
    print(format_classification_report(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
