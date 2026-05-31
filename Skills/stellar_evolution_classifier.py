"""Classify stellar evolution stage from Teff, log g, and luminosity."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class StellarEvolutionResult:
    teff_k: float
    logg: float
    luminosity_lsun: float | None
    stage: str
    spectral_class: str
    flag: str


def classify_evolution_stage(
    teff_k: float,
    logg: float,
    luminosity_lsun: float | None = None,
) -> StellarEvolutionResult:
    """Classify star as MS/subgiant/giant/supergiant/white_dwarf using Teff + log g."""
    if not math.isfinite(teff_k) or teff_k <= 0.0:
        return StellarEvolutionResult(
            teff_k=teff_k, logg=logg, luminosity_lsun=luminosity_lsun,
            stage="UNKNOWN", spectral_class="UNKNOWN", flag="INVALID_TEFF",
        )
    if not math.isfinite(logg):
        return StellarEvolutionResult(
            teff_k=teff_k, logg=logg, luminosity_lsun=luminosity_lsun,
            stage="UNKNOWN", spectral_class="UNKNOWN", flag="INVALID_LOGG",
        )

    # Spectral class from Teff
    if teff_k >= 30000:
        sp = "O"
    elif teff_k >= 10000:
        sp = "B"
    elif teff_k >= 7500:
        sp = "A"
    elif teff_k >= 6000:
        sp = "F"
    elif teff_k >= 5200:
        sp = "G"
    elif teff_k >= 3700:
        sp = "K"
    else:
        sp = "M"

    # Evolution stage from log g
    if logg >= 7.0:
        stage = "white_dwarf"
    elif logg >= 4.2:
        stage = "main_sequence"
    elif logg >= 3.5:
        stage = "subgiant"
    elif logg >= 2.0:
        stage = "giant"
    elif logg >= 0.0:
        stage = "supergiant"
    else:
        stage = "UNKNOWN"

    # Luminosity cross-check if provided
    flag = "OK"
    lum_ok = (
        luminosity_lsun is not None
        and math.isfinite(luminosity_lsun)
        and luminosity_lsun > 0.0
    )
    ms_too_bright = (
        stage == "main_sequence"
        and luminosity_lsun is not None
        and luminosity_lsun > 1000.0
    )
    giant_too_dim = (
        stage == "giant" and luminosity_lsun is not None and luminosity_lsun < 1.0
    )
    if lum_ok and (ms_too_bright or giant_too_dim):
        flag = "LUMINOSITY_INCONSISTENT"

    return StellarEvolutionResult(
        teff_k=teff_k,
        logg=logg,
        luminosity_lsun=luminosity_lsun,
        stage=stage,
        spectral_class=sp,
        flag=flag,
    )


def format_evolution_result(r: StellarEvolutionResult) -> str:
    lum_str = f"{r.luminosity_lsun:.3f}" if r.luminosity_lsun is not None else "N/A"
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Teff (K) | {r.teff_k:.0f} |\n"
        f"| log g | {r.logg:.2f} |\n"
        f"| Luminosity (L☉) | {lum_str} |\n"
        f"| Spectral class | {r.spectral_class} |\n"
        f"| Evolution stage | {r.stage} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Classify stellar evolution stage.")
    p.add_argument("teff_k", type=float)
    p.add_argument("logg", type=float)
    p.add_argument("--luminosity", type=float, default=None)
    args = p.parse_args()
    r = classify_evolution_stage(args.teff_k, args.logg, args.luminosity)
    print(format_evolution_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
