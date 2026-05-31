"""Calculate planetary insolation flux in Earth units."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class InsolationResult:
    luminosity_lsun: float
    sma_au: float
    insolation_earth: float
    hz_class: str
    flag: str


# Kopparapu (2013) conservative HZ boundaries for Sun-like star in S_earth
_HZ_INNER = 1.107  # runaway greenhouse
_HZ_OUTER = 0.356  # maximum greenhouse


def compute_insolation(
    luminosity_lsun: float,
    sma_au: float,
    teff_k: float = 5778.0,
) -> InsolationResult:
    """Compute incident flux S = L / a² in Earth units with HZ classification."""
    if not math.isfinite(luminosity_lsun) or luminosity_lsun <= 0.0:
        return InsolationResult(
            luminosity_lsun=luminosity_lsun, sma_au=sma_au,
            insolation_earth=float("nan"), hz_class="UNKNOWN", flag="INVALID_LUMINOSITY",
        )
    if not math.isfinite(sma_au) or sma_au <= 0.0:
        return InsolationResult(
            luminosity_lsun=luminosity_lsun, sma_au=sma_au,
            insolation_earth=float("nan"), hz_class="UNKNOWN", flag="INVALID_SMA",
        )

    s_eff = luminosity_lsun / sma_au**2

    t_star = teff_k - 5778.0
    t2 = t_star**2
    t3 = t_star**3
    t4 = t_star**4

    inner_s = 1.107 + 1.332e-4 * t_star + 1.580e-8 * t2 - 8.308e-12 * t3 - 1.931e-15 * t4
    outer_s = 0.356 + 6.171e-5 * t_star + 1.698e-9 * t2 - 3.198e-12 * t3 - 5.575e-16 * t4

    if s_eff > inner_s * 1.5:
        hz_class = "TOO_HOT"
    elif s_eff > inner_s:
        hz_class = "INNER_EDGE"
    elif s_eff >= outer_s:
        hz_class = "HABITABLE_ZONE"
    elif s_eff >= outer_s * 0.5:
        hz_class = "OUTER_EDGE"
    else:
        hz_class = "TOO_COLD"

    return InsolationResult(
        luminosity_lsun=luminosity_lsun,
        sma_au=sma_au,
        insolation_earth=round(s_eff, 6),
        hz_class=hz_class,
        flag="OK",
    )


def format_insolation_result(r: InsolationResult) -> str:
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Luminosity (L☉) | {r.luminosity_lsun:.4f} |\n"
        f"| Semi-major axis (AU) | {r.sma_au:.4f} |\n"
        f"| Insolation (S⊕) | {r.insolation_earth:.4f} |\n"
        f"| HZ Class | {r.hz_class} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Compute planetary insolation in Earth units.")
    p.add_argument("luminosity_lsun", type=float, help="Stellar luminosity in solar units")
    p.add_argument("sma_au", type=float, help="Semi-major axis in AU")
    p.add_argument("--teff", type=float, default=5778.0, help="Stellar Teff in K")
    args = p.parse_args()
    r = compute_insolation(args.luminosity_lsun, args.sma_au, args.teff)
    print(format_insolation_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
