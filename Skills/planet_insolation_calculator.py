"""Compute stellar insolation flux received by an exoplanet."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

_LSUN_W = 3.828e26
_AU_M = 1.495978707e11
_SFLUX_EARTH = _LSUN_W / (4.0 * math.pi * _AU_M ** 2)  # W/m² at 1 AU, 1 L_sun


@dataclass(frozen=True)
class InsolationResult:
    luminosity_lsun: float
    semi_major_axis_au: float
    insolation_earth: float      # insolation relative to Earth (S_earth units)
    insolation_w_m2: float
    equilibrium_temp_k: float
    hz_inner_au: float
    hz_outer_au: float
    hz_class: str                # HABITABLE_ZONE / TOO_HOT / TOO_COLD
    flag: str


def compute_insolation(
    luminosity_lsun: float,
    semi_major_axis_au: float,
    teff_k: float | None = None,
    albedo: float = 0.3,
) -> InsolationResult:
    """
    Compute stellar insolation flux at a planet.

    S = L* / (4 pi a^2).
    T_eq = 278.5 * (L/Lsun)^0.25 * (1-A)^0.25 / sqrt(a/AU).
    HZ boundaries from Kopparapu et al. (2013): inner ~0.97 AU, outer ~1.67 AU,
    scaled as sqrt(L/Lsun).

    Parameters
    ----------
    luminosity_lsun:    Stellar luminosity in solar units.
    semi_major_axis_au: Planet semi-major axis in AU.
    teff_k:             Stellar effective temperature (optional; not used in calculation).
    albedo:             Bond albedo (default 0.3).
    """
    if not math.isfinite(luminosity_lsun) or luminosity_lsun <= 0:
        return InsolationResult(
            luminosity_lsun=luminosity_lsun, semi_major_axis_au=semi_major_axis_au,
            insolation_earth=float("nan"), insolation_w_m2=float("nan"),
            equilibrium_temp_k=float("nan"), hz_inner_au=float("nan"),
            hz_outer_au=float("nan"), hz_class="UNKNOWN", flag="INVALID_LUMINOSITY",
        )
    if not math.isfinite(semi_major_axis_au) or semi_major_axis_au <= 0:
        return InsolationResult(
            luminosity_lsun=luminosity_lsun, semi_major_axis_au=semi_major_axis_au,
            insolation_earth=float("nan"), insolation_w_m2=float("nan"),
            equilibrium_temp_k=float("nan"), hz_inner_au=float("nan"),
            hz_outer_au=float("nan"), hz_class="UNKNOWN", flag="INVALID_SMA",
        )

    lum_w = luminosity_lsun * _LSUN_W
    a_m = semi_major_axis_au * _AU_M
    flux_w_m2 = lum_w / (4.0 * math.pi * a_m ** 2)
    flux_earth = flux_w_m2 / _SFLUX_EARTH

    t_eq = 278.5 * (luminosity_lsun ** 0.25) * ((1.0 - albedo) ** 0.25) / math.sqrt(
        semi_major_axis_au
    )

    hz_inner = 0.97 * math.sqrt(luminosity_lsun)
    hz_outer = 1.67 * math.sqrt(luminosity_lsun)

    if semi_major_axis_au < hz_inner:
        hz_class = "TOO_HOT"
    elif semi_major_axis_au > hz_outer:
        hz_class = "TOO_COLD"
    else:
        hz_class = "HABITABLE_ZONE"

    return InsolationResult(
        luminosity_lsun=luminosity_lsun,
        semi_major_axis_au=semi_major_axis_au,
        insolation_earth=round(flux_earth, 6),
        insolation_w_m2=round(flux_w_m2, 2),
        equilibrium_temp_k=round(t_eq, 2),
        hz_inner_au=round(hz_inner, 4),
        hz_outer_au=round(hz_outer, 4),
        hz_class=hz_class,
        flag="OK",
    )


def format_insolation_result(r: InsolationResult) -> str:
    def _f(v: float, fmt: str = ".4f") -> str:
        return format(v, fmt) if math.isfinite(v) else "N/A"

    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Luminosity (Lsun) | {_f(r.luminosity_lsun)} |\n"
        f"| Semi-major axis (AU) | {_f(r.semi_major_axis_au)} |\n"
        f"| Insolation (S_earth) | {_f(r.insolation_earth)} |\n"
        f"| Insolation (W/m2) | {_f(r.insolation_w_m2, '.2f')} |\n"
        f"| Equilibrium temp (K) | {_f(r.equilibrium_temp_k, '.2f')} |\n"
        f"| HZ inner (AU) | {_f(r.hz_inner_au)} |\n"
        f"| HZ outer (AU) | {_f(r.hz_outer_au)} |\n"
        f"| HZ class | {r.hz_class} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Compute planet insolation flux.")
    p.add_argument("luminosity_lsun", type=float)
    p.add_argument("semi_major_axis_au", type=float)
    p.add_argument("--albedo", type=float, default=0.3)
    p.add_argument("--teff-k", type=float, default=None)
    args = p.parse_args()
    r = compute_insolation(args.luminosity_lsun, args.semi_major_axis_au,
                           teff_k=args.teff_k, albedo=args.albedo)
    print(format_insolation_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
