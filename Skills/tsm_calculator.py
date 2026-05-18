"""Compute the Transmission Spectroscopy Metric (TSM) and Emission Spectroscopy
Metric (ESM) following Kempton et al. (2018).

TSM = scale_factor * Rp^3 * Teq / (Mp * Rs^2) * 10^(-mJ/5)
ESM = 4.29e6 * (Bλ(T_day)/Bλ(T_eff)) * (Rp/Rs)^2 * 10^(-K/5)

Public API
----------
TSMResult(tic_id, planet_radius_rearth, planet_mass_mearth, teq_k,
          stellar_radius_rsun, tmag, scale_factor, tsm, esm,
          size_class, flag)
compute_tsm(planet_radius_rearth, teq_k, stellar_radius_rsun, tmag, *,
            tic_id, planet_mass_mearth, depth_ppm, teff_star_k) -> TSMResult
format_tsm_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# Kempton+2018 Table 1 scale factors by size class
_SCALE_FACTORS = {
    "terrestrial": 0.190,    # Rp <= 1.5 R_earth
    "super-Earth": 1.26,     # 1.5 < Rp <= 2.75
    "sub-Neptune": 1.28,     # 2.75 < Rp <= 4.0
    "giant": 1.15,           # Rp > 4.0
}

_REARTH_TO_RJUP = 1.0 / 11.21
_RJUP_TO_RSUN = 0.10279

# Chen & Kipping (2017) M-R relation break points
_MR_BREAK_REARTH = 1.23   # boundary between volatile/rocky in R_earth
_MR_BREAK_REARTH2 = 14.26  # upper boundary


def _chen_kipping_mass(rp_rearth: float) -> float:
    """Empirical mass estimate from Chen & Kipping (2017) power-law M-R relation."""
    if rp_rearth <= _MR_BREAK_REARTH:
        return 0.9718 * rp_rearth ** 3.58   # rocky regime
    if rp_rearth <= _MR_BREAK_REARTH2:
        return 1.436 * rp_rearth ** 1.70    # volatile regime
    return 1.436 * _MR_BREAK_REARTH2 ** 1.70  # cap at upper boundary


def _size_class(rp_rearth: float) -> tuple[str, float]:
    """Return (size_class, scale_factor)."""
    if rp_rearth <= 1.5:
        return "terrestrial", _SCALE_FACTORS["terrestrial"]
    if rp_rearth <= 2.75:
        return "super-Earth", _SCALE_FACTORS["super-Earth"]
    if rp_rearth <= 4.0:
        return "sub-Neptune", _SCALE_FACTORS["sub-Neptune"]
    return "giant", _SCALE_FACTORS["giant"]


def _planck_ratio(t_planet: float, t_star: float) -> float:
    """Approximate ratio Bλ(T_planet)/Bλ(T_star) in the mid-infrared (7.5 µm).

    Uses the Rayleigh-Jeans / Wein hybrid approximation:
    Bλ ∝ T in R-J limit; here we use the full Planck at λ = 7.5 µm.
    """
    hck = 19232.0  # h*c/k_B in µm·K (λ in µm)
    lam = 7.5
    x_p = hck / (lam * t_planet) if t_planet > 0 else 1e10
    x_s = hck / (lam * t_star) if t_star > 0 else 1e10
    # B ∝ 1/(exp(x)-1); ratio = (exp(x_s)-1)/(exp(x_p)-1)
    try:
        return (math.expm1(x_s)) / (math.expm1(x_p))
    except (OverflowError, ZeroDivisionError):
        return 0.0


@dataclass(frozen=True)
class TSMResult:
    tic_id: int
    planet_radius_rearth: float
    planet_mass_mearth: float
    teq_k: float
    stellar_radius_rsun: float
    tmag: float
    scale_factor: float
    tsm: float
    esm: float
    size_class: str
    flag: str  # "OK" | "INVALID"


def compute_tsm(
    planet_radius_rearth: float,
    teq_k: float,
    stellar_radius_rsun: float,
    tmag: float,
    *,
    tic_id: int = 0,
    planet_mass_mearth: float | None = None,
    depth_ppm: float | None = None,
    teff_star_k: float | None = None,
) -> TSMResult:
    """Compute TSM and ESM.

    Args:
        planet_radius_rearth: Planet radius in Earth radii.
        teq_k: Equilibrium temperature in Kelvin.
        stellar_radius_rsun: Stellar radius in solar radii.
        tmag: TESS magnitude (used as proxy for J-band for TSM; K for ESM).
        tic_id: TIC ID.
        planet_mass_mearth: Planet mass in Earth masses (None → Chen & Kipping).
        depth_ppm: Transit depth in ppm (for Rp/Rs in ESM; derived if not given).
        teff_star_k: Stellar effective temperature for ESM (uses 5778 K if None).

    Returns:
        :class:`TSMResult`.
    """
    if planet_radius_rearth <= 0 or teq_k <= 0 or stellar_radius_rsun <= 0:
        return TSMResult(tic_id, planet_radius_rearth,
                         planet_mass_mearth or 0.0, teq_k,
                         stellar_radius_rsun, tmag, 0.0, 0.0, 0.0, "terrestrial", "INVALID")

    mp = (planet_mass_mearth if planet_mass_mearth is not None
          else _chen_kipping_mass(planet_radius_rearth))
    size_cls, scale = _size_class(planet_radius_rearth)

    # TSM (Kempton+2018 eq. 1) — using Tmag as proxy for J-band
    tsm = (scale * planet_radius_rearth ** 3 * teq_k
           / (mp * stellar_radius_rsun ** 2)
           * 10 ** (-tmag / 5.0))

    # ESM: depth proxy from radius ratio
    _rsun_rearth = 109.076
    rp_over_rs = planet_radius_rearth / (stellar_radius_rsun * _rsun_rearth)
    if depth_ppm is not None and depth_ppm > 0:
        rp_over_rs = math.sqrt(depth_ppm / 1e6)

    t_star = teff_star_k if teff_star_k is not None else 5778.0
    t_day = 1.10 * teq_k  # day-side temperature approximation

    planck_r = _planck_ratio(t_day, t_star)
    esm = 4.29e6 * planck_r * rp_over_rs ** 2 * 10 ** (-tmag / 5.0)

    return TSMResult(
        tic_id=tic_id,
        planet_radius_rearth=planet_radius_rearth,
        planet_mass_mearth=round(mp, 4),
        teq_k=teq_k,
        stellar_radius_rsun=stellar_radius_rsun,
        tmag=tmag,
        scale_factor=scale,
        tsm=round(tsm, 4),
        esm=round(esm, 4),
        size_class=size_cls,
        flag="OK",
    )


def format_tsm_result(result: TSMResult) -> str:
    """Format TSM result as Markdown."""
    lines = [
        "## Spectroscopy Metrics",
        "",
        f"- TIC ID: {result.tic_id}",
        f"- Rp: {result.planet_radius_rearth:.3f} R⊕ ({result.size_class})",
        f"- Mp: {result.planet_mass_mearth:.3f} M⊕",
        f"- Teq: {result.teq_k:.0f} K",
        f"- R★: {result.stellar_radius_rsun:.3f} R☉",
        f"- Tmag: {result.tmag:.2f}",
    ]
    if result.flag == "INVALID":
        lines.append("- **Flag: INVALID**")
    else:
        lines += [
            f"- Scale factor: {result.scale_factor}",
            f"- **TSM: {result.tsm:.2f}**",
            f"- **ESM: {result.esm:.2f}**",
            f"- **Flag: {result.flag}**",
        ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="tsm_calculator",
        description="Compute TSM and ESM for a planet candidate.",
    )
    parser.add_argument("planet_radius_rearth", type=float)
    parser.add_argument("teq_k", type=float)
    parser.add_argument("stellar_radius_rsun", type=float)
    parser.add_argument("tmag", type=float)
    parser.add_argument("--tic-id", type=int, default=0)
    parser.add_argument("--planet-mass-mearth", type=float, default=None)
    parser.add_argument("--teff-star-k", type=float, default=None)
    args = parser.parse_args(argv)

    result = compute_tsm(
        args.planet_radius_rearth, args.teq_k, args.stellar_radius_rsun, args.tmag,
        tic_id=args.tic_id,
        planet_mass_mearth=args.planet_mass_mearth,
        teff_star_k=args.teff_star_k,
    )
    print(format_tsm_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
