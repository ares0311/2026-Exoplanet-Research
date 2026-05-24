"""Calculate atmospheric scale height and transmission spectroscopy amplitude.

Uses H = k_B × T_eq / (μ × g) to compute the scale height, then estimates
the peak-to-peak signal amplitude in transmission spectroscopy as
ΔF/F = 2 × N_H × H × R_p / R_s².  Distinct from ``equilibrium_temperature_calculator``
(only computes T_eq) and ``tsm_calculator`` (TSM/ESM figure-of-merit).

Public API
----------
ScaleHeightResult(t_eq_k, mean_mol_weight_amu, gravity_cgs,
                  scale_height_km, amplitude_ppm, n_scale_heights,
                  flag)
compute_scale_height(t_eq_k, mean_mol_weight_amu, gravity_cgs, *,
                     rp_rearth, rs_rsun, n_scale_heights) -> ScaleHeightResult
format_scale_height_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# Physical constants
_KB = 1.380649e-23      # J/K  (Boltzmann)
_AMU_KG = 1.66053906660e-27  # kg/amu
_REARTH_CM = 6.371e8   # cm
_RSUN_CM = 6.957e10    # cm
_CM_PER_KM = 1e5


@dataclass(frozen=True)
class ScaleHeightResult:
    t_eq_k: float
    mean_mol_weight_amu: float
    gravity_cgs: float             # cm/s²
    scale_height_km: float | None
    amplitude_ppm: float | None    # 2 × N_H × H × Rp / Rs²  in ppm
    n_scale_heights: int           # number of scale heights assumed (default 5)
    flag: str  # "OK" | "INVALID"


def compute_scale_height(
    t_eq_k: float,
    mean_mol_weight_amu: float,
    gravity_cgs: float,
    *,
    rp_rearth: float = 2.0,
    rs_rsun: float = 1.0,
    n_scale_heights: int = 5,
) -> ScaleHeightResult:
    """Compute atmospheric scale height and transmission spectroscopy amplitude.

    H = k_B × T_eq / (μ × g)
    ΔF/F = 2 × N_H × H × R_p / R_s²   (in ppm)

    Args:
        t_eq_k: Equilibrium temperature (K).
        mean_mol_weight_amu: Mean molecular weight (amu).  Use 2.3 for H₂/He,
            18 for H₂O, 28 for N₂/CO, 44 for CO₂.
        gravity_cgs: Surface gravity (cm/s²).  Typical range: 300–5000.
        rp_rearth: Planet radius (Earth radii).
        rs_rsun: Stellar radius (solar radii).
        n_scale_heights: Number of scale heights for amplitude estimate (default 5).

    Returns:
        :class:`ScaleHeightResult`.
    """
    for val in (t_eq_k, mean_mol_weight_amu, gravity_cgs, rp_rearth, rs_rsun):
        if not math.isfinite(val) or val <= 0:
            return ScaleHeightResult(
                t_eq_k, mean_mol_weight_amu, gravity_cgs, None, None, n_scale_heights, "INVALID"
            )
    if n_scale_heights < 1:
        return ScaleHeightResult(
            t_eq_k, mean_mol_weight_amu, gravity_cgs, None, None, n_scale_heights, "INVALID"
        )

    # Scale height in CGS, then convert to km
    mu_kg = mean_mol_weight_amu * _AMU_KG
    g_si = gravity_cgs * 1e-2          # cm/s² → m/s²
    h_m = _KB * t_eq_k / (mu_kg * g_si)
    h_km = h_m / 1e3

    # Transmission amplitude
    rp_cm = rp_rearth * _REARTH_CM
    rs_cm = rs_rsun * _RSUN_CM
    h_cm = h_km * _CM_PER_KM
    amplitude_ppm = 2.0 * n_scale_heights * h_cm * rp_cm / rs_cm ** 2 * 1e6

    return ScaleHeightResult(
        t_eq_k=t_eq_k,
        mean_mol_weight_amu=mean_mol_weight_amu,
        gravity_cgs=gravity_cgs,
        scale_height_km=round(h_km, 3),
        amplitude_ppm=round(amplitude_ppm, 3),
        n_scale_heights=n_scale_heights,
        flag="OK",
    )


def format_scale_height_result(result: ScaleHeightResult) -> str:
    """Format scale height result as Markdown."""
    lines = [
        "## Atmospheric Scale Height Calculator",
        "",
        f"- Equilibrium temperature: {result.t_eq_k} K",
        f"- Mean molecular weight: {result.mean_mol_weight_amu} amu",
        f"- Surface gravity: {result.gravity_cgs} cm/s²",
        f"- **Scale height H: {result.scale_height_km} km**",
        f"- Scale heights assumed: {result.n_scale_heights}",
        f"- **Transmission amplitude: {result.amplitude_ppm} ppm**",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="atmospheric_scale_height_calculator",
        description="Compute atmospheric scale height and transmission spectroscopy amplitude.",
    )
    parser.add_argument("t_eq_k", type=float)
    parser.add_argument("mean_mol_weight_amu", type=float)
    parser.add_argument("gravity_cgs", type=float)
    parser.add_argument("--rp-rearth", type=float, default=2.0)
    parser.add_argument("--rs-rsun", type=float, default=1.0)
    parser.add_argument("--n-scale-heights", type=int, default=5)
    args = parser.parse_args(argv)

    result = compute_scale_height(
        args.t_eq_k, args.mean_mol_weight_amu, args.gravity_cgs,
        rp_rearth=args.rp_rearth, rs_rsun=args.rs_rsun,
        n_scale_heights=args.n_scale_heights,
    )
    print(format_scale_height_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
