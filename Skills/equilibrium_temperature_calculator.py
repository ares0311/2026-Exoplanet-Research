"""Compute planetary equilibrium temperature from stellar and orbital parameters.

T_eq = T_eff * (R_* / 2a)^(1/2) * (1 - A_B)^(1/4) * f^(1/4)

where f = 1/4 for uniform heat redistribution (default) or 2/3 for day-side only.

Public API
----------
EquilibriumTemperatureResult(teff_star_k, stellar_radius_rsun, semi_major_axis_au,
                              bond_albedo, heat_redistribution_f, teq_k, teq_err_k,
                              classification, flag)
compute_equilibrium_temperature(teff_star_k, stellar_radius_rsun, semi_major_axis_au,
                                *, bond_albedo, heat_redistribution_f,
                                a_au_err) -> EquilibriumTemperatureResult
format_equilibrium_temperature_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass

_RSUN_AU = 0.00465047  # solar radii to AU


@dataclass(frozen=True)
class EquilibriumTemperatureResult:
    teff_star_k: float
    stellar_radius_rsun: float
    semi_major_axis_au: float
    bond_albedo: float
    heat_redistribution_f: float
    teq_k: float
    teq_err_k: float | None
    classification: str  # "ultra-hot" | "hot" | "warm" | "temperate" | "cold"
    flag: str            # "OK" | "INVALID"


def _classify_teq(teq_k: float) -> str:
    if teq_k >= 2200:
        return "ultra-hot"
    if teq_k >= 1000:
        return "hot"
    if teq_k >= 400:
        return "warm"
    if teq_k >= 200:
        return "temperate"
    return "cold"


def compute_equilibrium_temperature(
    teff_star_k: float,
    stellar_radius_rsun: float,
    semi_major_axis_au: float,
    *,
    bond_albedo: float = 0.3,
    heat_redistribution_f: float = 0.25,
    a_au_err: float | None = None,
) -> EquilibriumTemperatureResult:
    """Compute planetary equilibrium temperature.

    Args:
        teff_star_k: Stellar effective temperature in Kelvin.
        stellar_radius_rsun: Stellar radius in solar radii.
        semi_major_axis_au: Semi-major axis in AU.
        bond_albedo: Bond albedo A_B (default 0.3).
        heat_redistribution_f: Heat redistribution factor f
            (0.25 = uniform; 0.667 = day-side only).
        a_au_err: Optional uncertainty on semi-major axis (AU).

    Returns:
        :class:`EquilibriumTemperatureResult`.
    """
    if (teff_star_k <= 0 or stellar_radius_rsun <= 0
            or semi_major_axis_au <= 0
            or not (0 <= bond_albedo < 1)
            or heat_redistribution_f <= 0):
        return EquilibriumTemperatureResult(
            teff_star_k, stellar_radius_rsun, semi_major_axis_au,
            bond_albedo, heat_redistribution_f,
            0.0, None, "cold", "INVALID",
        )

    r_star_au = stellar_radius_rsun * _RSUN_AU
    teq_k = (teff_star_k
             * math.sqrt(r_star_au / (2.0 * semi_major_axis_au))
             * (1.0 - bond_albedo) ** 0.25
             * heat_redistribution_f ** 0.25)

    teq_err_k: float | None = None
    if a_au_err is not None and a_au_err > 0:
        # Partial derivative dT/da = -0.5 * T / a
        teq_err_k = abs(0.5 * teq_k * a_au_err / semi_major_axis_au)
        teq_err_k = round(teq_err_k, 2)

    return EquilibriumTemperatureResult(
        teff_star_k=teff_star_k,
        stellar_radius_rsun=stellar_radius_rsun,
        semi_major_axis_au=semi_major_axis_au,
        bond_albedo=bond_albedo,
        heat_redistribution_f=heat_redistribution_f,
        teq_k=round(teq_k, 2),
        teq_err_k=teq_err_k,
        classification=_classify_teq(teq_k),
        flag="OK",
    )


def format_equilibrium_temperature_result(result: EquilibriumTemperatureResult) -> str:
    """Format equilibrium temperature result as Markdown."""
    lines = [
        "## Equilibrium Temperature",
        "",
        f"- T★: {result.teff_star_k:.0f} K",
        f"- R★: {result.stellar_radius_rsun:.3f} R☉",
        f"- a: {result.semi_major_axis_au:.4f} AU",
        f"- Bond albedo: {result.bond_albedo:.2f}",
        f"- Heat redistribution f: {result.heat_redistribution_f:.3f}",
    ]
    if result.flag == "INVALID":
        lines.append("- **Flag: INVALID** — non-positive or out-of-range inputs")
    else:
        err_str = (f" ± {result.teq_err_k:.1f} K"
                   if result.teq_err_k is not None else "")
        lines += [
            f"- T_eq: {result.teq_k:.1f}{err_str} K",
            f"- Classification: **{result.classification}**",
            f"- **Flag: {result.flag}**",
        ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="equilibrium_temperature_calculator",
        description="Compute planetary equilibrium temperature.",
    )
    parser.add_argument("teff_star_k", type=float)
    parser.add_argument("stellar_radius_rsun", type=float)
    parser.add_argument("semi_major_axis_au", type=float)
    parser.add_argument("--bond-albedo", type=float, default=0.3)
    parser.add_argument("--heat-redistribution-f", type=float, default=0.25)
    parser.add_argument("--a-au-err", type=float, default=None)
    args = parser.parse_args(argv)

    result = compute_equilibrium_temperature(
        args.teff_star_k, args.stellar_radius_rsun, args.semi_major_axis_au,
        bond_albedo=args.bond_albedo,
        heat_redistribution_f=args.heat_redistribution_f,
        a_au_err=args.a_au_err,
    )
    print(format_equilibrium_temperature_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
