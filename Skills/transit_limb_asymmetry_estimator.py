"""Estimate transit ingress/egress asymmetry from limb darkening and geometry."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class LimbAsymmetryResult:
    ingress_duration_hours: float
    egress_duration_hours: float
    asymmetry_ratio: float          # ingress / egress duration ratio
    asymmetry_seconds: float        # |T_ingress - T_egress| in seconds
    ld_depth_gradient_ppm: float    # depth variation over ingress due to LD
    flag: str


def compute_transit_limb_asymmetry(
    period_days: float,
    depth_ppm: float,
    impact_parameter: float = 0.0,
    stellar_radius_rsun: float = 1.0,
    stellar_mass_msun: float = 1.0,
    u1: float = 0.4,
    u2: float = 0.2,
    spin_orbit_lambda_deg: float = 0.0,
) -> LimbAsymmetryResult:
    """Compute ingress/egress asymmetry from LD + impact parameter + spin-orbit.

    For a misaligned orbit (λ ≠ 0°) the planet samples different LD profiles on
    ingress and egress, producing depth and timing asymmetry.

    Args:
        period_days: orbital period (days)
        depth_ppm: central transit depth (ppm)
        impact_parameter: impact parameter b
        stellar_radius_rsun: stellar radius (solar radii)
        stellar_mass_msun: stellar mass (solar masses)
        u1: linear limb-darkening coefficient
        u2: quadratic limb-darkening coefficient
        spin_orbit_lambda_deg: projected spin-orbit angle (degrees)
    """
    _G = 6.674e-11
    _MSUN_KG = 1.989e30
    _RSUN_M = 6.957e8

    if period_days <= 0.0:
        return LimbAsymmetryResult(float("nan"), float("nan"), float("nan"),
                                    float("nan"), float("nan"), "INVALID_PERIOD")
    if depth_ppm <= 0.0:
        return LimbAsymmetryResult(float("nan"), float("nan"), float("nan"),
                                    float("nan"), float("nan"), "INVALID_DEPTH")
    if not (0.0 <= impact_parameter < 1.0):
        return LimbAsymmetryResult(float("nan"), float("nan"), float("nan"),
                                    float("nan"), float("nan"), "INVALID_IMPACT")

    rs_m = stellar_radius_rsun * _RSUN_M
    ms_kg = stellar_mass_msun * _MSUN_KG
    p_s = period_days * 86400.0
    a_m = (_G * ms_kg * p_s**2 / (4.0 * math.pi**2)) ** (1.0 / 3.0)
    rp_rs = math.sqrt(depth_ppm * 1e-6)
    v = 2.0 * math.pi * a_m / p_s / rs_m  # in stellar radii per second

    t_ingress_s = 2.0 * rp_rs / math.sqrt(max(v**2 - (impact_parameter * v)**2, 1e-20))

    # For misaligned orbit, ingress sees limb at different angle than egress
    lam_rad = math.radians(spin_orbit_lambda_deg)
    # x_ingress_chord vs x_egress_chord limb position
    x_ingress = -(1.0 - rp_rs) * math.cos(lam_rad) - impact_parameter * math.sin(lam_rad)
    x_egress = (1.0 - rp_rs) * math.cos(lam_rad) - impact_parameter * math.sin(lam_rad)
    x_ingress = max(-1.0, min(1.0, x_ingress))
    x_egress = max(-1.0, min(1.0, x_egress))

    mu_ing = math.sqrt(max(1.0 - x_ingress**2, 0.0))
    mu_egr = math.sqrt(max(1.0 - x_egress**2, 0.0))
    I_ing = 1.0 - u1 * (1.0 - mu_ing) - u2 * (1.0 - mu_ing)**2
    I_egr = 1.0 - u1 * (1.0 - mu_egr) - u2 * (1.0 - mu_egr)**2

    t_egress_s = t_ingress_s * (I_egr / max(I_ing, 1e-10)) ** 0.5

    asym_s = abs(t_ingress_s - t_egress_s)
    asym_ratio = t_ingress_s / t_egress_s if t_egress_s > 0 else float("inf")
    ld_depth_grad = abs(I_ing - I_egr) * depth_ppm

    return LimbAsymmetryResult(
        ingress_duration_hours=t_ingress_s / 3600.0,
        egress_duration_hours=t_egress_s / 3600.0,
        asymmetry_ratio=asym_ratio,
        asymmetry_seconds=asym_s,
        ld_depth_gradient_ppm=ld_depth_grad,
        flag="OK",
    )


def format_limb_asymmetry_result(r: LimbAsymmetryResult) -> str:
    if r.flag != "OK":
        return f"LimbAsymmetry | flag={r.flag}"
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Ingress duration | {r.ingress_duration_hours*60:.2f} min |\n"
        f"| Egress duration | {r.egress_duration_hours*60:.2f} min |\n"
        f"| Ingress/egress ratio | {r.asymmetry_ratio:.4f} |\n"
        f"| Timing asymmetry | {r.asymmetry_seconds:.1f} s |\n"
        f"| LD depth gradient | {r.ld_depth_gradient_ppm:.2f} ppm |\n"
        f"| Flag | {r.flag} |"
    )


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Transit limb asymmetry estimator")
    p.add_argument("period_days", type=float)
    p.add_argument("depth_ppm", type=float)
    p.add_argument("--b", type=float, default=0.0)
    p.add_argument("--lambda-deg", type=float, default=0.0)
    args = p.parse_args()
    r = compute_transit_limb_asymmetry(args.period_days, args.depth_ppm,
                                        impact_parameter=args.b,
                                        spin_orbit_lambda_deg=args.lambda_deg)
    print(format_limb_asymmetry_result(r))


if __name__ == "__main__":
    _cli()
