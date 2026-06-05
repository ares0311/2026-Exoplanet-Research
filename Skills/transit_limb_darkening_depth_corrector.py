"""Correct observed transit depth for quadratic limb darkening."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class LDDepthCorrectionResult:
    observed_depth_ppm: float
    corrected_rp_rs: float          # true Rp/Rs accounting for LD
    geometric_depth_ppm: float      # δ_geom = (Rp/Rs)²  (no LD)
    ld_correction_factor: float     # δ_obs / δ_geom
    impact_parameter: float
    flag: str


def correct_depth_for_limb_darkening(
    observed_depth_ppm: float,
    u1: float,
    u2: float,
    impact_parameter: float = 0.0,
    n_iter: int = 20,
) -> LDDepthCorrectionResult:
    """Correct transit depth for quadratic limb darkening.

    For a quadratic LD law I(μ) = 1 - u1(1-μ) - u2(1-μ)²,
    the mid-transit depth is amplified over the geometric depth:
      δ_obs = δ_geom × I_centre(b) / I_mean
    where I_centre(b) = I(μ_c), μ_c = √(1 - b²)
    and I_mean = 1 - u1/3 - u2/6  (disk-integrated, Mandel & Agol 2002)

    Given δ_obs, invert to find true Rp/Rs:
      (Rp/Rs)² = δ_obs × I_mean / I_centre(b)

    Args:
        observed_depth_ppm: measured transit depth (ppm)
        u1: quadratic LD coefficient u1
        u2: quadratic LD coefficient u2
        impact_parameter: transit impact parameter b
        n_iter: iterations (unused; closed-form solution)
    """
    if observed_depth_ppm <= 0.0:
        return LDDepthCorrectionResult(observed_depth_ppm, float("nan"), float("nan"),
                                        float("nan"), impact_parameter, "INVALID_DEPTH")
    if not (-1.0 <= u1 <= 1.0 and -1.0 <= u2 <= 1.0):
        return LDDepthCorrectionResult(observed_depth_ppm, float("nan"), float("nan"),
                                        float("nan"), impact_parameter, "INVALID_LD")
    if not (0.0 <= impact_parameter < 1.5):
        return LDDepthCorrectionResult(observed_depth_ppm, float("nan"), float("nan"),
                                        float("nan"), impact_parameter, "INVALID_IMPACT")

    # μ at mid-transit centre
    mu_c = math.sqrt(max(1.0 - impact_parameter**2, 0.0))

    # LD intensity at centre of transit chord
    i_centre = 1.0 - u1 * (1.0 - mu_c) - u2 * (1.0 - mu_c)**2
    i_centre = max(i_centre, 1e-6)

    # Disk-integrated mean intensity (Mandel & Agol 2002)
    i_mean = 1.0 - u1 / 3.0 - u2 / 6.0
    i_mean = max(i_mean, 1e-6)

    # Correction factor: δ_obs = δ_geom × (i_centre / i_mean)  (simplified)
    # More precisely (small planet limit): δ_obs ≈ k² × i_centre / i_mean
    ld_factor = i_centre / i_mean

    # Geometric depth
    geom_depth_ppm = observed_depth_ppm / ld_factor
    rp_rs = math.sqrt(geom_depth_ppm * 1e-6)

    return LDDepthCorrectionResult(
        observed_depth_ppm=observed_depth_ppm,
        corrected_rp_rs=rp_rs,
        geometric_depth_ppm=geom_depth_ppm,
        ld_correction_factor=ld_factor,
        impact_parameter=impact_parameter,
        flag="OK",
    )


def format_ld_depth_correction_result(r: LDDepthCorrectionResult) -> str:
    if r.flag != "OK":
        return f"LDDepthCorrection | flag={r.flag}"
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Observed depth | {r.observed_depth_ppm:.1f} ppm |\n"
        f"| Geometric depth | {r.geometric_depth_ppm:.1f} ppm |\n"
        f"| Corrected Rp/Rs | {r.corrected_rp_rs:.5f} |\n"
        f"| LD correction factor | {r.ld_correction_factor:.4f} |\n"
        f"| Impact parameter | {r.impact_parameter:.3f} |\n"
        f"| Flag | {r.flag} |"
    )


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Transit depth limb-darkening corrector")
    p.add_argument("observed_depth_ppm", type=float)
    p.add_argument("u1", type=float)
    p.add_argument("u2", type=float)
    p.add_argument("--b", type=float, default=0.0)
    args = p.parse_args()
    r = correct_depth_for_limb_darkening(args.observed_depth_ppm, args.u1, args.u2,
                                          impact_parameter=args.b)
    print(format_ld_depth_correction_result(r))


if __name__ == "__main__":
    _cli()
