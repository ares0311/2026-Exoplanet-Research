"""Estimate transit depth bias from unocculted starspots."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StarspotDepthBiasResult:
    true_depth_ppm: float
    observed_depth_ppm: float
    depth_inflation_factor: float
    spot_contrast: float
    flag: str


def compute_starspot_depth_bias(
    observed_depth_ppm: float,
    spot_filling_factor: float,
    stellar_teff_k: float,
    spot_teff_k: float,
) -> StarspotDepthBiasResult:
    """Estimate true transit depth corrected for unocculted starspot bias.

    Unocculted starspots reduce the stellar flux baseline, inflating the observed
    transit depth. Désert et al. (2011) / Pont et al. (2013) formulation:

      δ_obs = δ_true / (1 - f_spot * (1 - F_spot/F_phot))

    where F_spot/F_phot ≈ (T_spot/T_phot)^4 (blackbody approximation).

    Args:
        observed_depth_ppm: measured transit depth (ppm)
        spot_filling_factor: fraction of stellar disk covered by spots (0–1)
        stellar_teff_k: photospheric temperature (K)
        spot_teff_k: spot temperature (K); typically Tphot - 200 to 400 K
    """
    if observed_depth_ppm <= 0.0:
        return StarspotDepthBiasResult(float("nan"), observed_depth_ppm,
                                        float("nan"), float("nan"), "INVALID_DEPTH")
    if not (0.0 <= spot_filling_factor < 1.0):
        return StarspotDepthBiasResult(float("nan"), observed_depth_ppm,
                                        float("nan"), float("nan"), "INVALID_FILLING_FACTOR")
    if stellar_teff_k <= 0.0:
        return StarspotDepthBiasResult(float("nan"), observed_depth_ppm,
                                        float("nan"), float("nan"), "INVALID_STELLAR_TEFF")
    if spot_teff_k <= 0.0 or spot_teff_k >= stellar_teff_k:
        return StarspotDepthBiasResult(float("nan"), observed_depth_ppm,
                                        float("nan"), float("nan"), "INVALID_SPOT_TEFF")

    spot_contrast = (spot_teff_k / stellar_teff_k) ** 4
    inflation_factor = 1.0 - spot_filling_factor * (1.0 - spot_contrast)
    true_depth = observed_depth_ppm * inflation_factor

    return StarspotDepthBiasResult(
        true_depth_ppm=true_depth,
        observed_depth_ppm=observed_depth_ppm,
        depth_inflation_factor=1.0 / inflation_factor,
        spot_contrast=spot_contrast,
        flag="OK",
    )


def format_starspot_depth_bias_result(r: StarspotDepthBiasResult) -> str:
    if r.flag != "OK":
        return f"StarspotDepthBias | flag={r.flag}"
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Observed depth | {r.observed_depth_ppm:.1f} ppm |\n"
        f"| True depth (corrected) | {r.true_depth_ppm:.1f} ppm |\n"
        f"| Depth inflation factor | {r.depth_inflation_factor:.4f} |\n"
        f"| Spot contrast (F_spot/F_phot) | {r.spot_contrast:.4f} |\n"
        f"| Flag | {r.flag} |"
    )


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Starspot transit depth bias estimator")
    p.add_argument("observed_depth_ppm", type=float, help="Observed transit depth (ppm)")
    p.add_argument("filling_factor", type=float, help="Spot filling factor (0–1)")
    p.add_argument("stellar_teff", type=float, help="Photospheric Teff (K)")
    p.add_argument("spot_teff", type=float, help="Spot temperature (K)")
    args = p.parse_args()
    r = compute_starspot_depth_bias(args.observed_depth_ppm, args.filling_factor,
                                     args.stellar_teff, args.spot_teff)
    print(format_starspot_depth_bias_result(r))


if __name__ == "__main__":
    _cli()
