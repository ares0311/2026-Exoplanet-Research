"""Correct transit depth and FPP for unresolved binary companion dilution."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BinaryDilutionResult:
    dilution_factor: float          # D = F_companion / (F_target + F_companion)
    corrected_depth_ppm: float      # δ_true = δ_obs / (1 - D)
    corrected_rp_rs: float          # √(δ_true)
    fpp_inflation_factor: float     # FPP is inflated by this factor due to blending
    depth_ratio: float              # δ_corrected / δ_observed
    flag: str


def correct_for_binary_dilution(
    observed_depth_ppm: float,
    delta_magnitude: float,
    bandpass: str = "TESS",
    flux_ratio_override: float | None = None,
) -> BinaryDilutionResult:
    """Correct transit depth for flux dilution by unresolved companion.

    Dilution factor from magnitude difference:
      F_comp / F_tot = 10^(-Δm/2.5) / (1 + 10^(-Δm/2.5))
    Corrected depth:
      δ_true = δ_obs / (1 - D)  where D = F_comp / F_tot

    Diluted transit depth underestimates Rp/Rs by (1-D)^(1/2).
    A blended background EB (on the companion) would have:
      FPP_apparent ~ FPP_true × (1 - D)  (easier to confuse with planet)

    Args:
        observed_depth_ppm: observed (diluted) transit depth (ppm)
        delta_magnitude: magnitude difference Δm = m_companion - m_target
            (positive means companion is fainter)
        bandpass: photometric bandpass for context (TESS / Kepler / V)
        flux_ratio_override: override flux ratio F_comp/F_target directly
    """
    if observed_depth_ppm <= 0.0:
        return BinaryDilutionResult(float("nan"), float("nan"), float("nan"),
                                     float("nan"), float("nan"), "INVALID_DEPTH")

    if flux_ratio_override is not None:
        if flux_ratio_override < 0.0:
            return BinaryDilutionResult(float("nan"), float("nan"), float("nan"),
                                         float("nan"), float("nan"), "INVALID_FLUX_RATIO")
        flux_ratio = flux_ratio_override
    else:
        flux_ratio = 10.0 ** (-delta_magnitude / 2.5)

    # Dilution factor D = F_comp / (F_target + F_comp)
    dilution = flux_ratio / (1.0 + flux_ratio)
    dilution = max(0.0, min(dilution, 0.999))

    # Corrected depth
    corrected_ppm = observed_depth_ppm / (1.0 - dilution)

    # Corrected Rp/Rs
    corrected_rp_rs = math.sqrt(corrected_ppm * 1e-6)

    # FPP inflation: blend makes shallow transits look deeper → lower apparent FPP
    fpp_inflation = 1.0 / (1.0 - dilution) if dilution < 1.0 else float("inf")

    depth_ratio = corrected_ppm / observed_depth_ppm

    return BinaryDilutionResult(
        dilution_factor=dilution,
        corrected_depth_ppm=corrected_ppm,
        corrected_rp_rs=corrected_rp_rs,
        fpp_inflation_factor=fpp_inflation,
        depth_ratio=depth_ratio,
        flag="OK",
    )


def format_binary_dilution_result(r: BinaryDilutionResult) -> str:
    if r.flag != "OK":
        return f"BinaryDilution | flag={r.flag}"
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Dilution factor D | {r.dilution_factor:.4f} |\n"
        f"| Corrected depth | {r.corrected_depth_ppm:.1f} ppm |\n"
        f"| Corrected Rp/Rs | {r.corrected_rp_rs:.4f} |\n"
        f"| Depth correction factor | {r.depth_ratio:.3f}× |\n"
        f"| FPP inflation factor | {r.fpp_inflation_factor:.3f}× |\n"
        f"| Flag | {r.flag} |"
    )


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Binary dilution corrector for transit depth")
    p.add_argument("observed_depth_ppm", type=float)
    p.add_argument("delta_magnitude", type=float)
    p.add_argument("--bandpass", type=str, default="TESS")
    args = p.parse_args()
    r = correct_for_binary_dilution(args.observed_depth_ppm, args.delta_magnitude,
                                     bandpass=args.bandpass)
    print(format_binary_dilution_result(r))


if __name__ == "__main__":
    _cli()
