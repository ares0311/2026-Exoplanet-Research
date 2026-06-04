"""Estimate quality of a detrended light curve relative to a transit signal."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DetrendingQualityResult:
    oot_rms_ppm: float
    transit_depth_ppm: float
    snr_ratio: float
    residual_scatter_ppm: float
    detrending_quality_score: float  # 0–1
    quality_grade: str               # A / B / C / D / F
    flag: str


def compute_detrending_quality(
    oot_flux: list[float],
    in_transit_flux: list[float],
    transit_depth_ppm: float,
    detrended_residuals: list[float] | None = None,
) -> DetrendingQualityResult:
    """Score the quality of a detrended light curve for transit detection.

    Quality score based on:
    - OOT RMS vs photon noise floor (lower is better detrending)
    - Residual scatter in detrended model vs transit depth (SNR proxy)
    - Absence of correlated systematics (measured via runs test on residuals)

    Args:
        oot_flux: out-of-transit flux values (normalised to 1.0)
        in_transit_flux: in-transit flux values
        transit_depth_ppm: expected or measured transit depth (ppm)
        detrended_residuals: residuals after detrending; if None uses oot_flux residuals
    """
    if len(oot_flux) < 5:
        return DetrendingQualityResult(float("nan"), transit_depth_ppm, float("nan"),
                                        float("nan"), 0.0, "F", "INSUFFICIENT_OOT_DATA")
    if transit_depth_ppm <= 0.0:
        return DetrendingQualityResult(float("nan"), transit_depth_ppm, float("nan"),
                                        float("nan"), 0.0, "F", "INVALID_DEPTH")

    n_oot = len(oot_flux)
    mean_oot = sum(oot_flux) / n_oot
    var_oot = sum((f - mean_oot) ** 2 for f in oot_flux) / (n_oot - 1)
    oot_rms = math.sqrt(var_oot) * 1e6  # ppm

    residuals = detrended_residuals if detrended_residuals is not None else oot_flux
    n_res = len(residuals)
    mean_res = sum(residuals) / n_res
    var_res = sum((r - mean_res) ** 2 for r in residuals) / max(n_res - 1, 1)
    res_scatter = math.sqrt(var_res) * 1e6

    snr_ratio = transit_depth_ppm / max(oot_rms, 1.0)

    # Simple runs test: count sign changes in residuals
    n_runs = 1
    for i in range(1, n_res):
        prev = residuals[i - 1] - mean_res
        curr = residuals[i] - mean_res
        if (prev >= 0) != (curr >= 0):
            n_runs += 1
    expected_runs = (2 * n_res - 1) / 3.0
    runs_fraction = n_runs / max(expected_runs, 1.0)
    runs_score = min(runs_fraction, 1.0)  # 1.0 = well randomised residuals

    # Overall quality score: 0.5*SNR_component + 0.3*runs + 0.2*scatter_component
    snr_score = min(snr_ratio / 10.0, 1.0)
    scatter_component = max(0.0, 1.0 - res_scatter / transit_depth_ppm)
    quality_score = 0.5 * snr_score + 0.3 * runs_score + 0.2 * scatter_component

    if quality_score >= 0.85:
        grade = "A"
    elif quality_score >= 0.70:
        grade = "B"
    elif quality_score >= 0.50:
        grade = "C"
    elif quality_score >= 0.30:
        grade = "D"
    else:
        grade = "F"

    return DetrendingQualityResult(
        oot_rms_ppm=oot_rms,
        transit_depth_ppm=transit_depth_ppm,
        snr_ratio=snr_ratio,
        residual_scatter_ppm=res_scatter,
        detrending_quality_score=quality_score,
        quality_grade=grade,
        flag="OK",
    )


def format_detrending_quality_result(r: DetrendingQualityResult) -> str:
    if r.flag != "OK":
        return f"DetrendingQuality | flag={r.flag}"
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| OOT RMS | {r.oot_rms_ppm:.1f} ppm |\n"
        f"| Transit depth | {r.transit_depth_ppm:.1f} ppm |\n"
        f"| Depth / OOT RMS (SNR proxy) | {r.snr_ratio:.2f} |\n"
        f"| Residual scatter | {r.residual_scatter_ppm:.1f} ppm |\n"
        f"| Quality score | {r.detrending_quality_score:.3f} |\n"
        f"| Quality grade | {r.quality_grade} |\n"
        f"| Flag | {r.flag} |"
    )


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Detrending quality estimator")
    p.add_argument("transit_depth_ppm", type=float, help="Transit depth (ppm)")
    p.add_argument("--oot", type=float, nargs="+", required=True, help="OOT flux values")
    p.add_argument("--in-transit", type=float, nargs="+", default=None)
    args = p.parse_args()
    in_tr = args.in_transit or [1.0 - args.transit_depth_ppm * 1e-6]
    r = compute_detrending_quality(args.oot, in_tr, args.transit_depth_ppm)
    print(format_detrending_quality_result(r))


if __name__ == "__main__":
    _cli()
