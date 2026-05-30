"""Estimate expected out-of-transit baseline flux from stellar magnitude and aperture.

Uses a simple magnitude-to-flux conversion with a bandpass throughput factor.

Public API:
    BaselineFluxResult  -- frozen dataclass
    estimate_baseline_flux(magnitude, zero_point_flux, aperture_area_cm2,
                            throughput, exposure_time_s) -> BaselineFluxResult
    format_baseline_flux_result(result) -> str
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BaselineFluxResult:
    magnitude: float
    aperture_area_cm2: float
    throughput: float
    exposure_time_s: float
    flux_photons_per_s: float
    total_counts: float
    shot_noise_ppm: float
    flag: str


def estimate_baseline_flux(
    magnitude: float,
    zero_point_flux: float,
    aperture_area_cm2: float,
    throughput: float = 0.80,
    exposure_time_s: float = 1800.0,
) -> BaselineFluxResult:
    if zero_point_flux <= 0:
        return BaselineFluxResult(
            magnitude=magnitude, aperture_area_cm2=aperture_area_cm2,
            throughput=throughput, exposure_time_s=exposure_time_s,
            flux_photons_per_s=0.0, total_counts=0.0, shot_noise_ppm=0.0,
            flag="INVALID_ZERO_POINT",
        )
    if aperture_area_cm2 <= 0:
        return BaselineFluxResult(
            magnitude=magnitude, aperture_area_cm2=aperture_area_cm2,
            throughput=throughput, exposure_time_s=exposure_time_s,
            flux_photons_per_s=0.0, total_counts=0.0, shot_noise_ppm=0.0,
            flag="INVALID_APERTURE",
        )
    if not (0.0 < throughput <= 1.0):
        return BaselineFluxResult(
            magnitude=magnitude, aperture_area_cm2=aperture_area_cm2,
            throughput=throughput, exposure_time_s=exposure_time_s,
            flux_photons_per_s=0.0, total_counts=0.0, shot_noise_ppm=0.0,
            flag="INVALID_THROUGHPUT",
        )
    if exposure_time_s <= 0:
        return BaselineFluxResult(
            magnitude=magnitude, aperture_area_cm2=aperture_area_cm2,
            throughput=throughput, exposure_time_s=exposure_time_s,
            flux_photons_per_s=0.0, total_counts=0.0, shot_noise_ppm=0.0,
            flag="INVALID_EXPOSURE_TIME",
        )
    # flux = zero_point * 10^(-mag/2.5) photons/s/cm^2
    flux_per_area = zero_point_flux * 10.0 ** (-magnitude / 2.5)
    flux_photons_per_s = flux_per_area * aperture_area_cm2 * throughput
    total_counts = flux_photons_per_s * exposure_time_s
    shot_noise_ppm = (1.0 / math.sqrt(total_counts) * 1e6) if total_counts > 0 else 0.0
    return BaselineFluxResult(
        magnitude=magnitude,
        aperture_area_cm2=aperture_area_cm2,
        throughput=throughput,
        exposure_time_s=exposure_time_s,
        flux_photons_per_s=flux_photons_per_s,
        total_counts=total_counts,
        shot_noise_ppm=shot_noise_ppm,
        flag="OK",
    )


def format_baseline_flux_result(result: BaselineFluxResult) -> str:
    lines = [
        "## Transit Baseline Flux Estimate",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Magnitude | {result.magnitude:.2f} |",
        f"| Aperture Area (cm²) | {result.aperture_area_cm2:.1f} |",
        f"| Throughput | {result.throughput:.2f} |",
        f"| Exposure Time (s) | {result.exposure_time_s:.1f} |",
        f"| Flux (photons/s) | {result.flux_photons_per_s:.2e} |",
        f"| Total Counts | {result.total_counts:.2e} |",
        f"| Shot Noise (ppm) | {result.shot_noise_ppm:.1f} |",
        f"| Flag | {result.flag} |",
    ]
    return "\n".join(lines)


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Estimate baseline flux from stellar magnitude.")
    parser.add_argument("magnitude", type=float)
    parser.add_argument("zero_point_flux", type=float, help="Photons/s/cm^2 at mag 0.")
    parser.add_argument("aperture_area_cm2", type=float)
    parser.add_argument("--throughput", type=float, default=0.80)
    parser.add_argument("--exposure-time-s", type=float, default=1800.0)
    args = parser.parse_args()
    result = estimate_baseline_flux(
        args.magnitude, args.zero_point_flux, args.aperture_area_cm2,
        args.throughput, args.exposure_time_s,
    )
    print(format_baseline_flux_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
