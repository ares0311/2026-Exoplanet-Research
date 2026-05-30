"""Assess the impact of a stellar flare on a transit depth measurement.

Public API:
    FlareImpactResult  -- frozen dataclass
    assess_flare_impact(flare_amplitude_ppm, flare_duration_hours,
                         transit_duration_hours, baseline_flux) -> FlareImpactResult
    format_flare_impact_result(result) -> str
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class FlareImpactResult:
    flare_amplitude_ppm: float
    flare_duration_hours: float
    transit_duration_hours: float
    overlap_fraction: float
    flare_energy_in_transit: float
    depth_contamination_ppm: float
    significant: bool
    flag: str


def assess_flare_impact(
    flare_amplitude_ppm: float,
    flare_duration_hours: float,
    transit_duration_hours: float,
    baseline_flux: float = 1.0,
    significance_threshold_ppm: float = 100.0,
) -> FlareImpactResult:
    if flare_amplitude_ppm < 0:
        return FlareImpactResult(
            flare_amplitude_ppm=flare_amplitude_ppm, flare_duration_hours=flare_duration_hours,
            transit_duration_hours=transit_duration_hours, overlap_fraction=0.0,
            flare_energy_in_transit=0.0, depth_contamination_ppm=0.0,
            significant=False, flag="INVALID_AMPLITUDE",
        )
    if flare_duration_hours <= 0:
        return FlareImpactResult(
            flare_amplitude_ppm=flare_amplitude_ppm, flare_duration_hours=flare_duration_hours,
            transit_duration_hours=transit_duration_hours, overlap_fraction=0.0,
            flare_energy_in_transit=0.0, depth_contamination_ppm=0.0,
            significant=False, flag="INVALID_FLARE_DURATION",
        )
    if transit_duration_hours <= 0:
        return FlareImpactResult(
            flare_amplitude_ppm=flare_amplitude_ppm, flare_duration_hours=flare_duration_hours,
            transit_duration_hours=transit_duration_hours, overlap_fraction=0.0,
            flare_energy_in_transit=0.0, depth_contamination_ppm=0.0,
            significant=False, flag="INVALID_TRANSIT_DURATION",
        )
    if baseline_flux <= 0:
        return FlareImpactResult(
            flare_amplitude_ppm=flare_amplitude_ppm, flare_duration_hours=flare_duration_hours,
            transit_duration_hours=transit_duration_hours, overlap_fraction=0.0,
            flare_energy_in_transit=0.0, depth_contamination_ppm=0.0,
            significant=False, flag="INVALID_BASELINE",
        )
    overlap = min(flare_duration_hours, transit_duration_hours)
    overlap_fraction = overlap / transit_duration_hours
    # Mean contamination in transit assuming triangular flare shape
    # Energy = 0.5 * amplitude * duration; mean amplitude in transit = 0.5 * A * overlap/flare_dur
    mean_flare_in_transit = 0.5 * flare_amplitude_ppm * (overlap / flare_duration_hours)
    flare_energy = mean_flare_in_transit * transit_duration_hours
    depth_contamination = mean_flare_in_transit  # ppm added to baseline during transit
    significant = depth_contamination >= significance_threshold_ppm
    flag = "SIGNIFICANT_CONTAMINATION" if significant else "OK"
    return FlareImpactResult(
        flare_amplitude_ppm=flare_amplitude_ppm,
        flare_duration_hours=flare_duration_hours,
        transit_duration_hours=transit_duration_hours,
        overlap_fraction=overlap_fraction,
        flare_energy_in_transit=flare_energy,
        depth_contamination_ppm=depth_contamination,
        significant=significant,
        flag=flag,
    )


def format_flare_impact_result(result: FlareImpactResult) -> str:
    lines = [
        "## Stellar Flare Impact Assessment",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Flare Amplitude (ppm) | {result.flare_amplitude_ppm:.1f} |",
        f"| Flare Duration (hr) | {result.flare_duration_hours:.2f} |",
        f"| Transit Duration (hr) | {result.transit_duration_hours:.2f} |",
        f"| Overlap Fraction | {result.overlap_fraction:.4f} |",
        f"| Depth Contamination (ppm) | {result.depth_contamination_ppm:.1f} |",
        f"| Significant | {result.significant} |",
        f"| Flag | {result.flag} |",
    ]
    return "\n".join(lines)


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Assess stellar flare impact on transit depth.")
    parser.add_argument("flare_amplitude_ppm", type=float)
    parser.add_argument("flare_duration_hours", type=float)
    parser.add_argument("transit_duration_hours", type=float)
    parser.add_argument("--baseline-flux", type=float, default=1.0)
    parser.add_argument("--threshold-ppm", type=float, default=100.0)
    args = parser.parse_args()
    result = assess_flare_impact(
        args.flare_amplitude_ppm, args.flare_duration_hours,
        args.transit_duration_hours, args.baseline_flux, args.threshold_ppm,
    )
    print(format_flare_impact_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
