"""
Scores stellar variability by computing peak-to-peak amplitude and RMS of a detrended flux series.

Public API:
    VariabilityResult     -- frozen dataclass holding variability diagnostics
    score_stellar_variability(flux) -> VariabilityResult
    format_variability(result) -> str
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class VariabilityResult:
    peak_to_peak_ppm: float
    rms_ppm: float
    variability_score: float
    flag: str


def score_stellar_variability(flux: list[float]) -> VariabilityResult:
    n = len(flux)
    if n == 0:
        return VariabilityResult(
            peak_to_peak_ppm=0.0,
            rms_ppm=0.0,
            variability_score=0.0,
            flag="NO_DATA",
        )

    mean_flux = sum(flux) / n
    if mean_flux == 0.0:
        mean_flux = 1.0

    flux_ppm = [(f / mean_flux - 1.0) * 1e6 for f in flux]

    peak_to_peak = max(flux_ppm) - min(flux_ppm)
    rms = math.sqrt(sum(v ** 2 for v in flux_ppm) / n)

    variability_score = min(rms / 1000.0, 1.0)
    flag = "HIGH_VARIABILITY" if rms > 500.0 else "OK"

    return VariabilityResult(
        peak_to_peak_ppm=peak_to_peak,
        rms_ppm=rms,
        variability_score=variability_score,
        flag=flag,
    )


def format_variability(result: VariabilityResult) -> str:
    lines = [
        "## Stellar Variability Score",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Peak-to-peak (ppm) | {result.peak_to_peak_ppm:.1f} |",
        f"| RMS (ppm) | {result.rms_ppm:.1f} |",
        f"| Variability score | {result.variability_score:.4f} |",
        f"| Flag | {result.flag} |",
    ]
    return "\n".join(lines)


def _cli() -> int:
    parser = argparse.ArgumentParser(
        description="Score stellar variability from a JSON flux file."
    )
    parser.add_argument("flux_file", help="JSON file containing list of flux values")
    args = parser.parse_args()

    with open(args.flux_file) as fh:
        flux = json.load(fh)

    result = score_stellar_variability(flux)
    print(format_variability(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
