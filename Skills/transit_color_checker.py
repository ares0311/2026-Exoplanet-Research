"""Check multi-band transit depth chromaticity as a stellar contamination indicator.

Public API:
    ColorCheckResult  -- frozen dataclass
    check_transit_color(depths_ppm, band_names) -> ColorCheckResult
    format_color_check_result(result) -> str
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class ColorCheckResult:
    depths_ppm: list[float]
    band_names: list[str]
    mean_depth_ppm: float
    max_deviation_ppm: float
    max_deviation_band: str
    relative_variation: float
    chromatic: bool
    flag: str


def check_transit_color(
    depths_ppm: list[float],
    band_names: list[str] | None = None,
    chromaticity_threshold: float = 0.10,
) -> ColorCheckResult:
    n = len(depths_ppm)
    if n < 2:
        return ColorCheckResult(
            depths_ppm=list(depths_ppm), band_names=band_names or [],
            mean_depth_ppm=0.0, max_deviation_ppm=0.0, max_deviation_band="",
            relative_variation=0.0, chromatic=False, flag="INSUFFICIENT_DATA",
        )
    names = list(band_names) if band_names else [f"Band{i+1}" for i in range(n)]
    if len(names) != n:
        return ColorCheckResult(
            depths_ppm=list(depths_ppm), band_names=names,
            mean_depth_ppm=0.0, max_deviation_ppm=0.0, max_deviation_band="",
            relative_variation=0.0, chromatic=False, flag="BAND_NAME_MISMATCH",
        )
    if any(d < 0 for d in depths_ppm):
        return ColorCheckResult(
            depths_ppm=list(depths_ppm), band_names=names,
            mean_depth_ppm=0.0, max_deviation_ppm=0.0, max_deviation_band="",
            relative_variation=0.0, chromatic=False, flag="INVALID_DEPTH",
        )
    mean_depth = sum(depths_ppm) / n
    deviations = [abs(d - mean_depth) for d in depths_ppm]
    max_dev = max(deviations)
    max_band = names[deviations.index(max_dev)]
    rel_var = max_dev / mean_depth if mean_depth > 0 else 0.0
    chromatic = rel_var > chromaticity_threshold
    flag = "CHROMATIC" if chromatic else "OK"
    return ColorCheckResult(
        depths_ppm=list(depths_ppm),
        band_names=names,
        mean_depth_ppm=mean_depth,
        max_deviation_ppm=max_dev,
        max_deviation_band=max_band,
        relative_variation=rel_var,
        chromatic=chromatic,
        flag=flag,
    )


def format_color_check_result(result: ColorCheckResult) -> str:
    lines = [
        "## Transit Color Check",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Mean Depth (ppm) | {result.mean_depth_ppm:.1f} |",
        f"| Max Deviation (ppm) | {result.max_deviation_ppm:.1f} |",
        f"| Max Deviation Band | {result.max_deviation_band} |",
        f"| Relative Variation | {result.relative_variation:.4f} |",
        f"| Chromatic | {result.chromatic} |",
        f"| Flag | {result.flag} |",
    ]
    return "\n".join(lines)


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Check multi-band transit depth chromaticity.")
    parser.add_argument("depths", nargs="+", type=float, help="Transit depths in ppm per band.")
    parser.add_argument("--bands", nargs="+", default=None)
    parser.add_argument("--threshold", type=float, default=0.10)
    args = parser.parse_args()
    result = check_transit_color(args.depths, args.bands, args.threshold)
    print(format_color_check_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
