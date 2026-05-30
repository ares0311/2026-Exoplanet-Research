"""
Checks spatial correlation between flux residuals and centroid offsets.

Public API:
    SpatialCorrResult     -- frozen dataclass holding correlation diagnostics
    check_spatial_correlation(flux_residuals, centroid_offsets) -> SpatialCorrResult
    format_spatial_corr(result) -> str
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SpatialCorrResult:
    pearson_r: float
    n_points: int
    abs_r: float
    correlated: bool
    flag: str


def check_spatial_correlation(
    flux_residuals: list[float],
    centroid_offsets: list[float],
) -> SpatialCorrResult:
    n = min(len(flux_residuals), len(centroid_offsets))
    if n < 2:
        return SpatialCorrResult(
            pearson_r=0.0,
            n_points=n,
            abs_r=0.0,
            correlated=False,
            flag="INSUFFICIENT_DATA",
        )

    x = flux_residuals[:n]
    y = centroid_offsets[:n]

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    dx = [v - mean_x for v in x]
    dy = [v - mean_y for v in y]

    numerator = sum(a * b for a, b in zip(dx, dy))
    denom_x = math.sqrt(sum(a ** 2 for a in dx))
    denom_y = math.sqrt(sum(b ** 2 for b in dy))

    if denom_x == 0.0 or denom_y == 0.0:
        pearson_r = 0.0
    else:
        pearson_r = numerator / (denom_x * denom_y)

    abs_r = abs(pearson_r)
    correlated = abs_r > 0.5
    flag = "SPATIAL_CORRELATION" if correlated else "OK"

    return SpatialCorrResult(
        pearson_r=pearson_r,
        n_points=n,
        abs_r=abs_r,
        correlated=correlated,
        flag=flag,
    )


def format_spatial_corr(result: SpatialCorrResult) -> str:
    lines = [
        "## Spatial Correlation Check",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Points | {result.n_points} |",
        f"| Pearson r | {result.pearson_r:.4f} |",
        f"| |r| | {result.abs_r:.4f} |",
        f"| Correlated (|r|>0.5) | {result.correlated} |",
        f"| Flag | {result.flag} |",
    ]
    return "\n".join(lines)


def _cli() -> int:
    parser = argparse.ArgumentParser(
        description="Check spatial correlation between flux residuals and centroid offsets."
    )
    parser.add_argument(
        "residuals_file", help="JSON file containing list of flux residual values"
    )
    parser.add_argument(
        "offsets_file", help="JSON file containing list of centroid offset values"
    )
    args = parser.parse_args()

    with open(args.residuals_file) as fh:
        residuals = json.load(fh)
    with open(args.offsets_file) as fh:
        offsets = json.load(fh)

    result = check_spatial_correlation(residuals, offsets)
    print(format_spatial_corr(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
