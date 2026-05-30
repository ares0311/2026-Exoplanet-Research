"""Check per-sector flux normalization quality.

Public API:
    FluxNormResult  -- frozen dataclass
    check_flux_normalization(sector_fluxes, *, target_median, tolerance) -> FluxNormResult
    format_flux_norm_result(result) -> str
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 == 1 else (s[mid - 1] + s[mid]) / 2.0


@dataclass(frozen=True)
class FluxNormResult:
    sector_medians: list[float]
    offsets: list[float]
    max_offset: float
    n_sectors_flagged: int
    flag: str


def check_flux_normalization(
    sector_fluxes: list[list[float]],
    *,
    target_median: float = 1.0,
    tolerance: float = 0.005,
) -> FluxNormResult:
    if len(sector_fluxes) < 1:
        return FluxNormResult(
            sector_medians=[], offsets=[], max_offset=0.0,
            n_sectors_flagged=0, flag="NO_DATA",
        )
    for s in sector_fluxes:
        if len(s) < 1:
            return FluxNormResult(
                sector_medians=[], offsets=[], max_offset=0.0,
                n_sectors_flagged=0, flag="EMPTY_SECTOR",
            )
    sector_medians = [_median(s) for s in sector_fluxes]
    offsets = [abs(m - target_median) for m in sector_medians]
    max_offset = max(offsets)
    n_flagged = sum(1 for o in offsets if o > tolerance)
    flag = "NORMALIZATION_ISSUE" if n_flagged > 0 else "OK"
    return FluxNormResult(
        sector_medians=sector_medians,
        offsets=offsets,
        max_offset=max_offset,
        n_sectors_flagged=n_flagged,
        flag=flag,
    )


def format_flux_norm_result(result: FluxNormResult) -> str:
    lines = [
        "## Flux Normalization Check",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| N Sectors | {len(result.sector_medians)} |",
        f"| Max Offset from Target | {result.max_offset:.6f} |",
        f"| N Sectors Flagged | {result.n_sectors_flagged} |",
        f"| Flag | {result.flag} |",
    ]
    return "\n".join(lines)


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Check flux normalization per sector.")
    parser.add_argument("sector_fluxes_file", help="JSON array of per-sector flux arrays.")
    parser.add_argument("--target-median", type=float, default=1.0)
    parser.add_argument("--tolerance", type=float, default=0.005)
    args = parser.parse_args()
    with open(args.sector_fluxes_file) as fh:
        sector_fluxes = json.load(fh)
    result = check_flux_normalization(
        sector_fluxes, target_median=args.target_median, tolerance=args.tolerance,
    )
    print(format_flux_norm_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
