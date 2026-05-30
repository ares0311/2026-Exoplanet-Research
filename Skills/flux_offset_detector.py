"""Flux offset detector across TESS sectors.

Detects systematic flux level offsets between TESS sectors, which can indicate
data-reduction artefacts or background contamination that varies between
spacecraft pointings.

Public API
----------
FluxOffsetResult(sector_means, grand_mean, max_offset, offset_fraction,
                 offset_detected, flag)
detect_flux_offset(sector_fluxes, *, offset_threshold=0.005)
    -> FluxOffsetResult
format_flux_offset_result(result) -> str
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class FluxOffsetResult:
    sector_means: list[float]
    grand_mean: float
    max_offset: float
    offset_fraction: float
    offset_detected: bool
    flag: str


def detect_flux_offset(
    sector_fluxes: list[list[float]],
    *,
    offset_threshold: float = 0.005,
) -> FluxOffsetResult:
    """Detect flux-level offsets between sectors.

    Parameters
    ----------
    sector_fluxes:    list of per-sector flux lists (normalised relative flux)
    offset_threshold: maximum relative offset before the flag is raised
                      (default 0.005 = 0.5 %)

    Returns
    -------
    FluxOffsetResult with flag one of:
    INSUFFICIENT_SECTORS, EMPTY_SECTOR, OFFSET_DETECTED, OK
    """
    if len(sector_fluxes) < 2:
        return FluxOffsetResult(
            sector_means=[],
            grand_mean=0.0,
            max_offset=0.0,
            offset_fraction=0.0,
            offset_detected=False,
            flag="INSUFFICIENT_SECTORS",
        )

    for fluxes in sector_fluxes:
        if len(fluxes) == 0:
            return FluxOffsetResult(
                sector_means=[],
                grand_mean=0.0,
                max_offset=0.0,
                offset_fraction=0.0,
                offset_detected=False,
                flag="EMPTY_SECTOR",
            )

    sector_means = [sum(fluxes) / len(fluxes) for fluxes in sector_fluxes]
    grand_mean = sum(sector_means) / len(sector_means)

    offsets: list[float] = []
    for m in sector_means:
        if grand_mean != 0.0:
            offsets.append(abs(m - grand_mean) / abs(grand_mean))
        else:
            offsets.append(abs(m))

    max_offset = max(offsets)
    offset_fraction = max_offset
    offset_detected = max_offset > offset_threshold

    flag = "OFFSET_DETECTED" if offset_detected else "OK"

    return FluxOffsetResult(
        sector_means=sector_means,
        grand_mean=grand_mean,
        max_offset=max_offset,
        offset_fraction=offset_fraction,
        offset_detected=offset_detected,
        flag=flag,
    )


def format_flux_offset_result(result: FluxOffsetResult) -> str:
    """Return a Markdown table summarising the flux offset detection result."""
    means_str = (
        ", ".join(f"{m:.6f}" for m in result.sector_means)
        if result.sector_means
        else "N/A"
    )
    lines = [
        "| Metric | Value |",
        "| --- | --- |",
        f"| Sector Means | {means_str} |",
        f"| Grand Mean | {result.grand_mean:.6f} |",
        f"| Max Offset | {result.max_offset:.6f} |",
        f"| Offset Fraction | {result.offset_fraction:.6f} |",
        f"| Offset Detected | {result.offset_detected} |",
        f"| Flag | {result.flag} |",
    ]
    return "\n".join(lines)


def _cli() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Detect flux-level offsets between TESS sectors. "
            "Provide sector fluxes as a JSON array of arrays."
        )
    )
    parser.add_argument(
        "sector_fluxes",
        type=str,
        metavar="JSON",
        help=(
            "JSON array of per-sector flux arrays, "
            "e.g. '[[1.0,1.0],[1.01,1.01]]'."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.005,
        metavar="FRAC",
        help="Relative offset threshold (default 0.005).",
    )
    args = parser.parse_args()
    try:
        sector_fluxes = json.loads(args.sector_fluxes)
    except json.JSONDecodeError as exc:
        print(f"Error: invalid JSON: {exc}")
        return 1
    result = detect_flux_offset(sector_fluxes, offset_threshold=args.threshold)
    print(format_flux_offset_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
