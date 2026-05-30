"""Multi-aperture transit depth checker.

Compares transit depths measured in different photometric apertures.  A large
depth scatter relative to the mean depth suggests flux contamination from a
nearby source that falls in some apertures but not others.

Public API
----------
ApertureDepthResult(depths, mean_depth, depth_rms, contamination_flag, flag)
check_aperture_depths(aperture_depths_ppm, *, contamination_threshold=0.15)
    -> ApertureDepthResult
format_aperture_depth_result(result) -> str
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ApertureDepthResult:
    depths: list[float]
    mean_depth: float
    depth_rms: float
    contamination_flag: bool
    flag: str


def check_aperture_depths(
    aperture_depths_ppm: list[float],
    *,
    contamination_threshold: float = 0.15,
) -> ApertureDepthResult:
    """Check consistency of transit depths across multiple apertures.

    Parameters
    ----------
    aperture_depths_ppm:     transit depths in ppm for each aperture
    contamination_threshold: relative RMS threshold above which contamination
                             is suspected (default 0.15 = 15 %)

    Returns
    -------
    ApertureDepthResult with flag one of:
    INSUFFICIENT_DATA, CONTAMINATION_SUSPECTED, OK
    """
    if len(aperture_depths_ppm) < 2:
        return ApertureDepthResult(
            depths=list(aperture_depths_ppm),
            mean_depth=0.0,
            depth_rms=0.0,
            contamination_flag=False,
            flag="INSUFFICIENT_DATA",
        )

    depths = list(aperture_depths_ppm)
    n = len(depths)
    mean_depth = sum(depths) / n
    depth_rms = math.sqrt(sum((d - mean_depth) ** 2 for d in depths) / n)

    relative_rms = depth_rms / mean_depth if mean_depth > 0.0 else 0.0
    contamination_flag = relative_rms > contamination_threshold
    flag = "CONTAMINATION_SUSPECTED" if contamination_flag else "OK"

    return ApertureDepthResult(
        depths=depths,
        mean_depth=mean_depth,
        depth_rms=depth_rms,
        contamination_flag=contamination_flag,
        flag=flag,
    )


def format_aperture_depth_result(result: ApertureDepthResult) -> str:
    """Return a Markdown table summarising the aperture depth check."""
    depths_str = ", ".join(f"{d:.1f}" for d in result.depths)
    lines = [
        "| Metric | Value |",
        "| --- | --- |",
        f"| Depths (ppm) | {depths_str} |",
        f"| Mean Depth (ppm) | {result.mean_depth:.2f} |",
        f"| Depth RMS (ppm) | {result.depth_rms:.2f} |",
        f"| Contamination Flag | {result.contamination_flag} |",
        f"| Flag | {result.flag} |",
    ]
    return "\n".join(lines)


def _cli() -> int:
    parser = argparse.ArgumentParser(
        description="Check transit depth consistency across multiple apertures."
    )
    parser.add_argument(
        "depths",
        nargs="+",
        type=float,
        metavar="DEPTH_PPM",
        help="Transit depth in ppm for each aperture.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.15,
        metavar="FRAC",
        help="Relative RMS threshold for contamination flag (default 0.15).",
    )
    args = parser.parse_args()
    result = check_aperture_depths(args.depths, contamination_threshold=args.threshold)
    print(format_aperture_depth_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
