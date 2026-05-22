"""Compute expected transit depth from planet/star radii and dilution.

The geometric transit depth is (Rp/R★)².  When the aperture is diluted by
neighbour stars the observed depth is shallower.  This module computes both
the undiluted and observed expected depths, and compares them to a measured
depth if provided.

Public API
----------
ExpectedDepthResult(rp_rstar_ratio, geometric_depth_ppm, diluted_depth_ppm,
                    observed_depth_ppm, depth_ratio, flag)
compute_expected_depth(rp_rsun, rstar_rsun, *, dilution_factor,
                       observed_depth_ppm) -> ExpectedDepthResult
format_expected_depth_result(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExpectedDepthResult:
    rp_rstar_ratio: float
    geometric_depth_ppm: float        # (Rp/R★)² × 10⁶
    diluted_depth_ppm: float          # geometric_depth × dilution_factor
    observed_depth_ppm: float | None  # from measurement (if provided)
    depth_ratio: float | None         # observed / diluted — should be ~1
    flag: str  # "OK" | "INVALID"


def compute_expected_depth(
    rp_rsun: float,
    rstar_rsun: float,
    *,
    dilution_factor: float = 1.0,
    observed_depth_ppm: float | None = None,
) -> ExpectedDepthResult:
    """Compute expected transit depth.

    Args:
        rp_rsun: Planet radius in solar radii.
        rstar_rsun: Stellar radius in solar radii.
        dilution_factor: Fraction of flux from the target star
            (1.0 = uncontaminated; < 1 means contaminated).
        observed_depth_ppm: Measured transit depth in ppm (optional).

    Returns:
        :class:`ExpectedDepthResult`.
    """
    if rp_rsun <= 0 or rstar_rsun <= 0:
        return ExpectedDepthResult(0.0, 0.0, 0.0, None, None, "INVALID")
    if not (0 < dilution_factor <= 1.0):
        return ExpectedDepthResult(0.0, 0.0, 0.0, None, None, "INVALID")

    ratio = rp_rsun / rstar_rsun
    geom_depth = ratio ** 2 * 1e6
    diluted = geom_depth * dilution_factor

    depth_ratio: float | None = None
    if observed_depth_ppm is not None and diluted > 1e-6:
        depth_ratio = round(observed_depth_ppm / diluted, 4)

    return ExpectedDepthResult(
        rp_rstar_ratio=round(ratio, 6),
        geometric_depth_ppm=round(geom_depth, 2),
        diluted_depth_ppm=round(diluted, 2),
        observed_depth_ppm=observed_depth_ppm,
        depth_ratio=depth_ratio,
        flag="OK",
    )


def format_expected_depth_result(result: ExpectedDepthResult) -> str:
    """Format expected depth result as Markdown."""
    lines = [
        "## Expected Transit Depth",
        "",
        f"- Rp/R★: {result.rp_rstar_ratio:.5f}",
        f"- Geometric depth: {result.geometric_depth_ppm:.1f} ppm",
        f"- Diluted depth: {result.diluted_depth_ppm:.1f} ppm",
        f"- Observed depth: {result.observed_depth_ppm} ppm",
        f"- Depth ratio (obs/expected): {result.depth_ratio}",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="expected_depth_calculator",
        description="Compute expected transit depth from Rp and R★.",
    )
    parser.add_argument("rp_rsun", type=float)
    parser.add_argument("rstar_rsun", type=float)
    parser.add_argument("--dilution", type=float, default=1.0)
    parser.add_argument("--observed-depth-ppm", type=float, default=None)
    args = parser.parse_args(argv)

    result = compute_expected_depth(
        args.rp_rsun, args.rstar_rsun,
        dilution_factor=args.dilution,
        observed_depth_ppm=args.observed_depth_ppm,
    )
    print(format_expected_depth_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
