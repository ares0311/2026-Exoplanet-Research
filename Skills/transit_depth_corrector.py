"""Correct transit depths for flux dilution from nearby stars.

A measured transit depth is diluted when contaminating flux from a nearby star
falls in the photometric aperture. The corrected depth is deeper.

Public API
----------
DepthCorrectionResult(measured_depth_ppm, corrected_depth_ppm, dilution_factor,
                      contamination_ratio, correction_applied)
correct_transit_depth(measured_depth_ppm, *, contamination_ratio, crowdsap) -> DepthCorrectionResult
format_depth_correction(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DepthCorrectionResult:
    measured_depth_ppm: float
    corrected_depth_ppm: float
    dilution_factor: float      # 0–1; fraction of flux from target star
    contamination_ratio: float  # total contamination / target flux
    correction_applied: bool    # False if contamination unknown or zero


def correct_transit_depth(
    measured_depth_ppm: float,
    *,
    contamination_ratio: float | None = None,
    crowdsap: float | None = None,
) -> DepthCorrectionResult:
    """Correct a measured transit depth for flux dilution.

    Provide *either* ``contamination_ratio`` (neighbour flux / target flux)
    *or* ``crowdsap`` (TESS CROWDSAP keyword: fraction of aperture flux from
    the target).  If both are given, ``crowdsap`` takes precedence.

    The correction is:
        dilution_factor = 1 / (1 + contamination_ratio)
        corrected_depth = measured_depth / dilution_factor

    Args:
        measured_depth_ppm: Observed transit depth in ppm.
        contamination_ratio: Ratio of contaminating flux to target flux.
        crowdsap: TESS CROWDSAP header value (target fraction; 0–1).

    Returns:
        :class:`DepthCorrectionResult`.
    """
    if crowdsap is not None:
        crowdsap_clamped = max(0.0, min(1.0, float(crowdsap)))
        if crowdsap_clamped <= 0.0:
            dilution = 0.0
            cont = float("inf")
        else:
            dilution = crowdsap_clamped
            cont = (1.0 - crowdsap_clamped) / crowdsap_clamped
    elif contamination_ratio is not None:
        cont = max(0.0, float(contamination_ratio))
        dilution = 1.0 / (1.0 + cont)
    else:
        return DepthCorrectionResult(
            measured_depth_ppm=measured_depth_ppm,
            corrected_depth_ppm=measured_depth_ppm,
            dilution_factor=1.0,
            contamination_ratio=0.0,
            correction_applied=False,
        )

    if dilution <= 0.0 or cont == 0.0:
        corrected = measured_depth_ppm
        applied = False
    else:
        corrected = measured_depth_ppm / dilution
        applied = True

    return DepthCorrectionResult(
        measured_depth_ppm=measured_depth_ppm,
        corrected_depth_ppm=round(corrected, 3),
        dilution_factor=round(dilution, 6),
        contamination_ratio=round(cont, 6),
        correction_applied=applied,
    )


def format_depth_correction(result: DepthCorrectionResult) -> str:
    """Format depth correction result as Markdown."""
    lines = [
        "## Transit Depth Dilution Correction",
        "",
        f"- Measured depth: {result.measured_depth_ppm:.1f} ppm",
        f"- Dilution factor: {result.dilution_factor:.4f}",
        f"- Contamination ratio: {result.contamination_ratio:.4f}",
    ]
    if result.correction_applied:
        lines.append(f"- Corrected depth: **{result.corrected_depth_ppm:.1f} ppm**")
    else:
        lines.append("- Correction: not applied (no contamination data)")
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="transit_depth_corrector",
        description="Correct transit depth for flux dilution.",
    )
    parser.add_argument("depth", type=float, help="Measured depth in ppm.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--contamination-ratio", type=float, default=None)
    group.add_argument("--crowdsap", type=float, default=None)
    args = parser.parse_args(argv)

    result = correct_transit_depth(
        args.depth,
        contamination_ratio=args.contamination_ratio,
        crowdsap=args.crowdsap,
    )
    print(format_depth_correction(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
