"""Compare photometric properties between two aperture extractions.

A discrepancy in transit depth between a small (target-only) aperture and a
large (contaminated) aperture is diagnostic of a nearby contaminating source.
This module quantifies depth, RMS, and scatter differences.

Public API
----------
ApertureCompareResult(depth_discrepancy_frac, rms_ratio, scatter_ratio,
                      depth_discrepant, rms_discrepant, contamination_index,
                      flag)
compare_apertures(aperture_a, aperture_b, *, depth_tol_frac,
                  rms_tol_frac) -> ApertureCompareResult
format_aperture_compare_result(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ApertureCompareResult:
    depth_discrepancy_frac: float | None  # |depth_a - depth_b| / max(depth_a, depth_b)
    rms_ratio: float | None               # rms_b / rms_a  (>1 means b noisier)
    scatter_ratio: float | None           # scatter_b / scatter_a
    depth_discrepant: bool
    rms_discrepant: bool
    contamination_index: float | None     # composite 0-1 score
    flag: str  # "OK" | "INVALID"


def compare_apertures(
    aperture_a: dict,
    aperture_b: dict,
    *,
    depth_tol_frac: float = 0.10,
    rms_tol_frac: float = 0.20,
) -> ApertureCompareResult:
    """Compare two aperture photometry extractions.

    Each aperture dict should contain one or more of:
    ``depth_ppm``, ``rms_ppm``, ``scatter_ppm``.

    Aperture A is treated as the reference (smaller/cleaner aperture).

    Args:
        aperture_a: Reference aperture dict.
        aperture_b: Comparison aperture dict.
        depth_tol_frac: Fractional depth difference threshold.
        rms_tol_frac: Fractional RMS difference threshold.

    Returns:
        :class:`ApertureCompareResult`.
    """
    if not isinstance(aperture_a, dict) or not isinstance(aperture_b, dict):
        return ApertureCompareResult(None, None, None, False, False, None, "INVALID")

    da = aperture_a.get("depth_ppm")
    db = aperture_b.get("depth_ppm")
    rms_a = aperture_a.get("rms_ppm")
    rms_b = aperture_b.get("rms_ppm")
    sca_a = aperture_a.get("scatter_ppm")
    sca_b = aperture_b.get("scatter_ppm")

    # Depth discrepancy
    depth_disc: float | None = None
    depth_discrepant = False
    if da is not None and db is not None and da > 0 and db > 0:
        ref = max(da, db)
        depth_disc = round(abs(da - db) / ref, 6)
        depth_discrepant = depth_disc > depth_tol_frac

    # RMS ratio
    rms_rat: float | None = None
    rms_discrepant = False
    if rms_a is not None and rms_b is not None and rms_a > 0:
        rms_rat = round(rms_b / rms_a, 4)
        rms_discrepant = abs(rms_rat - 1.0) > rms_tol_frac

    # Scatter ratio
    sca_rat: float | None = None
    if sca_a is not None and sca_b is not None and sca_a > 0:
        sca_rat = round(sca_b / sca_a, 4)

    # Contamination index: composite of depth discrepancy and rms excess
    ci: float | None = None
    ci_parts: list[float] = []
    if depth_disc is not None:
        ci_parts.append(min(1.0, depth_disc / max(depth_tol_frac, 1e-9)))
    if rms_rat is not None:
        ci_parts.append(min(1.0, max(0.0, rms_rat - 1.0) / max(rms_tol_frac, 1e-9)))
    if ci_parts:
        ci = round(sum(ci_parts) / len(ci_parts), 4)

    if da is None and db is None and rms_a is None and rms_b is None:
        return ApertureCompareResult(None, None, None, False, False, None, "INVALID")

    return ApertureCompareResult(
        depth_discrepancy_frac=depth_disc,
        rms_ratio=rms_rat,
        scatter_ratio=sca_rat,
        depth_discrepant=depth_discrepant,
        rms_discrepant=rms_discrepant,
        contamination_index=ci,
        flag="OK",
    )


def format_aperture_compare_result(result: ApertureCompareResult) -> str:
    """Format aperture comparison result as Markdown."""
    lines = [
        "## Multi-Aperture Comparator",
        "",
        f"- Depth discrepancy: {result.depth_discrepancy_frac}",
        f"- Depth discrepant: {'Yes' if result.depth_discrepant else 'No'}",
        f"- RMS ratio (B/A): {result.rms_ratio}",
        f"- RMS discrepant: {'Yes' if result.rms_discrepant else 'No'}",
        f"- Scatter ratio (B/A): {result.scatter_ratio}",
        f"- Contamination index: {result.contamination_index}",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="multi_aperture_comparator",
        description="Compare photometric properties across two aperture extractions.",
    )
    parser.add_argument("--depth-a", type=float, default=None)
    parser.add_argument("--depth-b", type=float, default=None)
    args = parser.parse_args(argv)

    a = {}
    b = {}
    if args.depth_a:
        a["depth_ppm"] = args.depth_a
    if args.depth_b:
        b["depth_ppm"] = args.depth_b

    result = compare_apertures(a, b)
    print(format_aperture_compare_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
