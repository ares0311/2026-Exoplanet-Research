"""Calculate minimum angular separation to exclude a contaminating source.

A background star within a given contrast ratio can mimic a transit signal if
it is an eclipsing binary.  This module calculates the exclusion zone radius:
the minimum angular separation at which a source of given contrast can be
ruled out as the transit source based on the observed transit depth.

Public API
----------
ExclusionZoneResult(contrast_ratio, depth_ppm, min_separation_arcsec,
                    can_exclude, excluded_by, flag)
compute_exclusion_zone(observed_depth_ppm, *, contrast_ratio,
                       pixel_scale_arcsec, centroid_offset_arcsec,
                       centroid_sigma) -> ExclusionZoneResult
format_exclusion_result(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExclusionZoneResult:
    contrast_ratio: float | None      # F_contaminant / F_target
    depth_ppm: float                  # observed transit depth
    min_separation_arcsec: float | None  # exclusion zone radius
    can_exclude: bool                 # whether source can be excluded
    excluded_by: str                  # "centroid" | "depth" | "none"
    flag: str  # "OK" | "INVALID"


def compute_exclusion_zone(
    observed_depth_ppm: float,
    *,
    contrast_ratio: float | None = None,
    pixel_scale_arcsec: float = 21.0,
    centroid_offset_arcsec: float | None = None,
    centroid_sigma: float | None = None,
    sigma_threshold: float = 3.0,
) -> ExclusionZoneResult:
    """Compute the exclusion zone for a contaminating background source.

    A background eclipsing binary with eclipse depth D_bg and flux ratio
    f = F_bg / F_total would produce an apparent depth of D_obs = f × D_bg.
    For D_bg ≤ 1 (total eclipse), the minimum contrast is D_obs.
    The exclusion zone is expressed in arcsec assuming the centroid shift
    is resolvable.

    Args:
        observed_depth_ppm: Observed transit/eclipse depth in ppm.
        contrast_ratio: F_contaminant/F_target flux ratio (if known).
        pixel_scale_arcsec: Pixel scale in arcsec/pixel.
        centroid_offset_arcsec: Measured centroid offset in arcsec.
        centroid_sigma: Significance of the centroid offset.
        sigma_threshold: Significance threshold for centroid exclusion.

    Returns:
        :class:`ExclusionZoneResult`.
    """
    if observed_depth_ppm <= 0:
        return ExclusionZoneResult(
            contrast_ratio, observed_depth_ppm, None, False, "none", "INVALID"
        )

    depth_frac = observed_depth_ppm / 1e6

    # Minimum flux ratio needed to cause the observed depth (if D_bg = 100%)
    min_contrast = depth_frac  # f_min = D_obs / 1.0

    # Centroid exclusion: if offset is significant, the source can be localised
    excluded_by = "none"
    can_exclude = False

    if (centroid_offset_arcsec is not None and centroid_sigma is not None
            and centroid_sigma >= sigma_threshold):
        excluded_by = "centroid"
        can_exclude = True

    # Depth exclusion: if contrast_ratio implies depth too shallow
    if contrast_ratio is not None and not can_exclude:
        # If the contaminant has contrast_ratio, max depth it can cause is contrast_ratio
        max_depth_from_contam = contrast_ratio / (1.0 + contrast_ratio)
        if max_depth_from_contam < depth_frac * 0.9:
            excluded_by = "depth"
            can_exclude = True

    # Exclusion zone radius: where a source of min_contrast would be detected
    # by centroid motion of >= 1 pixel
    if min_contrast > 0:
        # Expected centroid offset if source at separation r:
        # delta_cen = f_contam / (1 + f_contam) * separation
        # Set delta_cen = 1 pixel → r = pixel_scale / (f_min / (1 + f_min))
        weight = min_contrast / (1.0 + min_contrast)
        min_sep = pixel_scale_arcsec / weight if weight > 1e-9 else None
    else:
        min_sep = None

    return ExclusionZoneResult(
        contrast_ratio=contrast_ratio,
        depth_ppm=observed_depth_ppm,
        min_separation_arcsec=round(min_sep, 2) if min_sep is not None else None,
        can_exclude=can_exclude,
        excluded_by=excluded_by,
        flag="OK",
    )


def format_exclusion_result(result: ExclusionZoneResult) -> str:
    """Format exclusion zone result as Markdown."""
    lines = [
        "## Exclusion Zone Calculator",
        "",
        f"- Observed depth: {result.depth_ppm:.1f} ppm",
        f"- Contrast ratio (F_contam/F_target): {result.contrast_ratio}",
        f"- Exclusion zone radius: {result.min_separation_arcsec} arcsec",
        f"- Can exclude background source: {'Yes' if result.can_exclude else 'No'}",
        f"- Excluded by: {result.excluded_by}",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="exclusion_zone_calculator",
        description="Calculate minimum separation to exclude a background source.",
    )
    parser.add_argument("depth_ppm", type=float)
    parser.add_argument("--contrast-ratio", type=float, default=None)
    parser.add_argument("--pixel-scale", type=float, default=21.0)
    args = parser.parse_args(argv)

    result = compute_exclusion_zone(args.depth_ppm, contrast_ratio=args.contrast_ratio,
                                    pixel_scale_arcsec=args.pixel_scale)
    print(format_exclusion_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
