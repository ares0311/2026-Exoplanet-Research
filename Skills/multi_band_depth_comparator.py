"""Compare transit depths across multiple photometric bands for chromaticity.

A transit depth that varies significantly with wavelength indicates stellar
contamination (e.g. a background EB or a spots/faculae scenario) rather than
a genuine planet.  Distinct from ``transit_depth_corrector`` (applies a single
dilution correction) and ``odd_even_analyzer`` (compares odd/even depths).

Public API
----------
MultiBandDepthResult(n_bands, band_names, depths_ppm, depth_errors_ppm,
                     weighted_mean_depth_ppm, max_fractional_difference,
                     is_chromatic, reference_band, flag)
compare_multi_band_depths(band_names, depths_ppm, depth_errors_ppm, *,
                          chromatic_threshold) -> MultiBandDepthResult
format_multi_band_result(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MultiBandDepthResult:
    n_bands: int
    band_names: tuple[str, ...]
    depths_ppm: tuple[float, ...]
    depth_errors_ppm: tuple[float, ...]
    weighted_mean_depth_ppm: float | None
    max_fractional_difference: float | None   # max |d_i - d_ref| / d_ref
    is_chromatic: bool
    reference_band: str | None               # band with smallest error
    flag: str  # "OK" | "INSUFFICIENT" | "INVALID"


def compare_multi_band_depths(
    band_names: list[str],
    depths_ppm: list[float],
    depth_errors_ppm: list[float],
    *,
    chromatic_threshold: float = 0.10,
) -> MultiBandDepthResult:
    """Test whether transit depth varies significantly with photometric band.

    Args:
        band_names: Names of photometric bands (e.g. ``["B", "V", "R"]``).
        depths_ppm: Transit depths in each band (ppm).
        depth_errors_ppm: 1-σ depth uncertainties (ppm).
        chromatic_threshold: Fractional depth difference threshold above which
            the transit is flagged as chromatic (default 0.10 = 10%).

    Returns:
        :class:`MultiBandDepthResult`.
    """
    n = len(band_names)
    if n != len(depths_ppm) or n != len(depth_errors_ppm):
        return MultiBandDepthResult(n, (), (), (), None, None, False, None, "INVALID")
    if n < 2:
        return MultiBandDepthResult(
            n, tuple(band_names), tuple(depths_ppm),
            tuple(depth_errors_ppm), None, None, False, None, "INSUFFICIENT"
        )
    if any(e <= 0 for e in depth_errors_ppm):
        return MultiBandDepthResult(n, tuple(band_names), tuple(depths_ppm),
                                    tuple(depth_errors_ppm), None, None, False, None, "INVALID")

    # Inverse-variance weighted mean
    weights = [1.0 / e ** 2 for e in depth_errors_ppm]
    w_sum = sum(weights)
    d_mean = sum(w * d for w, d in zip(weights, depths_ppm, strict=False)) / w_sum

    if d_mean <= 0:
        return MultiBandDepthResult(n, tuple(band_names), tuple(depths_ppm),
                                    tuple(depth_errors_ppm), round(d_mean, 4),
                                    None, False, None, "INVALID")

    # Reference band = smallest error
    ref_idx = min(range(n), key=lambda i: depth_errors_ppm[i])
    ref_band = band_names[ref_idx]

    # Max fractional difference from weighted mean
    max_frac = max(abs(d - d_mean) / d_mean for d in depths_ppm)
    is_chrom = max_frac > chromatic_threshold

    return MultiBandDepthResult(
        n_bands=n,
        band_names=tuple(band_names),
        depths_ppm=tuple(depths_ppm),
        depth_errors_ppm=tuple(depth_errors_ppm),
        weighted_mean_depth_ppm=round(d_mean, 4),
        max_fractional_difference=round(max_frac, 6),
        is_chromatic=is_chrom,
        reference_band=ref_band,
        flag="OK",
    )


def format_multi_band_result(result: MultiBandDepthResult) -> str:
    """Format multi-band depth comparison as Markdown."""
    chrom_label = "**CHROMATIC — possible contamination**" if result.is_chromatic else "Achromatic"
    lines = [
        "## Multi-Band Depth Comparator",
        "",
        f"- Bands: {', '.join(result.band_names)}",
        f"- Weighted mean depth: {result.weighted_mean_depth_ppm} ppm",
        f"- Max fractional difference: {result.max_fractional_difference}",
        f"- **Chromatic: {chrom_label}**",
        f"- Reference band: {result.reference_band}",
        f"- **Flag: {result.flag}**",
    ]
    if result.band_names:
        lines += ["", "| Band | Depth (ppm) | Err (ppm) |", "|---|---|---|"]
        for name, d, e in zip(result.band_names, result.depths_ppm,
                               result.depth_errors_ppm, strict=False):
            lines.append(f"| {name} | {d:.1f} | {e:.1f} |")
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="multi_band_depth_comparator",
        description="Compare transit depths across photometric bands.",
    )
    parser.add_argument("--threshold", type=float, default=0.10)
    args = parser.parse_args(argv)

    result = compare_multi_band_depths(["V", "R"], [1000.0, 1050.0], [20.0, 25.0],
                                        chromatic_threshold=args.threshold)
    print(format_multi_band_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
