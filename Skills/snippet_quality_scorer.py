"""Score normalized CNN light-curve snippets for quality metrics.

Evaluates per-snippet coverage, depth SNR, and out-of-transit noise to
produce a composite quality score and OK/INSUFFICIENT/INVALID flag.

Public API
----------
SnippetQualityResult
score_snippet_quality(flux, *, min_coverage, transit_phase_half_width) -> SnippetQualityResult
score_snippet_batch(snippets, **kwargs) -> list[SnippetQualityResult]
format_snippet_quality(result) -> str
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class SnippetQualityResult:
    n_bins: int
    n_populated: int
    coverage_fraction: float
    depth_snr: float | None
    oot_noise: float | None
    in_transit_dip: float | None
    quality_score: float
    flag: str  # "OK" | "INSUFFICIENT" | "INVALID"


def score_snippet_quality(
    flux: list[float | None],
    *,
    min_coverage: float = 0.70,
    transit_phase_half_width: float = 0.15,
) -> SnippetQualityResult:
    """Score a single normalized CNN snippet for quality.

    Args:
        flux: 201-element normalized flux array (None = missing bin).
        min_coverage: Below this coverage_fraction → INSUFFICIENT.
        transit_phase_half_width: Phase range [-w, w] defines in-transit bins.

    Returns:
        :class:`SnippetQualityResult`.
    """
    n_bins = len(flux)
    if n_bins == 0:
        return SnippetQualityResult(
            n_bins=0,
            n_populated=0,
            coverage_fraction=0.0,
            depth_snr=None,
            oot_noise=None,
            in_transit_dip=None,
            quality_score=0.0,
            flag="INVALID",
        )

    # Count populated bins (non-None and finite)
    populated = [f for f in flux if f is not None and math.isfinite(f)]
    n_populated = len(populated)
    coverage_fraction = n_populated / n_bins

    # Phases run from -0.5 to +0.5, evenly spaced over n_bins
    # Phase of bin i: -0.5 + (i + 0.5) / n_bins
    in_transit_vals: list[float] = []
    oot_vals: list[float] = []
    for i, f in enumerate(flux):
        if f is None or not math.isfinite(f):
            continue
        phase = -0.5 + (i + 0.5) / n_bins
        if abs(phase) <= transit_phase_half_width:
            in_transit_vals.append(f)
        else:
            oot_vals.append(f)

    # OOT statistics
    oot_noise: float | None = None
    depth_snr: float | None = None
    in_transit_dip: float | None = None

    if len(oot_vals) >= 2:
        oot_noise = statistics.stdev(oot_vals)
        oot_median = statistics.median(oot_vals)

        if in_transit_vals:
            it_median = statistics.median(in_transit_vals)
            in_transit_dip = it_median - oot_median  # negative = dip

            min_flux = min(in_transit_vals)
            if oot_noise > 0:
                depth_snr = abs(min_flux - oot_median) / oot_noise

    # Composite quality score [0, 1]
    # coverage * 0.4 + snr_sub * 0.4 + (1 - noise_sub) * 0.2
    snr_sub = 0.0
    if depth_snr is not None:
        snr_sub = min(depth_snr / 10.0, 1.0)  # saturate at SNR = 10

    noise_sub = 0.0
    if oot_noise is not None:
        noise_sub = min(oot_noise / 0.1, 1.0)  # saturate at noise = 0.1

    quality_score = (
        coverage_fraction * 0.4
        + snr_sub * 0.4
        + (1.0 - noise_sub) * 0.2
    )
    quality_score = max(0.0, min(1.0, quality_score))

    # Determine flag
    flag = "INSUFFICIENT" if coverage_fraction < min_coverage else "OK"

    return SnippetQualityResult(
        n_bins=n_bins,
        n_populated=n_populated,
        coverage_fraction=round(coverage_fraction, 6),
        depth_snr=round(depth_snr, 4) if depth_snr is not None else None,
        oot_noise=round(oot_noise, 6) if oot_noise is not None else None,
        in_transit_dip=round(in_transit_dip, 6) if in_transit_dip is not None else None,
        quality_score=round(quality_score, 6),
        flag=flag,
    )


def score_snippet_batch(
    snippets: list[list[float | None]],
    **kwargs,
) -> list[SnippetQualityResult]:
    """Score a batch of snippets.

    Args:
        snippets: List of flux arrays.
        **kwargs: Passed to :func:`score_snippet_quality`.

    Returns:
        List of :class:`SnippetQualityResult`.
    """
    return [score_snippet_quality(s, **kwargs) for s in snippets]


def format_snippet_quality(result: SnippetQualityResult) -> str:
    """Format snippet quality result as Markdown."""
    lines = [
        "## Snippet Quality Scorer",
        "",
        f"- Bins: {result.n_bins} total, {result.n_populated} populated",
        f"- Coverage fraction: {result.coverage_fraction:.3f}",
        f"- Depth SNR: {result.depth_snr}",
        f"- OOT noise: {result.oot_noise}",
        f"- In-transit dip: {result.in_transit_dip}",
        f"- **Quality score: {result.quality_score:.4f}**",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="snippet_quality_scorer",
        description="Score a normalized CNN snippet for quality metrics.",
    )
    parser.add_argument(
        "flux_json",
        help="JSON array of flux values (use null for missing bins).",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.70,
    )
    args = parser.parse_args(argv)

    flux = json.loads(args.flux_json)
    result = score_snippet_quality(flux, min_coverage=args.min_coverage)
    print(format_snippet_quality(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
