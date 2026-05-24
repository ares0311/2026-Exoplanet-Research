"""Score the quality of a phase-folded transit light curve.

Assesses phase coverage inside the transit window, ingress/egress symmetry,
and the signal-to-noise ratio of the phase-folded transit.  Distinct from
``phase_coverage_checker`` (raw phase-bin coverage) and ``scatter_metric_calculator``
(global scatter metrics) — this specifically evaluates the transit window.

Public API
----------
PhaseFoldQualityResult(n_points, n_bins, n_bins_in_transit,
                       coverage_fraction, transit_snr, symmetry_score,
                       quality_grade, flag)
check_phase_fold_quality(phase, flux, *, transit_width_phase,
                          n_bins) -> PhaseFoldQualityResult
format_phase_fold_quality(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PhaseFoldQualityResult:
    n_points: int
    n_bins: int
    n_bins_in_transit: int
    coverage_fraction: float | None   # filled transit bins / expected transit bins
    transit_snr: float | None         # |mean_in - mean_oot| / std_oot
    symmetry_score: float | None      # 1 - |depth_ingress - depth_egress| / mean_depth
    quality_grade: str | None         # "A" | "B" | "C" | "D"
    flag: str  # "OK" | "INSUFFICIENT" | "INVALID"


def _grade(coverage: float, snr: float) -> str:
    if coverage >= 0.90 and snr >= 5.0:
        return "A"
    if coverage >= 0.75 and snr >= 3.0:
        return "B"
    if coverage >= 0.50 and snr >= 2.0:
        return "C"
    return "D"


def check_phase_fold_quality(
    phase: list[float],
    flux: list[float],
    *,
    transit_width_phase: float = 0.05,
    n_bins: int = 50,
) -> PhaseFoldQualityResult:
    """Evaluate phase-fold quality for a single transit signal.

    Args:
        phase: Phase array in [-0.5, 0.5).
        flux: Normalised flux array (out-of-transit ≈ 1.0).
        transit_width_phase: Half-width of transit window in phase units.
            The transit window is [-transit_width_phase, +transit_width_phase].
        n_bins: Number of phase bins across [-0.5, 0.5).

    Returns:
        :class:`PhaseFoldQualityResult`.
    """
    if len(phase) != len(flux):
        return PhaseFoldQualityResult(len(phase), n_bins, 0, None, None, None, None, "INVALID")
    if len(phase) < 10:
        return PhaseFoldQualityResult(len(phase), n_bins, 0, None, None, None, None, "INSUFFICIENT")
    if n_bins < 4 or transit_width_phase <= 0:
        return PhaseFoldQualityResult(len(phase), n_bins, 0, None, None, None, None, "INVALID")

    # Bin phase-folded data
    bin_width = 1.0 / n_bins
    bins: dict[int, list[float]] = {}
    for ph, fl in zip(phase, flux, strict=False):
        bi = int((ph + 0.5) / bin_width)
        bi = max(0, min(n_bins - 1, bi))
        bins.setdefault(bi, []).append(fl)

    # Identify transit bins
    transit_half = transit_width_phase
    transit_bin_lo = int((-transit_half + 0.5) / bin_width)
    transit_bin_hi = int((transit_half + 0.5) / bin_width)
    n_expected_transit = max(1, transit_bin_hi - transit_bin_lo)

    transit_bins = [i for i in range(transit_bin_lo, transit_bin_hi + 1) if i in bins]
    n_bins_in_transit = len(transit_bins)
    coverage = n_bins_in_transit / n_expected_transit

    # Out-of-transit flux
    oot_flux = [f for i, fs in bins.items()
                for f in fs if i < transit_bin_lo or i > transit_bin_hi]
    in_flux = [f for i in transit_bins for f in bins[i]]

    if not oot_flux or not in_flux:
        return PhaseFoldQualityResult(
            len(phase), n_bins, n_bins_in_transit, round(coverage, 4),
            None, None, None, "INSUFFICIENT"
        )

    mean_oot = sum(oot_flux) / len(oot_flux)
    mean_in = sum(in_flux) / len(in_flux)
    std_oot = math.sqrt(sum((f - mean_oot) ** 2 for f in oot_flux) / len(oot_flux))

    snr = abs(mean_oot - mean_in) / std_oot if std_oot > 0 else 0.0

    # Symmetry: compare first half vs second half of transit bins
    half = len(transit_bins) // 2
    ing_flux = [f for i in transit_bins[:half] for f in bins[i]]
    egr_flux = [f for i in transit_bins[half:] for f in bins[i]]
    depth = mean_oot - mean_in
    if depth > 0 and ing_flux and egr_flux:
        d_ing = mean_oot - sum(ing_flux) / len(ing_flux)
        d_egr = mean_oot - sum(egr_flux) / len(egr_flux)
        sym = max(0.0, 1.0 - abs(d_ing - d_egr) / depth)
    else:
        sym = None

    grade = _grade(coverage, snr)

    return PhaseFoldQualityResult(
        n_points=len(phase),
        n_bins=n_bins,
        n_bins_in_transit=n_bins_in_transit,
        coverage_fraction=round(coverage, 4),
        transit_snr=round(snr, 3),
        symmetry_score=round(sym, 4) if sym is not None else None,
        quality_grade=grade,
        flag="OK",
    )


def format_phase_fold_quality(result: PhaseFoldQualityResult) -> str:
    """Format phase-fold quality result as Markdown."""
    lines = [
        "## Phase-Fold Quality Checker",
        "",
        f"- Data points: {result.n_points}",
        f"- Bins in transit: {result.n_bins_in_transit} / {result.n_bins}",
        f"- **Coverage fraction: {result.coverage_fraction}**",
        f"- **Transit SNR: {result.transit_snr}**",
        f"- Symmetry score: {result.symmetry_score}",
        f"- **Quality grade: {result.quality_grade}**",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="phase_fold_quality_checker",
        description="Score phase-fold quality of a transit light curve.",
    )
    parser.add_argument("--transit-width", type=float, default=0.05)
    parser.add_argument("--n-bins", type=int, default=50)
    args = parser.parse_args(argv)

    result = check_phase_fold_quality([], [], transit_width_phase=args.transit_width,
                                      n_bins=args.n_bins)
    print(format_phase_fold_quality(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
