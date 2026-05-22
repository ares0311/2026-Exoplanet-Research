"""Analyse residuals of a phase-folded light curve after transit model subtraction.

After subtracting a box-transit model from the phase-folded light curve the
residuals should be consistent with Gaussian noise if the transit model is
adequate.  This module bins the residuals, checks for systematics, and
computes scatter metrics.

Public API
----------
FoldedResidualResult(n_cadences, rms_residual, mad_residual, skewness,
                     excess_kurtosis, chi2_reduced, is_gaussian, flag)
analyze_folded_residuals(phase, flux, *, flux_err, n_bins,
                         depth_ppm, half_width_phase) -> FoldedResidualResult
format_residual_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FoldedResidualResult:
    n_cadences: int
    rms_residual: float
    mad_residual: float
    skewness: float | None
    excess_kurtosis: float | None
    chi2_reduced: float | None  # per-bin chi² / (n_bins - 1)
    is_gaussian: bool
    flag: str  # "OK" | "INSUFFICIENT" | "INVALID"


def _subtract_box(
    phase: list[float],
    flux: list[float],
    depth_frac: float,
    half_width: float,
) -> list[float]:
    """Subtract a simple box model from the flux."""
    return [
        flux[i] - (-depth_frac if abs(phase[i]) <= half_width else 0.0)
        for i in range(len(flux))
    ]


def analyze_folded_residuals(
    phase: list[float],
    flux: list[float],
    *,
    flux_err: list[float] | None = None,
    n_bins: int = 20,
    depth_ppm: float = 0.0,
    half_width_phase: float = 0.05,
) -> FoldedResidualResult:
    """Analyse residuals after subtracting a box transit model.

    Args:
        phase: Phase values in [-0.5, 0.5).
        flux: Normalised flux array (same length as phase).
        flux_err: Per-point uncertainties (optional).
        n_bins: Number of phase bins for chi² calculation.
        depth_ppm: Transit depth in ppm (used to build box model).
        half_width_phase: Half-width of the box model in phase units.

    Returns:
        :class:`FoldedResidualResult`.
    """
    n = len(flux)
    if n < 5 or len(phase) != n:
        return FoldedResidualResult(n, 0.0, 0.0, None, None, None, False, "INVALID")

    depth_frac = depth_ppm * 1e-6
    residuals = _subtract_box(phase, flux, depth_frac, half_width_phase)

    # Basic statistics
    mean_r = sum(residuals) / n
    rms = math.sqrt(sum((r - mean_r) ** 2 for r in residuals) / n)

    sorted_r = sorted(residuals)
    med = sorted_r[n // 2] if n % 2 == 1 else (sorted_r[n // 2 - 1] + sorted_r[n // 2]) / 2.0
    devs = sorted(abs(r - med) for r in residuals)
    mad = devs[n // 2] if n % 2 == 1 else (devs[n // 2 - 1] + devs[n // 2]) / 2.0

    skew: float | None = None
    kurt: float | None = None
    if rms > 1e-12 and n >= 10:
        skew = sum(((r - mean_r) / rms) ** 3 for r in residuals) / n
        kurt = sum(((r - mean_r) / rms) ** 4 for r in residuals) / n - 3.0

    # Binned chi²
    chi2: float | None = None
    if flux_err is not None and len(flux_err) == n and n_bins >= 3:
        bins: list[list[tuple[float, float]]] = [[] for _ in range(n_bins)]
        for ph, r, e in zip(phase, residuals, flux_err, strict=False):
            idx = min(int((ph + 0.5) * n_bins), n_bins - 1)
            bins[idx].append((r, e))
        non_empty = [b for b in bins if b]
        if non_empty:
            chi2_sum = 0.0
            dof = 0
            for b in non_empty:
                if len(b) < 2:
                    continue
                bin_mean = sum(r for r, _ in b) / len(b)
                bin_err2 = sum(e ** 2 for _, e in b) / len(b) ** 2
                if bin_err2 > 1e-30:
                    chi2_sum += bin_mean ** 2 / bin_err2
                    dof += 1
            if dof > 1:
                chi2 = round(chi2_sum / (dof - 1), 4)

    # Gaussian check: |skewness| < 1.0 and |excess_kurtosis| < 3.0
    is_gauss = True
    if skew is not None and abs(skew) >= 1.0:
        is_gauss = False
    if kurt is not None and abs(kurt) >= 3.0:
        is_gauss = False

    flag = "OK" if n >= 10 else "INSUFFICIENT"

    return FoldedResidualResult(
        n_cadences=n,
        rms_residual=round(rms, 8),
        mad_residual=round(mad, 8),
        skewness=round(skew, 4) if skew is not None else None,
        excess_kurtosis=round(kurt, 4) if kurt is not None else None,
        chi2_reduced=chi2,
        is_gaussian=is_gauss,
        flag=flag,
    )


def format_residual_result(result: FoldedResidualResult) -> str:
    """Format folded residual analysis result as Markdown."""
    lines = [
        "## Folded Residual Analysis",
        "",
        f"- Cadences: {result.n_cadences}",
        f"- RMS residual: {result.rms_residual:.3e}",
        f"- MAD residual: {result.mad_residual:.3e}",
        f"- Skewness: {result.skewness}",
        f"- Excess kurtosis: {result.excess_kurtosis}",
        f"- χ² reduced: {result.chi2_reduced}",
        f"- Gaussian-like: {'Yes' if result.is_gaussian else 'No'}",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="folded_residual_analyzer",
        description="Analyse phase-folded light curve residuals.",
    )
    parser.add_argument("--depth-ppm", type=float, default=0.0)
    parser.add_argument("--half-width", type=float, default=0.05)
    args = parser.parse_args(argv)

    result = analyze_folded_residuals(
        [], [],
        depth_ppm=args.depth_ppm,
        half_width_phase=args.half_width,
    )
    print(format_residual_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
