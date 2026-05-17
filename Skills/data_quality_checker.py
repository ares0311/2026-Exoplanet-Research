"""Pre-pipeline light-curve quality assessment.

Evaluates a light curve for common pathologies before investing full
pipeline time.  Returns a QualityReport with a letter grade (A–D) and
machine-readable reason codes.

Public API
----------
check_data_quality(time, flux, flux_err, *, cadence_seconds) -> QualityReport
format_quality_report(report) -> str
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class QualityReport:
    grade: str                           # "A", "B", "C", "D"
    outlier_fraction: float
    gap_fraction: float
    scatter_to_noise_ratio: float        # measured scatter / Poisson floor (1.0 = ideal)
    cadence_regularity: float            # fraction of cadences within 10% of median spacing
    n_cadences: int
    reason_codes: list[str] = field(default_factory=list)


def check_data_quality(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray | None = None,
    *,
    cadence_seconds: float = 120.0,
    outlier_sigma: float = 5.0,
    gap_threshold_factor: float = 3.0,
) -> QualityReport:
    """Assess light-curve quality.

    Args:
        time: Time array (any consistent unit, e.g. BJD days).
        flux: Relative or absolute flux array.
        flux_err: Per-point flux uncertainties.  If ``None``, uses the
            median absolute deviation to estimate noise.
        cadence_seconds: Expected cadence in seconds (for regularity check).
        outlier_sigma: Points beyond this many σ from median are outliers.
        gap_threshold_factor: Time step > this × median step is a gap.

    Returns:
        :class:`QualityReport` with grade A–D.
    """
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    n = len(time)

    reason_codes: list[str] = []

    # --- Outlier fraction ---
    median_f = float(np.median(flux))
    mad = float(np.median(np.abs(flux - median_f))) * 1.4826
    sigma = max(mad, 1e-12)
    n_outliers = int(np.sum(np.abs(flux - median_f) > outlier_sigma * sigma))
    outlier_frac = n_outliers / max(n, 1)
    if outlier_frac > 0.05:
        reason_codes.append("HIGH_OUTLIER_FRACTION")

    # --- Gap fraction ---
    if n > 1:
        diffs = np.diff(time)
        med_step = float(np.median(diffs))
        n_gaps = int(np.sum(diffs > gap_threshold_factor * med_step))
        total_expected = (time[-1] - time[0]) / med_step if med_step > 0 else n
        gap_frac = n_gaps / max(total_expected, 1)
        # Cadence regularity: fraction of steps within 10% of median
        regularity = float(np.mean(np.abs(diffs - med_step) < 0.10 * med_step))
    else:
        gap_frac = 0.0
        regularity = 1.0

    if gap_frac > 0.15:
        reason_codes.append("HIGH_GAP_FRACTION")
    if regularity < 0.80:
        reason_codes.append("IRREGULAR_CADENCE")

    # --- Scatter vs. Poisson floor ---
    if flux_err is not None:
        flux_err = np.asarray(flux_err, dtype=float)
        poisson_floor = float(np.median(flux_err))
        scatter_to_noise = sigma / max(poisson_floor, 1e-12)
    else:
        scatter_to_noise = 1.0  # can't compute without errors

    if scatter_to_noise > 3.0:
        reason_codes.append("EXCESS_SCATTER")

    # --- Grade assignment ---
    if not reason_codes:
        grade = "A"
    elif len(reason_codes) == 1 and reason_codes[0] in {
        "HIGH_OUTLIER_FRACTION", "IRREGULAR_CADENCE"
    }:
        grade = "B"
    elif len(reason_codes) <= 2:
        grade = "C"
    else:
        grade = "D"

    return QualityReport(
        grade=grade,
        outlier_fraction=outlier_frac,
        gap_fraction=float(gap_frac),
        scatter_to_noise_ratio=scatter_to_noise,
        cadence_regularity=regularity,
        n_cadences=n,
        reason_codes=reason_codes,
    )


def format_quality_report(report: QualityReport) -> str:
    """Format a QualityReport as a short Markdown summary.

    Args:
        report: From :func:`check_data_quality`.

    Returns:
        Markdown string.
    """
    flags = ", ".join(report.reason_codes) if report.reason_codes else "none"
    lines = [
        f"## Data Quality: Grade {report.grade}",
        "",
        f"- N cadences: {report.n_cadences}",
        f"- Outlier fraction: {report.outlier_fraction:.3%}",
        f"- Gap fraction: {report.gap_fraction:.3%}",
        f"- Scatter/noise ratio: {report.scatter_to_noise_ratio:.2f}",
        f"- Cadence regularity: {report.cadence_regularity:.3%}",
        f"- Flags: {flags}",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="data_quality_checker",
        description="Assess light-curve quality before running the full pipeline.",
    )
    parser.add_argument("--target", type=str, required=True, metavar="ID",
                        help='Target identifier, e.g. "TIC 150428135".')
    parser.add_argument("--cadence", type=float, default=120.0, metavar="SECS",
                        help="Expected cadence in seconds (default: 120).")
    args = parser.parse_args(argv)

    print(f"Quality check for {args.target} requires a loaded light curve.")
    print("Use the library API: check_data_quality(time, flux, flux_err).")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
