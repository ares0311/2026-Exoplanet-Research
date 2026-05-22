"""Fit a linear ephemeris T(n) = T0 + n * P to a set of measured transit midpoints.

Given a list of (transit_number, measured_midpoint) pairs the module fits for
the reference epoch T0 and period P using weighted least squares, returning
residuals (O-C) and uncertainties.

Public API
----------
EphemerisFitResult(t0, t0_err, period_days, period_err, rms_oc_minutes,
                   n_transits, chi2_reduced, flag)
fit_linear_ephemeris(transit_numbers, midpoints, *, midpoint_errors) -> EphemerisFitResult
format_ephemeris_fit_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class EphemerisFitResult:
    t0: float                 # reference epoch (same units as midpoints)
    t0_err: float | None
    period_days: float
    period_err: float | None
    rms_oc_minutes: float     # RMS of O-C residuals in minutes
    n_transits: int
    chi2_reduced: float | None
    oc_residuals: tuple[float, ...]  # O-C in minutes, one per transit
    flag: str  # "OK" | "INSUFFICIENT" | "INVALID"


def fit_linear_ephemeris(
    transit_numbers: list[int],
    midpoints: list[float],
    *,
    midpoint_errors: list[float] | None = None,
) -> EphemerisFitResult:
    """Fit T(n) = T0 + n * P to measured transit midpoints.

    Args:
        transit_numbers: Integer transit epoch numbers (arbitrary zero-point).
        midpoints: Measured mid-transit times (days, BJD or similar).
        midpoint_errors: Per-midpoint uncertainties in days (optional).

    Returns:
        :class:`EphemerisFitResult`.
    """
    n = len(midpoints)
    if n < 2 or len(transit_numbers) != n:
        return EphemerisFitResult(0.0, None, 0.0, None, 0.0, n, None, (), "INVALID")
    if n < 3:
        # Two-point case: exact solution, no uncertainties
        dn = transit_numbers[1] - transit_numbers[0]
        if dn == 0:
            return EphemerisFitResult(0.0, None, 0.0, None, 0.0, n, None, (), "INVALID")
        p = (midpoints[1] - midpoints[0]) / dn
        t0 = midpoints[0] - transit_numbers[0] * p
        return EphemerisFitResult(t0, None, p, None, 0.0, n, None, (0.0, 0.0), "INSUFFICIENT")

    errs = (
        midpoint_errors
        if midpoint_errors is not None and len(midpoint_errors) == n
        else [1.0] * n
    )
    w = [1.0 / max(e ** 2, 1e-30) for e in errs]

    # Weighted least squares: minimise sum w_i (T_i - T0 - n_i * P)^2
    # Design matrix columns: [1, n_i]
    # Normal equations:
    #   [sum_w       sum_wn  ] [T0]   [sum_wT ]
    #   [sum_wn      sum_wn2 ] [P ] = [sum_wnT]

    sum_w = sum(w)
    sum_wn = sum(w[i] * transit_numbers[i] for i in range(n))
    sum_wn2 = sum(w[i] * transit_numbers[i] ** 2 for i in range(n))
    sum_wT = sum(w[i] * midpoints[i] for i in range(n))
    sum_wnT = sum(w[i] * transit_numbers[i] * midpoints[i] for i in range(n))

    det = sum_w * sum_wn2 - sum_wn ** 2
    if abs(det) < 1e-30:
        return EphemerisFitResult(0.0, None, 0.0, None, 0.0, n, None, (), "INVALID")

    t0 = (sum_wT * sum_wn2 - sum_wnT * sum_wn) / det
    p = (sum_w * sum_wnT - sum_wn * sum_wT) / det

    # Residuals
    oc_days = [midpoints[i] - (t0 + transit_numbers[i] * p) for i in range(n)]
    oc_min = [r * 1440.0 for r in oc_days]
    rms = math.sqrt(sum(r ** 2 for r in oc_min) / n)

    # Uncertainties from inverse normal matrix
    cov_t0 = sum_wn2 / det
    cov_p = sum_w / det
    t0_err = math.sqrt(max(cov_t0, 0.0))
    p_err = math.sqrt(max(cov_p, 0.0))

    # Reduced chi-squared
    chi2: float | None = None
    if n > 2:
        chi2 = sum(w[i] * oc_days[i] ** 2 for i in range(n)) / (n - 2)
        chi2 = round(chi2, 4)

    return EphemerisFitResult(
        t0=round(t0, 8),
        t0_err=round(t0_err, 8),
        period_days=round(p, 8),
        period_err=round(p_err, 8),
        rms_oc_minutes=round(rms, 4),
        n_transits=n,
        chi2_reduced=chi2,
        oc_residuals=tuple(round(r, 4) for r in oc_min),
        flag="OK",
    )


def format_ephemeris_fit_result(result: EphemerisFitResult) -> str:
    """Format ephemeris fit result as Markdown."""
    lines = [
        "## Linear Ephemeris Fit",
        "",
        f"- T0: {result.t0:.6f} ± {result.t0_err} days",
        f"- Period: {result.period_days:.6f} ± {result.period_err} days",
        f"- Transits fitted: {result.n_transits}",
        f"- RMS O-C: {result.rms_oc_minutes:.2f} min",
        f"- χ² reduced: {result.chi2_reduced}",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="multi_epoch_fitter",
        description="Fit linear ephemeris to measured transit midpoints.",
    )
    parser.add_argument("--transit-numbers", nargs="+", type=int, default=[])
    parser.add_argument("--midpoints", nargs="+", type=float, default=[])
    args = parser.parse_args(argv)

    result = fit_linear_ephemeris(args.transit_numbers, args.midpoints)
    print(format_ephemeris_fit_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
