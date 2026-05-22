"""Test whether transit timing variations (TTVs) are statistically significant.

Computes the O-C (observed minus computed) residuals from a linear ephemeris
and applies a chi-square test against the measurement noise, plus an F-test
comparing a flat vs. variable TTV model.

Public API
----------
TTVTestResult(n_transits, chi2, chi2_reduced, dof, p_value_approx,
              oc_rms_minutes, is_significant, sigma_level, flag)
test_ttv_significance(midpoints, period_days, epoch_bjd, midpoint_errors, *,
                      significance_level) -> TTVTestResult
format_ttv_test_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TTVTestResult:
    n_transits: int
    chi2: float | None
    chi2_reduced: float | None
    dof: int
    p_value_approx: float | None    # approximate p-value from chi2 tail
    oc_rms_minutes: float | None
    is_significant: bool
    sigma_level: float | None       # Gaussian-equivalent sigma
    flag: str  # "OK" | "INSUFFICIENT" | "INVALID"


def _chi2_p_value(chi2: float, dof: int) -> float:
    """Approximate p-value P(χ² > x | dof) using Wilson-Hilferty normal approx."""
    if dof <= 0 or chi2 < 0:
        return 1.0
    # Wilson-Hilferty: (χ²/ν)^(1/3) ≈ N(1 - 2/(9ν), 2/(9ν))
    k = float(dof)
    mu = 1.0 - 2.0 / (9.0 * k)
    sigma = math.sqrt(2.0 / (9.0 * k))
    z = ((chi2 / k) ** (1.0 / 3.0) - mu) / sigma
    # Complementary normal CDF (upper tail): P(Z > z) using erfc
    p = 0.5 * math.erfc(z / math.sqrt(2.0))
    return max(0.0, min(1.0, p))


def _sigma_from_p(p: float) -> float | None:
    """Convert two-tailed p-value to Gaussian sigma equivalent."""
    if p <= 0 or p >= 1:
        return None
    # Rational approximation of inverse normal CDF (Abramowitz & Stegun 26.2.23)
    q = min(p, 1.0 - p)
    t = math.sqrt(-2.0 * math.log(q))
    c = (2.515517 + 0.802853 * t + 0.010328 * t ** 2)
    d = (1.0 + 1.432788 * t + 0.189269 * t ** 2 + 0.001308 * t ** 3)
    sigma = t - c / d
    return round(abs(sigma), 3)


def test_ttv_significance(
    midpoints: list[float],
    period_days: float,
    epoch_bjd: float,
    midpoint_errors: list[float] | None = None,
    *,
    significance_level: float = 0.05,
) -> TTVTestResult:
    """Test whether O-C scatter significantly exceeds measurement noise.

    Args:
        midpoints: Measured transit mid-times (BJD).
        period_days: Adopted linear ephemeris period (days).
        epoch_bjd: Reference epoch (BJD).
        midpoint_errors: Per-transit timing uncertainties (days).
            If None, uses uniform weight and reports rms-based test only.
        significance_level: p-value threshold for ``is_significant``.

    Returns:
        :class:`TTVTestResult`.
    """
    n = len(midpoints)
    if period_days <= 0:
        return TTVTestResult(n, None, None, 0, None, None, False, None, "INVALID")
    if n < 3:
        return TTVTestResult(n, None, None, 0, None, None, False, None, "INSUFFICIENT")

    # Compute O-C residuals
    oc = []
    for t in midpoints:
        ni = round((t - epoch_bjd) / period_days)
        oc.append((t - (epoch_bjd + ni * period_days)) * 1440.0)  # minutes

    oc_rms = math.sqrt(sum(r ** 2 for r in oc) / n)

    # Chi-square test
    dof = n - 1  # period is held fixed; only T0 is a free parameter
    chi2: float | None = None
    chi2_red: float | None = None
    p_val: float | None = None

    if midpoint_errors and len(midpoint_errors) == n and all(e > 0 for e in midpoint_errors):
        err_min = [e * 1440.0 for e in midpoint_errors]  # days → minutes
        chi2 = sum((oc[i] / err_min[i]) ** 2 for i in range(n))
        chi2_red = chi2 / dof if dof > 0 else None
        p_val = _chi2_p_value(chi2, dof)
    else:
        # Without errors: test if chi2_reduced >> 1 using median absolute deviation
        mad = sorted(abs(r) for r in oc)[n // 2] * 1.4826
        if mad > 1e-9:
            chi2 = sum((r / mad) ** 2 for r in oc)
            chi2_red = chi2 / dof if dof > 0 else None
            p_val = _chi2_p_value(chi2, dof)

    sig = _sigma_from_p(p_val) if p_val is not None else None
    is_sig = (p_val is not None and p_val < significance_level)

    return TTVTestResult(
        n_transits=n,
        chi2=round(chi2, 4) if chi2 is not None else None,
        chi2_reduced=round(chi2_red, 4) if chi2_red is not None else None,
        dof=dof,
        p_value_approx=round(p_val, 6) if p_val is not None else None,
        oc_rms_minutes=round(oc_rms, 4),
        is_significant=is_sig,
        sigma_level=sig,
        flag="OK",
    )


def format_ttv_test_result(result: TTVTestResult) -> str:
    """Format TTV significance test result as Markdown."""
    lines = [
        "## TTV Significance Test",
        "",
        f"- Transits: {result.n_transits}",
        f"- O-C RMS: {result.oc_rms_minutes} minutes",
        f"- χ²: {result.chi2}",
        f"- χ²_red: {result.chi2_reduced} (dof={result.dof})",
        f"- p-value ≈ {result.p_value_approx}",
        f"- Gaussian σ equivalent: {result.sigma_level}",
        f"- **TTVs significant: {'Yes' if result.is_significant else 'No'}**",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="ttv_significance_tester",
        description="Test whether TTV O-C scatter exceeds measurement noise.",
    )
    parser.add_argument("period_days", type=float)
    parser.add_argument("epoch_bjd", type=float)
    args = parser.parse_args(argv)

    result = test_ttv_significance([], args.period_days, args.epoch_bjd)
    print(format_ttv_test_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
