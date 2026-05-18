"""Verify a BLS period claim using a chi-square / F-test on the phase-folded LC.

Bins the phase-folded light curve and computes chi² for a flat model vs. the
binned phase model. An F-test with an incomplete-beta p-value determines
whether the periodicity is statistically significant.

Public API
----------
ChiSquarePeriodResult(period_days, chi2_null, chi2_folded, delta_chi2,
                      dof_null, dof_folded, f_statistic, p_value,
                      is_significant, flag)
check_chi_square_period(time, flux, period_days, epoch_bjd, *,
                        flux_err, n_phase_bins,
                        significance_threshold) -> ChiSquarePeriodResult
format_chi_square_period_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """Compute I_x(a, b) via continued fraction (Numerical Recipes §6.4)."""
    if x < 0.0 or x > 1.0:
        return 0.0
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0

    # Use symmetry relation for numerical stability
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _regularized_incomplete_beta(b, a, 1.0 - x)

    # Log of beta function via Lanczos approximation
    def _log_beta(a: float, b: float) -> float:
        return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)

    lbeta_ab = _log_beta(a, b)
    front = math.exp(math.log(x) * a + math.log(1.0 - x) * b - lbeta_ab) / a

    # Continued fraction via Lentz method
    f = 1.0
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / max(abs(d), 1e-30) * math.copysign(1, d)
    f = d

    for m in range(1, 201):
        # Even step
        num = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        d = 1.0 + num * d
        c = 1.0 + num / c
        d = 1.0 / max(abs(d), 1e-30) * math.copysign(1, d)
        c = max(abs(c), 1e-30) * math.copysign(1, c)
        delta = c * d
        f *= delta
        # Odd step
        num = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        c = 1.0 + num / c
        d = 1.0 / max(abs(d), 1e-30) * math.copysign(1, d)
        c = max(abs(c), 1e-30) * math.copysign(1, c)
        delta = c * d
        f *= delta
        if abs(delta - 1.0) < 1e-10:
            break

    return front * f


def _f_distribution_cdf(f_stat: float, d1: int, d2: int) -> float:
    """CDF of F(d1, d2) at f_stat — returns P(F <= f_stat)."""
    if f_stat <= 0:
        return 0.0
    x = d1 * f_stat / (d1 * f_stat + d2)
    return _regularized_incomplete_beta(d1 / 2.0, d2 / 2.0, x)


@dataclass(frozen=True)
class ChiSquarePeriodResult:
    period_days: float
    chi2_null: float
    chi2_folded: float
    delta_chi2: float
    dof_null: int
    dof_folded: int
    f_statistic: float
    p_value: float
    is_significant: bool
    flag: str  # "SIGNIFICANT" | "MARGINAL" | "NOT_SIGNIFICANT" | "INSUFFICIENT"


def check_chi_square_period(
    time: list[float],
    flux: list[float],
    period_days: float,
    epoch_bjd: float,
    *,
    flux_err: list[float] | None = None,
    n_phase_bins: int = 20,
    significance_threshold: float = 0.01,
) -> ChiSquarePeriodResult:
    """Check period significance via chi-square F-test on phase-folded bins.

    Args:
        time: Time array (BJD).
        flux: Normalised flux array.
        period_days: Period to test in days.
        epoch_bjd: Reference epoch.
        flux_err: Per-point uncertainties (1.0 assumed if None).
        n_phase_bins: Number of phase bins.
        significance_threshold: p-value below which the period is significant.

    Returns:
        :class:`ChiSquarePeriodResult`.
    """
    n = len(flux)
    if n < n_phase_bins + 2 or period_days <= 0:
        return ChiSquarePeriodResult(
            period_days, 0.0, 0.0, 0.0, 0, 0, 0.0, 1.0, False, "INSUFFICIENT",
        )

    errs = flux_err if (flux_err is not None and len(flux_err) == n) else [1.0] * n

    # Compute global weighted mean for null model
    w_sum = sum(1.0 / e ** 2 for e in errs)
    wf_sum = sum(f / e ** 2 for f, e in zip(flux, errs, strict=False))
    global_mean = wf_sum / w_sum if w_sum > 0 else 0.0

    # Chi2 for null (flat) model
    chi2_null = sum(
        (f - global_mean) ** 2 / e ** 2
        for f, e in zip(flux, errs, strict=False)
    )
    dof_null = n - 1

    # Phase-fold and bin
    bins: list[list[tuple[float, float]]] = [[] for _ in range(n_phase_bins)]
    for t, f, e in zip(time, flux, errs, strict=False):
        ph = ((t - epoch_bjd) % period_days) / period_days
        idx = min(int(ph * n_phase_bins), n_phase_bins - 1)
        bins[idx].append((f, e))

    # Compute binned model means
    bin_means: list[float] = []
    for b in bins:
        if not b:
            bin_means.append(global_mean)
        else:
            bw = sum(1.0 / e ** 2 for _, e in b)
            bf = sum(f / e ** 2 for f, e in b)
            bin_means.append(bf / bw if bw > 0 else global_mean)

    # Chi2 for folded model
    chi2_folded = 0.0
    for t, f, e in zip(time, flux, errs, strict=False):
        ph = ((t - epoch_bjd) % period_days) / period_days
        idx = min(int(ph * n_phase_bins), n_phase_bins - 1)
        chi2_folded += (f - bin_means[idx]) ** 2 / e ** 2

    n_nonempty_bins = sum(1 for b in bins if b)
    dof_folded = max(n - n_nonempty_bins, 1)
    delta_chi2 = chi2_null - chi2_folded
    delta_dof = max(dof_null - dof_folded, 1)

    if chi2_folded <= 0 or dof_folded <= 0:
        return ChiSquarePeriodResult(
            period_days, round(chi2_null, 4), 0.0, round(delta_chi2, 4),
            dof_null, dof_folded, 0.0, 1.0, False, "INSUFFICIENT",
        )

    f_stat = (delta_chi2 / delta_dof) / (chi2_folded / dof_folded)
    f_stat = max(f_stat, 0.0)

    # p-value = P(F > f_stat) = 1 - CDF
    p_value = 1.0 - _f_distribution_cdf(f_stat, delta_dof, dof_folded)
    p_value = max(0.0, min(1.0, p_value))

    is_significant = p_value < significance_threshold
    if p_value < significance_threshold:
        flag = "SIGNIFICANT"
    elif p_value < 0.10:
        flag = "MARGINAL"
    else:
        flag = "NOT_SIGNIFICANT"

    return ChiSquarePeriodResult(
        period_days=period_days,
        chi2_null=round(chi2_null, 4),
        chi2_folded=round(chi2_folded, 4),
        delta_chi2=round(delta_chi2, 4),
        dof_null=dof_null,
        dof_folded=dof_folded,
        f_statistic=round(f_stat, 4),
        p_value=round(p_value, 6),
        is_significant=is_significant,
        flag=flag,
    )


def format_chi_square_period_result(result: ChiSquarePeriodResult) -> str:
    """Format chi-square period check result as Markdown."""
    lines = [
        "## Chi-Square Period Check",
        "",
        f"- Period: {result.period_days:.4f} days",
        f"- χ² (null): {result.chi2_null:.2f} (dof={result.dof_null})",
        f"- χ² (folded): {result.chi2_folded:.2f} (dof={result.dof_folded})",
        f"- Δχ²: {result.delta_chi2:.2f}",
        f"- F-statistic: {result.f_statistic:.4f}",
        f"- p-value: {result.p_value:.4e}",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="chi_square_period_checker",
        description="Test BLS period significance via chi-square F-test.",
    )
    parser.add_argument("period_days", type=float)
    parser.add_argument("epoch_bjd", type=float)
    parser.add_argument("--n-phase-bins", type=int, default=20)
    parser.add_argument("--significance-threshold", type=float, default=0.01)
    args = parser.parse_args(argv)

    result = check_chi_square_period(
        [], [], args.period_days, args.epoch_bjd,
        n_phase_bins=args.n_phase_bins,
        significance_threshold=args.significance_threshold,
    )
    print(format_chi_square_period_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
