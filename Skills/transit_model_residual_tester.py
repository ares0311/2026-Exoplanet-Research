"""Statistical tests on transit model residuals for autocorrelation/systematics.

After fitting a transit model, the residuals should be white noise.
This module computes:
  - Durbin-Watson statistic (autocorrelation in residuals)
  - Runs test (non-randomness of sign sequences)
  - Reduced chi-squared (goodness of fit)

Public API
----------
ResidualTestResult(n_points, durbin_watson, dw_interpretation,
                   runs_z_score, chi2_reduced, is_white_noise, flag)
test_model_residuals(residuals, *, flux_err, dw_threshold_lo,
                     dw_threshold_hi, runs_z_threshold,
                     chi2_threshold) -> ResidualTestResult
format_residual_test_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ResidualTestResult:
    n_points: int
    durbin_watson: float | None    # 0–4; 2 = no autocorrelation
    dw_interpretation: str         # "positive_AC" | "negative_AC" | "no_AC"
    runs_z_score: float | None     # z-score of runs test; |z| > 1.96 → non-random
    chi2_reduced: float | None
    is_white_noise: bool
    flag: str  # "OK" | "INSUFFICIENT" | "INVALID"


def _durbin_watson(residuals: list[float]) -> float:
    """Durbin-Watson statistic."""
    n = len(residuals)
    if n < 2:
        return float("nan")
    num = sum((residuals[i] - residuals[i - 1]) ** 2 for i in range(1, n))
    denom = sum(r ** 2 for r in residuals)
    if denom < 1e-30:
        return 2.0
    return num / denom


def _runs_z_score(residuals: list[float]) -> float | None:
    """Z-score for the runs test (Wald-Wolfowitz)."""
    signs = [1 if r >= 0 else -1 for r in residuals]
    n = len(signs)
    n_pos = sum(1 for s in signs if s > 0)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    runs = 1 + sum(1 for i in range(1, n) if signs[i] != signs[i - 1])
    exp_runs = (2 * n_pos * n_neg) / n + 1
    var_runs = (2 * n_pos * n_neg * (2 * n_pos * n_neg - n)) / (n ** 2 * (n - 1))
    if var_runs <= 0:
        return None
    return (runs - exp_runs) / math.sqrt(var_runs)


def test_model_residuals(
    residuals: list[float],
    *,
    flux_err: list[float] | None = None,
    dw_threshold_lo: float = 1.5,
    dw_threshold_hi: float = 2.5,
    runs_z_threshold: float = 1.96,
    chi2_threshold: float = 3.0,
) -> ResidualTestResult:
    """Run statistical whiteness tests on model residuals.

    Args:
        residuals: Residual flux values (observed minus model).
        flux_err: Per-point uncertainties for chi² calculation.
        dw_threshold_lo: DW below this → positive autocorrelation.
        dw_threshold_hi: DW above this → negative autocorrelation.
        runs_z_threshold: |z| above this → non-random residuals.
        chi2_threshold: Reduced chi² above this → poor fit.

    Returns:
        :class:`ResidualTestResult`.
    """
    n = len(residuals)
    if n < 5:
        return ResidualTestResult(n, None, "no_AC", None, None, False, "INVALID")

    dw = _durbin_watson(residuals)
    if math.isnan(dw):
        dw_interp = "no_AC"
    elif dw < dw_threshold_lo:
        dw_interp = "positive_AC"
    elif dw > dw_threshold_hi:
        dw_interp = "negative_AC"
    else:
        dw_interp = "no_AC"

    runs_z = _runs_z_score(residuals)
    runs_z_r = round(runs_z, 4) if runs_z is not None else None

    chi2: float | None = None
    if flux_err is not None and len(flux_err) == n and n > 1:
        chi2_sum = sum((residuals[i] / max(flux_err[i], 1e-30)) ** 2 for i in range(n))
        chi2 = round(chi2_sum / (n - 1), 4)

    is_white = (
        dw_interp == "no_AC"
        and (runs_z is None or abs(runs_z) < runs_z_threshold)
        and (chi2 is None or chi2 < chi2_threshold)
    )

    return ResidualTestResult(
        n_points=n,
        durbin_watson=round(dw, 4) if not math.isnan(dw) else None,
        dw_interpretation=dw_interp,
        runs_z_score=runs_z_r,
        chi2_reduced=chi2,
        is_white_noise=is_white,
        flag="INSUFFICIENT" if n < 10 else "OK",
    )


def format_residual_test_result(result: ResidualTestResult) -> str:
    """Format residual test result as Markdown."""
    lines = [
        "## Transit Model Residual Tests",
        "",
        f"- Points: {result.n_points}",
        f"- Durbin-Watson: {result.durbin_watson} ({result.dw_interpretation})",
        f"- Runs z-score: {result.runs_z_score}",
        f"- χ² reduced: {result.chi2_reduced}",
        f"- White noise: {'Yes' if result.is_white_noise else 'No'}",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="transit_model_residual_tester",
        description="Statistical tests on transit model residuals.",
    )
    parser.parse_args(argv)
    result = test_model_residuals([])
    print(format_residual_test_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
