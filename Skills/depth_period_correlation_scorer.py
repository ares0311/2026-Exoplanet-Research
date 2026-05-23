"""Test whether transit depth correlates anomalously with orbital period.

For a population of planet candidates, an unexpected correlation between
transit depth and period can indicate population-level systematics (e.g.
contamination varying with orbital distance, or blend scenarios).  Returns
Pearson and Spearman correlation coefficients plus a significance test.

Public API
----------
DepthPeriodResult(n_candidates, pearson_r, spearman_r, pearson_p_approx,
                  slope_ppm_per_day, is_anomalous, flag)
score_depth_period_correlation(periods, depths, *,
                               depth_errs) -> DepthPeriodResult
format_depth_period_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DepthPeriodResult:
    n_candidates: int
    pearson_r: float | None          # Pearson correlation coefficient
    spearman_r: float | None         # Spearman rank correlation
    pearson_p_approx: float | None   # approximate two-tailed p-value
    slope_ppm_per_day: float | None  # OLS slope (depth vs period)
    is_anomalous: bool               # |pearson_r| > 0.5 and p < 0.05
    flag: str  # "OK" | "INSUFFICIENT" | "INVALID"


def _pearson(x: list[float], y: list[float]) -> tuple[float, float]:
    """Pearson r and approximate two-tailed p-value (t-distribution approx)."""
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    dx = math.sqrt(sum((v - mx) ** 2 for v in x))
    dy = math.sqrt(sum((v - my) ** 2 for v in y))
    if dx < 1e-15 or dy < 1e-15:
        return 0.0, 1.0
    r = num / (dx * dy)
    r = max(-1.0, min(1.0, r))
    if n <= 2 or abs(r) >= 1.0:
        return r, 1.0
    t = r * math.sqrt((n - 2) / (1.0 - r ** 2))
    # Approximate p-value via normal approximation for large n
    z = abs(t) / math.sqrt((n - 2 + t ** 2) / (n - 2))
    p = math.erfc(z / math.sqrt(2.0))
    return r, min(1.0, max(0.0, p))


def _spearman(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation coefficient."""
    n = len(x)
    rx = [sorted(x).index(v) + 1 for v in x]
    ry = [sorted(y).index(v) + 1 for v in y]
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    if n <= 1:
        return 0.0
    return 1.0 - 6.0 * d2 / (n * (n ** 2 - 1))


def _ols_slope(x: list[float], y: list[float]) -> float:
    """Ordinary least-squares slope."""
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    den = sum((v - mx) ** 2 for v in x)
    return num / den if abs(den) > 1e-15 else 0.0


def score_depth_period_correlation(
    periods: list[float],
    depths: list[float],
    *,
    depth_errs: list[float] | None = None,
) -> DepthPeriodResult:
    """Test for anomalous depth–period correlation.

    Args:
        periods: Orbital periods (days).
        depths: Transit depths (ppm).
        depth_errs: Optional per-candidate depth uncertainties (ppm).
            Currently stored but not used in the correlation calculation.

    Returns:
        :class:`DepthPeriodResult`.
    """
    if len(periods) != len(depths):
        return DepthPeriodResult(0, None, None, None, None, False, "INVALID")
    n = len(periods)
    if n < 3:
        return DepthPeriodResult(n, None, None, None, None, False, "INSUFFICIENT")

    pr, pp = _pearson(periods, depths)
    sr = _spearman(periods, depths)
    slope = _ols_slope(periods, depths)
    is_anomalous = abs(pr) > 0.5 and pp < 0.05

    return DepthPeriodResult(
        n_candidates=n,
        pearson_r=round(pr, 6),
        spearman_r=round(sr, 6),
        pearson_p_approx=round(pp, 6),
        slope_ppm_per_day=round(slope, 4),
        is_anomalous=is_anomalous,
        flag="OK",
    )


def format_depth_period_result(result: DepthPeriodResult) -> str:
    """Format depth–period correlation result as Markdown."""
    lines = [
        "## Depth–Period Correlation Scorer",
        "",
        f"- Candidates: {result.n_candidates}",
        f"- Pearson r: {result.pearson_r}",
        f"- Spearman r: {result.spearman_r}",
        f"- p-value (approx): {result.pearson_p_approx}",
        f"- Slope: {result.slope_ppm_per_day} ppm/day",
        f"- **Anomalous: {'Yes — possible systematic' if result.is_anomalous else 'No'}**",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="depth_period_correlation_scorer",
        description="Test depth–period correlation across a candidate population.",
    )
    parser.parse_args(argv)

    result = score_depth_period_correlation([], [])
    print(format_depth_period_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
