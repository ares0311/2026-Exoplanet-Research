"""Estimate the false alarm probability (FAP) for a BLS peak.

Analytic method uses the Baluev (2008) extreme-value approximation:
  N_eff = n_cadences * log(P_max / P_min) * n_durations
  FAP = 1 - (1 - exp(-power))^N_eff

Empirical method requires a list of noise power values from scrambled
light curves and computes FAP as the fraction exceeding the observed power.

Public API
----------
FAPResult(period_days, bls_power, n_independent_frequencies, fap, log10_fap,
          significance_sigma, method, flag)
estimate_fap(bls_power, n_cadences, period_days, *, period_min_days,
             period_max_days, n_durations, method,
             noise_powers) -> FAPResult
format_fap_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


def _erfinv_approx(x: float) -> float:
    """Approximate inverse error function for |x| < 0.99 (Winitzki 2008)."""
    a = 0.147
    ln_term = math.log(1.0 - x * x) if abs(x) < 1.0 else -1e10
    term = (2.0 / (math.pi * a) + ln_term / 2.0) ** 2 - ln_term / a
    term = max(term, 0.0)
    return math.copysign(math.sqrt(math.sqrt(term) - (2.0 / (math.pi * a) + ln_term / 2.0)), x)


def _fap_to_sigma(fap: float) -> float:
    """Convert FAP to equivalent Gaussian sigma (two-tailed p-value)."""
    if fap <= 0:
        return 8.0  # effectively infinite significance
    if fap >= 1.0:
        return 0.0
    p = 1.0 - fap
    p = max(1e-15, min(p, 1.0 - 1e-15))
    # sigma = sqrt(2) * erfinv(p)
    return math.sqrt(2.0) * _erfinv_approx(p)


@dataclass(frozen=True)
class FAPResult:
    period_days: float
    bls_power: float
    n_independent_frequencies: int
    fap: float
    log10_fap: float
    significance_sigma: float
    method: str  # "analytic" | "empirical"
    flag: str    # "SIGNIFICANT" | "MARGINAL" | "NOT_SIGNIFICANT"


def estimate_fap(
    bls_power: float,
    n_cadences: int,
    period_days: float,
    *,
    period_min_days: float = 0.5,
    period_max_days: float | None = None,
    n_durations: int = 3,
    method: str = "analytic",
    noise_powers: list[float] | None = None,
) -> FAPResult:
    """Estimate false alarm probability for a BLS peak.

    Args:
        bls_power: Observed BLS peak power.
        n_cadences: Number of light curve cadences.
        period_days: Period of the detected signal in days.
        period_min_days: Minimum period searched.
        period_max_days: Maximum period searched (defaults to n_cadences/2 days if None).
        n_durations: Number of transit duration steps tried.
        method: "analytic" (Baluev 2008) or "empirical" (requires noise_powers).
        noise_powers: BLS powers from scrambled light curves (empirical method only).

    Returns:
        :class:`FAPResult`.
    """
    if period_min_days <= 0 or n_cadences <= 0 or bls_power < 0:
        return FAPResult(period_days, bls_power, 0, 1.0, 0.0, 0.0, method, "NOT_SIGNIFICANT")

    p_max = period_max_days if period_max_days is not None else n_cadences / 2.0
    p_max = max(p_max, period_min_days * 1.01)

    if method == "empirical" and noise_powers is not None and len(noise_powers) > 0:
        n_exceed = sum(1 for p in noise_powers if p >= bls_power)
        fap = n_exceed / len(noise_powers)
        n_eff = len(noise_powers)
        used_method = "empirical"
    else:
        # Analytic (Baluev 2008 extreme-value approximation)
        log_ratio = math.log(p_max / period_min_days) if p_max > period_min_days else 1.0
        n_eff = int(n_cadences * log_ratio * n_durations)
        n_eff = max(n_eff, 1)
        # Single-trial probability: use chi-square approximation
        p1 = math.exp(-bls_power) if bls_power < 700 else 0.0
        fap = 1.0 - (1.0 - p1) ** n_eff
        fap = max(0.0, min(1.0, fap))
        used_method = "analytic"

    log10_fap = math.log10(fap) if fap > 0 else -15.0
    sigma = _fap_to_sigma(fap)

    if fap < 0.01:
        flag = "SIGNIFICANT"
    elif fap < 0.10:
        flag = "MARGINAL"
    else:
        flag = "NOT_SIGNIFICANT"

    return FAPResult(
        period_days=period_days,
        bls_power=bls_power,
        n_independent_frequencies=n_eff,
        fap=round(fap, 8),
        log10_fap=round(log10_fap, 4),
        significance_sigma=round(sigma, 3),
        method=used_method,
        flag=flag,
    )


def format_fap_result(result: FAPResult) -> str:
    """Format FAP result as Markdown."""
    lines = [
        "## False Alarm Probability",
        "",
        f"- Period: {result.period_days:.4f} days",
        f"- BLS power: {result.bls_power:.4f}",
        f"- N_eff frequencies: {result.n_independent_frequencies}",
        f"- Method: {result.method}",
        f"- FAP: {result.fap:.2e}",
        f"- log₁₀(FAP): {result.log10_fap:.2f}",
        f"- Significance: {result.significance_sigma:.2f}σ",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="false_alarm_probability_estimator",
        description="Estimate FAP for a BLS peak.",
    )
    parser.add_argument("bls_power", type=float)
    parser.add_argument("n_cadences", type=int)
    parser.add_argument("period_days", type=float)
    parser.add_argument("--period-min-days", type=float, default=0.5)
    parser.add_argument("--period-max-days", type=float, default=None)
    parser.add_argument("--n-durations", type=int, default=3)
    parser.add_argument("--method", default="analytic")
    args = parser.parse_args(argv)

    result = estimate_fap(
        args.bls_power, args.n_cadences, args.period_days,
        period_min_days=args.period_min_days,
        period_max_days=args.period_max_days,
        n_durations=args.n_durations,
        method=args.method,
    )
    print(format_fap_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
