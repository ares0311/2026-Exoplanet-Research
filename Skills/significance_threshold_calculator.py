"""Compute empirical BLS/SNR detection thresholds from out-of-transit noise.

Uses bootstrap scrambling of the out-of-transit flux to estimate the
false-alarm rate and set a significance threshold for the BLS power or SNR.

Public API
----------
ThresholdResult(n_bootstrap, snr_threshold, power_threshold,
                false_alarm_rate, sigma_level, flag)
compute_snr_threshold(flux_oot, *, n_bootstrap, false_alarm_rate,
                      n_transit_points) -> ThresholdResult
compute_bls_threshold(flux_oot, period_days, *, n_bootstrap,
                      false_alarm_rate, duration_hours) -> ThresholdResult
format_threshold_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ThresholdResult:
    n_bootstrap: int
    snr_threshold: float | None
    power_threshold: float | None
    false_alarm_rate: float
    sigma_level: float | None      # Gaussian equivalent sigma
    flag: str  # "OK" | "INSUFFICIENT" | "INVALID"


def _lcg_rand(seed: int = 12345):
    """Simple LCG pseudo-random number generator (no numpy)."""
    state = seed
    while True:
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        yield state / 0xFFFFFFFF


def _bootstrap_snr(flux: list[float], n_in: int, n_boot: int, seed: int = 42) -> list[float]:
    """Bootstrap distribution of in-transit mean - OOT mean SNRs."""
    n = len(flux)
    if n < n_in + 2:
        return []
    rng = _lcg_rand(seed)
    m = sum(flux) / n
    std = math.sqrt(sum((f - m) ** 2 for f in flux) / n) or 1e-9
    snrs: list[float] = []
    perm = list(range(n))
    for _ in range(n_boot):
        # Fisher-Yates shuffle for n_in elements
        p = perm[:]
        for i in range(n_in):
            j = i + int(next(rng) * (n - i))
            p[i], p[j] = p[j], p[i]
        in_mean = sum(flux[p[i]] for i in range(n_in)) / n_in
        oot_idx = p[n_in:]
        oot_mean = sum(flux[idx] for idx in oot_idx) / len(oot_idx)
        snr = abs(in_mean - oot_mean) / (std / math.sqrt(n_in))
        snrs.append(snr)
    return snrs


def compute_snr_threshold(
    flux_oot: list[float],
    *,
    n_bootstrap: int = 200,
    false_alarm_rate: float = 0.01,
    n_transit_points: int = 5,
) -> ThresholdResult:
    """Compute an empirical SNR detection threshold via bootstrap.

    Args:
        flux_oot: Out-of-transit (or full) flux array.
        n_bootstrap: Number of bootstrap permutations.
        false_alarm_rate: Target false-alarm probability (e.g. 0.01 = 1%).
        n_transit_points: Number of in-transit cadences to simulate.

    Returns:
        :class:`ThresholdResult`.
    """
    n = len(flux_oot)
    if n < 10:
        return ThresholdResult(0, None, None, false_alarm_rate, None, "INSUFFICIENT")
    if n_bootstrap < 10 or false_alarm_rate <= 0 or false_alarm_rate >= 1:
        return ThresholdResult(0, None, None, false_alarm_rate, None, "INVALID")

    snrs = _bootstrap_snr(flux_oot, n_transit_points, n_bootstrap)
    if not snrs:
        return ThresholdResult(n_bootstrap, None, None, false_alarm_rate, None, "INSUFFICIENT")

    snrs_sorted = sorted(snrs)
    idx = max(0, int((1.0 - false_alarm_rate) * len(snrs_sorted)) - 1)
    threshold = snrs_sorted[idx]

    # Gaussian sigma equivalent: Φ^{-1}(1 - FAR) using rational approximation
    p = 1.0 - false_alarm_rate
    if 0 < p < 1:
        t = math.sqrt(-2.0 * math.log(min(p, 1 - p)))
        c = (2.515517 + 0.802853 * t + 0.010328 * t ** 2)
        d = (1 + 1.432788 * t + 0.189269 * t ** 2 + 0.001308 * t ** 3)
        sigma_equiv = t - c / d
        if p < 0.5:
            sigma_equiv = -sigma_equiv
    else:
        sigma_equiv = None

    return ThresholdResult(
        n_bootstrap=n_bootstrap,
        snr_threshold=round(threshold, 4),
        power_threshold=None,
        false_alarm_rate=false_alarm_rate,
        sigma_level=round(sigma_equiv, 3) if sigma_equiv is not None else None,
        flag="OK",
    )


def compute_bls_threshold(
    flux_oot: list[float],
    period_days: float,
    *,
    n_bootstrap: int = 200,
    false_alarm_rate: float = 0.01,
    duration_hours: float = 2.0,
) -> ThresholdResult:
    """Compute an empirical BLS power threshold via bootstrap.

    Approximates BLS power as (depth/noise)² scaled by transit fraction.

    Args:
        flux_oot: Out-of-transit flux array.
        period_days: Orbital period in days.
        n_bootstrap: Number of bootstrap permutations.
        false_alarm_rate: Target false-alarm probability.
        duration_hours: Expected transit duration in hours.

    Returns:
        :class:`ThresholdResult`.
    """
    n = len(flux_oot)
    if n < 10 or period_days <= 0:
        return ThresholdResult(0, None, None, false_alarm_rate, None, "INSUFFICIENT")

    transit_frac = (duration_hours / 24.0) / period_days
    n_in = max(2, int(transit_frac * n))
    snr_result = compute_snr_threshold(
        flux_oot, n_bootstrap=n_bootstrap,
        false_alarm_rate=false_alarm_rate,
        n_transit_points=n_in,
    )
    if snr_result.snr_threshold is None:
        return ThresholdResult(n_bootstrap, None, None, false_alarm_rate, None, snr_result.flag)

    # BLS power ≈ SNR² * transit_frac
    power_thresh = snr_result.snr_threshold ** 2 * transit_frac

    return ThresholdResult(
        n_bootstrap=n_bootstrap,
        snr_threshold=snr_result.snr_threshold,
        power_threshold=round(power_thresh, 6),
        false_alarm_rate=false_alarm_rate,
        sigma_level=snr_result.sigma_level,
        flag="OK",
    )


def format_threshold_result(result: ThresholdResult) -> str:
    """Format threshold result as Markdown."""
    lines = [
        "## Detection Significance Threshold",
        "",
        f"- Bootstrap samples: {result.n_bootstrap}",
        f"- False-alarm rate: {result.false_alarm_rate:.3f}",
        f"- SNR threshold: {result.snr_threshold}",
        f"- BLS power threshold: {result.power_threshold}",
        f"- Gaussian sigma equivalent: {result.sigma_level}",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="significance_threshold_calculator",
        description="Compute empirical detection significance threshold.",
    )
    parser.add_argument("--n-bootstrap", type=int, default=200)
    parser.add_argument("--far", type=float, default=0.01)
    parser.add_argument("--n-transit-points", type=int, default=5)
    args = parser.parse_args(argv)

    result = compute_snr_threshold([], n_bootstrap=args.n_bootstrap,
                                   false_alarm_rate=args.far,
                                   n_transit_points=args.n_transit_points)
    print(format_threshold_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
