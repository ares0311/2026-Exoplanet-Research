"""Compare observed transit duration T14 to the expected duration.

Uses stellar density + period to compute the expected T14 via Kepler's third
law and flags observations that deviate by more than sigma_threshold.

Public API
----------
DurationAnomalyResult
check_duration_anomaly(duration_hours, period_days, *, ...) -> DurationAnomalyResult
format_duration_anomaly(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# Physical constants
_G = 6.674e-11        # m^3 kg^-1 s^-2
_R_SUN = 6.957e8      # m
_M_SUN = 1.989e30     # kg
_S_PER_DAY = 86400.0


@dataclass(frozen=True)
class DurationAnomalyResult:
    observed_hours: float
    expected_hours: float | None
    ratio: float | None
    sigma_deviation: float | None
    is_anomalous: bool
    anomaly_type: str | None   # "too_long" | "too_short" | None
    flag: str  # "OK" | "ANOMALOUS" | "INSUFFICIENT_DATA" | "INVALID"


def check_duration_anomaly(
    duration_hours: float,
    period_days: float,
    *,
    stellar_radius_rsun: float | None = None,
    stellar_mass_msun: float | None = None,
    sigma_threshold: float = 2.0,
) -> DurationAnomalyResult:
    """Compare observed T14 to expected duration from stellar parameters.

    T14_expected = (R_star / (pi * a)) * period  (b=0 approximation)

    where a = (G * M_star * T^2 / (4*pi^2))^(1/3)

    Args:
        duration_hours: Observed transit duration T14 in hours.
        period_days: Orbital period in days.
        stellar_radius_rsun: Stellar radius in solar radii (optional).
        stellar_mass_msun: Stellar mass in solar masses (optional).
        sigma_threshold: Number of sigma for anomaly flag.

    Returns:
        :class:`DurationAnomalyResult`.
    """
    if not math.isfinite(duration_hours) or duration_hours <= 0:
        return DurationAnomalyResult(
            observed_hours=duration_hours,
            expected_hours=None,
            ratio=None,
            sigma_deviation=None,
            is_anomalous=False,
            anomaly_type=None,
            flag="INVALID",
        )
    if not math.isfinite(period_days) or period_days <= 0:
        return DurationAnomalyResult(
            observed_hours=duration_hours,
            expected_hours=None,
            ratio=None,
            sigma_deviation=None,
            is_anomalous=False,
            anomaly_type=None,
            flag="INVALID",
        )

    # If stellar params not provided, return INSUFFICIENT_DATA
    if stellar_radius_rsun is None or stellar_mass_msun is None:
        return DurationAnomalyResult(
            observed_hours=duration_hours,
            expected_hours=None,
            ratio=None,
            sigma_deviation=None,
            is_anomalous=False,
            anomaly_type=None,
            flag="INSUFFICIENT_DATA",
        )

    if (
        not math.isfinite(stellar_radius_rsun) or stellar_radius_rsun <= 0
        or not math.isfinite(stellar_mass_msun) or stellar_mass_msun <= 0
    ):
        return DurationAnomalyResult(
            observed_hours=duration_hours,
            expected_hours=None,
            ratio=None,
            sigma_deviation=None,
            is_anomalous=False,
            anomaly_type=None,
            flag="INVALID",
        )

    # Convert to SI
    r_star_m = stellar_radius_rsun * _R_SUN
    m_star_kg = stellar_mass_msun * _M_SUN
    period_s = period_days * _S_PER_DAY

    # Semi-major axis via Kepler's 3rd law
    a_m = (((_G * m_star_kg) * (period_s ** 2)) / (4.0 * math.pi ** 2)) ** (1.0 / 3.0)

    # Expected T14 (hours), b=0 approximation
    expected_days = (r_star_m / (math.pi * a_m)) * period_days
    expected_hours = expected_days * 24.0

    ratio = duration_hours / expected_hours

    # 20% fractional uncertainty on expected duration
    sigma_expected = 0.20 * expected_hours
    sigma_deviation = (duration_hours - expected_hours) / sigma_expected

    is_anomalous = abs(sigma_deviation) > sigma_threshold
    anomaly_type: str | None = None
    if is_anomalous:
        anomaly_type = "too_long" if duration_hours > expected_hours else "too_short"

    flag = "ANOMALOUS" if is_anomalous else "OK"

    return DurationAnomalyResult(
        observed_hours=duration_hours,
        expected_hours=round(expected_hours, 4),
        ratio=round(ratio, 4),
        sigma_deviation=round(sigma_deviation, 4),
        is_anomalous=is_anomalous,
        anomaly_type=anomaly_type,
        flag=flag,
    )


def format_duration_anomaly(result: DurationAnomalyResult) -> str:
    """Format duration anomaly result as Markdown."""
    exp_str = f"{result.expected_hours:.4f} h" if result.expected_hours is not None else "N/A"
    ratio_str = f"{result.ratio:.4f}" if result.ratio is not None else "N/A"
    sigma_str = f"{result.sigma_deviation:.2f} σ" if result.sigma_deviation is not None else "N/A"
    atype_str = result.anomaly_type or "None"
    lines = [
        "## Transit Duration Anomaly Checker",
        "",
        f"- Observed T14: {result.observed_hours:.4f} h",
        f"- Expected T14: {exp_str}",
        f"- Ratio (obs/exp): {ratio_str}",
        f"- Sigma deviation: {sigma_str}",
        f"- Anomalous: {result.is_anomalous}",
        f"- Anomaly type: {atype_str}",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="transit_duration_anomaly_checker",
        description="Compare observed T14 to expected from stellar density + period.",
    )
    parser.add_argument("duration_hours", type=float)
    parser.add_argument("period_days", type=float)
    parser.add_argument("--stellar-radius-rsun", type=float, default=None)
    parser.add_argument("--stellar-mass-msun", type=float, default=None)
    parser.add_argument("--sigma-threshold", type=float, default=2.0)
    args = parser.parse_args(argv)

    result = check_duration_anomaly(
        args.duration_hours,
        args.period_days,
        stellar_radius_rsun=args.stellar_radius_rsun,
        stellar_mass_msun=args.stellar_mass_msun,
        sigma_threshold=args.sigma_threshold,
    )
    print(format_duration_anomaly(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
