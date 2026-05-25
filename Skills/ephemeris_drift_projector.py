"""Project cumulative ephemeris timing uncertainty growth.

Computes σ_T(n) = sqrt(σ_epoch² + n²·σ_period²) for a given elapsed number
of cycles and flags when the drift exceeds a configurable threshold.

Public API
----------
EphemerisDriftResult
project_ephemeris_drift(period_days, epoch_bjd, sigma_epoch_days, sigma_period_days, ...)
    -> EphemerisDriftResult
format_ephemeris_drift(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class EphemerisDriftResult:
    period_days: float
    epoch_bjd: float
    sigma_epoch_days: float
    sigma_period_days: float
    n_cycles: int
    sigma_tn_days: float
    sigma_tn_minutes: float
    drift_threshold_days: float
    exceeds_threshold: bool
    next_transit_bjd: float
    flag: str  # "OK" | "EXCEEDS_THRESHOLD" | "INVALID"


def project_ephemeris_drift(
    period_days: float,
    epoch_bjd: float,
    sigma_epoch_days: float,
    sigma_period_days: float,
    *,
    reference_bjd: float | None = None,
    transit_duration_days: float | None = None,
    threshold_fraction: float = 0.5,
) -> EphemerisDriftResult:
    """Project ephemeris timing uncertainty growth.

    σ_T(n) = sqrt(σ_epoch² + n²·σ_period²)

    Args:
        period_days: Orbital period in days.
        epoch_bjd: Reference transit epoch in BJD.
        sigma_epoch_days: Uncertainty in epoch (days).
        sigma_period_days: Uncertainty in period (days).
        reference_bjd: BJD at which to evaluate; defaults to epoch.
        transit_duration_days: Transit duration for threshold; defaults to period/10.
        threshold_fraction: Threshold = fraction * transit_duration_days.

    Returns:
        :class:`EphemerisDriftResult`.
    """
    # Validate inputs
    values = [period_days, epoch_bjd, sigma_epoch_days, sigma_period_days]
    if (
        not all(math.isfinite(v) for v in values)
        or period_days <= 0
        or sigma_epoch_days < 0
        or sigma_period_days < 0
    ):
        return EphemerisDriftResult(
            period_days=period_days,
            epoch_bjd=epoch_bjd,
            sigma_epoch_days=sigma_epoch_days,
            sigma_period_days=sigma_period_days,
            n_cycles=0,
            sigma_tn_days=0.0,
            sigma_tn_minutes=0.0,
            drift_threshold_days=0.0,
            exceeds_threshold=False,
            next_transit_bjd=epoch_bjd,
            flag="INVALID",
        )

    ref_bjd = epoch_bjd if reference_bjd is None else reference_bjd

    # Number of cycles elapsed since epoch
    n_cycles = max(0, round((ref_bjd - epoch_bjd) / period_days))

    # Uncertainty propagation
    sigma_tn_days = math.sqrt(
        sigma_epoch_days ** 2 + (n_cycles ** 2) * (sigma_period_days ** 2)
    )
    sigma_tn_minutes = sigma_tn_days * 1440.0

    # Threshold
    t_dur = transit_duration_days if transit_duration_days is not None else period_days / 10.0
    drift_threshold_days = threshold_fraction * t_dur

    exceeds_threshold = sigma_tn_days > drift_threshold_days

    next_transit_bjd = epoch_bjd + n_cycles * period_days

    flag = "EXCEEDS_THRESHOLD" if exceeds_threshold else "OK"

    return EphemerisDriftResult(
        period_days=period_days,
        epoch_bjd=epoch_bjd,
        sigma_epoch_days=sigma_epoch_days,
        sigma_period_days=sigma_period_days,
        n_cycles=n_cycles,
        sigma_tn_days=round(sigma_tn_days, 8),
        sigma_tn_minutes=round(sigma_tn_minutes, 5),
        drift_threshold_days=round(drift_threshold_days, 8),
        exceeds_threshold=exceeds_threshold,
        next_transit_bjd=round(next_transit_bjd, 6),
        flag=flag,
    )


def format_ephemeris_drift(result: EphemerisDriftResult) -> str:
    """Format ephemeris drift result as Markdown."""
    lines = [
        "## Ephemeris Drift Projector",
        "",
        f"- Period: {result.period_days} days",
        f"- Epoch: {result.epoch_bjd} BJD",
        f"- σ(epoch): {result.sigma_epoch_days} days",
        f"- σ(period): {result.sigma_period_days} days",
        f"- Cycles elapsed: {result.n_cycles}",
        f"- **σ_T(n): {result.sigma_tn_days:.6f} days ({result.sigma_tn_minutes:.3f} min)**",
        f"- Drift threshold: {result.drift_threshold_days:.6f} days",
        f"- Exceeds threshold: {result.exceeds_threshold}",
        f"- Next transit: {result.next_transit_bjd} BJD",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="ephemeris_drift_projector",
        description="Project cumulative ephemeris timing uncertainty growth.",
    )
    parser.add_argument("period_days", type=float)
    parser.add_argument("epoch_bjd", type=float)
    parser.add_argument("sigma_epoch_days", type=float)
    parser.add_argument("sigma_period_days", type=float)
    parser.add_argument("--reference-bjd", type=float, default=None)
    parser.add_argument("--transit-duration-days", type=float, default=None)
    parser.add_argument("--threshold-fraction", type=float, default=0.5)
    args = parser.parse_args(argv)

    result = project_ephemeris_drift(
        args.period_days,
        args.epoch_bjd,
        args.sigma_epoch_days,
        args.sigma_period_days,
        reference_bjd=args.reference_bjd,
        transit_duration_days=args.transit_duration_days,
        threshold_fraction=args.threshold_fraction,
    )
    print(format_ephemeris_drift(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
