"""Project how transit-timing uncertainty grows with time since last observation.

Models linear ephemeris uncertainty propagation:
  σ_T(n) = sqrt(σ_T0² + (n · σ_P)²)
where n is the number of cycles elapsed.

Public API
----------
EphemerisUncertaintyResult(sigma_t0_minutes, sigma_period_minutes,
                            predictions, baseline_bjd, flag)
TransitPrediction(transit_number, bjd, sigma_minutes, window_hours)
project_ephemeris_uncertainty(epoch_bjd, period_days, *,
                               epoch_err_days, period_err_days,
                               n_cycles, baseline_bjd) -> EphemerisUncertaintyResult
format_ephemeris_uncertainty_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TransitPrediction:
    transit_number: int
    bjd: float
    sigma_minutes: float
    window_hours: float        # ±3σ window in hours


@dataclass(frozen=True)
class EphemerisUncertaintyResult:
    sigma_t0_minutes: float
    sigma_period_minutes: float
    predictions: tuple[TransitPrediction, ...]
    baseline_bjd: float
    n_cycles_total: int
    flag: str                  # "WELL_CONSTRAINED", "MARGINAL", "POORLY_CONSTRAINED"


def project_ephemeris_uncertainty(
    epoch_bjd: float,
    period_days: float,
    *,
    epoch_err_days: float,
    period_err_days: float,
    n_cycles: int = 10,
    baseline_bjd: float | None = None,
    well_constrained_minutes: float = 30.0,
    poorly_constrained_minutes: float = 120.0,
) -> EphemerisUncertaintyResult:
    """Project ephemeris timing uncertainty over future transits.

    Args:
        epoch_bjd: Reference transit epoch (BJD).
        period_days: Orbital period in days.
        epoch_err_days: 1-sigma epoch uncertainty in days.
        period_err_days: 1-sigma period uncertainty in days.
        n_cycles: Number of future transit cycles to project.
        baseline_bjd: Start BJD for transit numbering (default = epoch_bjd).
        well_constrained_minutes: Threshold for WELL_CONSTRAINED flag.
        poorly_constrained_minutes: Threshold for POORLY_CONSTRAINED flag.

    Returns:
        :class:`EphemerisUncertaintyResult`.
    """
    if period_days <= 0:
        return EphemerisUncertaintyResult(
            0.0, 0.0, (), epoch_bjd, 0, "POORLY_CONSTRAINED"
        )

    base = baseline_bjd if baseline_bjd is not None else epoch_bjd
    sigma_t0_d = max(float(epoch_err_days), 0.0)
    sigma_P_d = max(float(period_err_days), 0.0)
    sigma_t0_min = sigma_t0_d * 1440.0
    sigma_P_min = sigma_P_d * 1440.0

    # First transit number at or after baseline
    n_start = math.ceil((base - epoch_bjd) / period_days) if period_days > 0 else 0

    predictions: list[TransitPrediction] = []
    for i in range(n_cycles):
        n = n_start + i
        t_n = epoch_bjd + n * period_days
        sigma_n_min = math.sqrt(sigma_t0_min ** 2 + (n * sigma_P_min) ** 2)
        window_h = 3.0 * sigma_n_min / 60.0
        predictions.append(TransitPrediction(
            transit_number=int(n),
            bjd=round(t_n, 6),
            sigma_minutes=round(sigma_n_min, 3),
            window_hours=round(window_h, 3),
        ))

    # Flag based on worst-case (last) prediction
    worst_sigma = predictions[-1].sigma_minutes if predictions else sigma_t0_min
    if worst_sigma <= well_constrained_minutes:
        flag = "WELL_CONSTRAINED"
    elif worst_sigma <= poorly_constrained_minutes:
        flag = "MARGINAL"
    else:
        flag = "POORLY_CONSTRAINED"

    return EphemerisUncertaintyResult(
        sigma_t0_minutes=round(sigma_t0_min, 3),
        sigma_period_minutes=round(sigma_P_min, 4),
        predictions=tuple(predictions),
        baseline_bjd=base,
        n_cycles_total=n_cycles,
        flag=flag,
    )


def format_ephemeris_uncertainty_result(result: EphemerisUncertaintyResult) -> str:
    """Format ephemeris uncertainty result as Markdown."""
    lines = [
        "## Ephemeris Uncertainty Growth",
        "",
        f"- σ(T₀): {result.sigma_t0_minutes:.2f} min",
        f"- σ(P): {result.sigma_period_minutes:.3f} min/cycle",
        f"- Flag: **{result.flag}**",
        "",
        "### Predicted Transits",
        "",
        "| N | BJD | σ (min) | ±3σ window (h) |",
        "|---|---|---|---|",
    ]
    for p in result.predictions:
        lines.append(
            f"| {p.transit_number} | {p.bjd:.4f} | {p.sigma_minutes:.1f} | {p.window_hours:.2f} |"
        )
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="ephemeris_uncertainty_growth",
        description="Project transit-timing uncertainty growth over future cycles.",
    )
    parser.add_argument("epoch_bjd", type=float)
    parser.add_argument("period_days", type=float)
    parser.add_argument("--epoch-err", type=float, required=True)
    parser.add_argument("--period-err", type=float, required=True)
    parser.add_argument("--n-cycles", type=int, default=10)
    args = parser.parse_args(argv)

    result = project_ephemeris_uncertainty(
        args.epoch_bjd, args.period_days,
        epoch_err_days=args.epoch_err,
        period_err_days=args.period_err,
        n_cycles=args.n_cycles,
    )
    print(format_ephemeris_uncertainty_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
