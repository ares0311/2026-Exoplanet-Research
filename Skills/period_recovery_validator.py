"""Validate a BLS-detected period by comparing transit SNR at P, P/2, and 2P.

A genuine planetary signal should have maximum flux depth at the detected
period P.  If P/2 or 2P gives equal or better SNR, the detected period is
likely an alias or harmonic.

Public API
----------
PeriodValidationResult(period_days, snr_at_p, snr_at_half_p, snr_at_double_p,
                       best_period, is_correct_period, confidence)
validate_period(time, flux, period, epoch, *, duration_days, snr_fn) -> PeriodValidationResult
format_validation_result(result) -> str
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class PeriodValidationResult:
    period_days: float
    snr_at_p: float
    snr_at_half_p: float
    snr_at_double_p: float
    best_period: float          # the period (P, P/2, or 2P) with highest SNR
    is_correct_period: bool     # True if P has highest SNR
    confidence: float           # snr_at_p / max(snr_at_half_p, snr_at_double_p)


def _fold_snr(
    time: list[float],
    flux: list[float],
    period: float,
    epoch: float,
    half_dur: float,
) -> float:
    """Compute SNR of the folded transit at this period."""
    in_transit, out_transit = [], []
    for t, f in zip(time, flux, strict=False):
        ph = (t - epoch) % period
        if ph > period / 2:
            ph -= period
        if abs(ph) <= half_dur:
            in_transit.append(f)
        else:
            out_transit.append(f)

    if len(in_transit) < 2 or len(out_transit) < 2:
        return 0.0

    mean_in  = sum(in_transit) / len(in_transit)
    mean_out = sum(out_transit) / len(out_transit)

    # Noise from out-of-transit scatter
    m = mean_out
    var = sum((f - m) ** 2 for f in out_transit) / (len(out_transit) - 1)
    std_out = math.sqrt(var) if var > 0 else 1e-9

    # SNR: depth / (noise / sqrt(n_in))
    depth = abs(mean_out - mean_in)
    snr = depth / (std_out / max(len(in_transit) ** 0.5, 1))
    return snr


def validate_period(
    time: list[float],
    flux: list[float],
    period: float,
    epoch: float,
    *,
    duration_days: float = 0.1,
    snr_fn: Callable[
        [list[float], list[float], float, float, float], float
    ] | None = None,
) -> PeriodValidationResult:
    """Validate period P by comparing folded-transit SNR at P, P/2, and 2P.

    Args:
        time: BJD time array.
        flux: Normalised flux.
        period: Detected BLS period in days.
        epoch: Mid-transit epoch (BJD).
        duration_days: Transit duration for phase-window masking.
        snr_fn: Injectable SNR function for testing.

    Returns:
        :class:`PeriodValidationResult`.
    """
    if period <= 0:
        raise ValueError(f"period must be positive, got {period}")

    fn = snr_fn if snr_fn is not None else _fold_snr
    half_dur = duration_days / 2.0

    snr_p   = fn(time, flux, period,       epoch, half_dur)
    snr_hp  = fn(time, flux, period / 2.0, epoch, half_dur)
    snr_2p  = fn(time, flux, period * 2.0, epoch, half_dur)

    candidates = {period: snr_p, period / 2.0: snr_hp, period * 2.0: snr_2p}
    best_p = max(candidates, key=lambda k: candidates[k])
    is_correct = (best_p == period)

    denom = max(snr_hp, snr_2p, 1e-9)
    confidence = snr_p / denom

    return PeriodValidationResult(
        period_days=period,
        snr_at_p=snr_p,
        snr_at_half_p=snr_hp,
        snr_at_double_p=snr_2p,
        best_period=best_p,
        is_correct_period=is_correct,
        confidence=confidence,
    )


def format_validation_result(result: PeriodValidationResult) -> str:
    """Format period validation as a Markdown block."""
    verdict = "CONFIRMED" if result.is_correct_period else "SUSPECT"
    lines = [
        "## Period Validation",
        "",
        f"- Tested period: {result.period_days:.4f} d",
        f"- SNR at P:    {result.snr_at_p:.2f}",
        f"- SNR at P/2:  {result.snr_at_half_p:.2f}",
        f"- SNR at 2P:   {result.snr_at_double_p:.2f}",
        f"- Best period: {result.best_period:.4f} d",
        f"- Confidence (SNR_P / max_alias): {result.confidence:.3f}",
        f"- **Verdict: {verdict}**",
    ]
    if not result.is_correct_period:
        lines.append(
            f"  - Best SNR is at {result.best_period:.4f} d, "
            "not the detected period — period may be an alias."
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(
        prog="period_recovery_validator",
        description="Validate a BLS period by comparing SNR at P, P/2, and 2P.",
    )
    parser.add_argument("--lc", required=True, metavar="JSON")
    parser.add_argument("--period", type=float, required=True)
    parser.add_argument("--epoch", type=float, required=True)
    parser.add_argument("--duration", type=float, default=0.1)
    args = parser.parse_args(argv)

    lc = json.loads(Path(args.lc).read_text())
    result = validate_period(
        lc["time"], lc["flux"],
        args.period, args.epoch,
        duration_days=args.duration,
    )
    print(format_validation_result(result))
    return 0 if result.is_correct_period else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
