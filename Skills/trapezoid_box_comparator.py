"""Compare a box (flat-bottomed) transit model vs a trapezoid model via Δχ² / ΔBIC.

The trapezoid model has an ingress/egress slope parameterised by
``ingress_fraction`` (fraction of the transit duration spent in ingress/egress).
A large BIC improvement for the trapezoid model suggests non-box morphology
(limb-darkened planet or blended EB), while comparable fits suggest insufficient
data to distinguish.

Public API
----------
TrapezoidBoxResult(period_days, epoch_bjd, duration_hours, depth_ppm,
                   chi2_box, chi2_trapezoid, delta_chi2, delta_bic,
                   best_ingress_fraction, preferred_model, flag)
compare_trapezoid_box(time, flux, period_days, epoch_bjd, *,
                      duration_hours, depth_ppm, flux_err,
                      ingress_fractions) -> TrapezoidBoxResult
format_trapezoid_box_result(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrapezoidBoxResult:
    period_days: float
    epoch_bjd: float
    duration_hours: float
    depth_ppm: float
    chi2_box: float
    chi2_trapezoid: float
    delta_chi2: float  # chi2_box - chi2_trapezoid (positive = trapezoid better)
    delta_bic: float   # BIC_box - BIC_trapezoid (positive = trapezoid preferred)
    best_ingress_fraction: float
    preferred_model: str  # "trapezoid" | "box" | "indeterminate"
    flag: str  # "OK" | "INSUFFICIENT" | "INVALID"


def _phase_fold(time: list[float], epoch: float, period: float) -> list[float]:
    phases = []
    for t in time:
        ph = ((t - epoch) % period) / period
        if ph >= 0.5:
            ph -= 1.0
        phases.append(ph)
    return phases


def _box_model(phase: float, half_width: float, depth: float) -> float:
    """Box model: flat bottom inside half_width."""
    return 1.0 - depth if abs(phase) <= half_width else 1.0


def _trapezoid_model(
    phase: float, half_width: float, depth: float, ingress_frac: float
) -> float:
    """Trapezoid: linear ingress/egress of width ingress_frac * half_width."""
    ing = ingress_frac * half_width
    ph = abs(phase)
    if ph >= half_width:
        return 1.0
    if ph <= half_width - ing:
        return 1.0 - depth
    # In the ingress/egress ramp
    slope = (ph - (half_width - ing)) / max(ing, 1e-15)
    return 1.0 - depth * (1.0 - slope)


def compare_trapezoid_box(
    time: list[float],
    flux: list[float],
    period_days: float,
    epoch_bjd: float,
    *,
    duration_hours: float = 2.0,
    depth_ppm: float = 1000.0,
    flux_err: list[float] | None = None,
    ingress_fractions: list[float] | None = None,
) -> TrapezoidBoxResult:
    """Compare box vs trapezoid transit model using Δχ² and ΔBIC.

    Args:
        time: Time array (BJD).
        flux: Normalised flux array.
        period_days: Orbital period in days.
        epoch_bjd: Reference epoch (mid-transit).
        duration_hours: Transit duration in hours.
        depth_ppm: Transit depth in ppm.
        flux_err: Per-point uncertainties (uniform 1.0 if None).
        ingress_fractions: Ingress fractions to grid-search (default 5-point).

    Returns:
        :class:`TrapezoidBoxResult`.
    """
    n = len(flux)
    if n < 10 or period_days <= 0 or duration_hours <= 0 or depth_ppm <= 0:
        return TrapezoidBoxResult(
            period_days, epoch_bjd, duration_hours, depth_ppm,
            0.0, 0.0, 0.0, 0.0, 0.0, "indeterminate", "INVALID",
        )

    errs = flux_err if (flux_err is not None and len(flux_err) == n) else [1.0] * n
    phases = _phase_fold(time, epoch_bjd, period_days)
    half_width = (duration_hours / 24.0) / period_days / 2.0
    depth = depth_ppm / 1e6

    fracs = ingress_fractions if ingress_fractions else [0.05, 0.15, 0.25, 0.40, 0.60]

    # Collect in-transit and nearby out-of-transit points
    window = 3 * half_width
    in_phases: list[float] = []
    in_flux: list[float] = []
    in_errs: list[float] = []
    for ph, f, e in zip(phases, flux, errs, strict=False):
        if abs(ph) <= window:
            in_phases.append(ph)
            in_flux.append(f)
            in_errs.append(e)

    if len(in_phases) < 5:
        return TrapezoidBoxResult(
            period_days, epoch_bjd, duration_hours, depth_ppm,
            0.0, 0.0, 0.0, 0.0, 0.0, "indeterminate", "INSUFFICIENT",
        )

    # chi2 for box model
    chi2_box = sum(
        (f - _box_model(ph, half_width, depth)) ** 2 / max(e ** 2, 1e-30)
        for ph, f, e in zip(in_phases, in_flux, in_errs, strict=False)
    )

    # Grid search over ingress_fractions for trapezoid
    best_chi2_trap = chi2_box * 2
    best_frac = fracs[0]
    for frac in fracs:
        chi2 = sum(
            (f - _trapezoid_model(ph, half_width, depth, frac)) ** 2
            / max(e ** 2, 1e-30)
            for ph, f, e in zip(in_phases, in_flux, in_errs, strict=False)
        )
        if chi2 < best_chi2_trap:
            best_chi2_trap = chi2
            best_frac = frac

    n_pts = len(in_phases)
    delta_chi2 = chi2_box - best_chi2_trap
    # BIC = chi2 + k*ln(n); box has 1 free param (depth given), trapezoid has 2
    bic_box = chi2_box + 1 * (n_pts ** 0.5)  # approximate; k=1 (depth fixed, 0 free params)
    bic_trap = best_chi2_trap + 2 * (n_pts ** 0.5)
    delta_bic = bic_box - bic_trap  # positive → trapezoid preferred

    if delta_bic > 2.0:
        preferred = "trapezoid"
    elif delta_bic < -2.0:
        preferred = "box"
    else:
        preferred = "indeterminate"

    return TrapezoidBoxResult(
        period_days=period_days,
        epoch_bjd=epoch_bjd,
        duration_hours=duration_hours,
        depth_ppm=depth_ppm,
        chi2_box=round(chi2_box, 4),
        chi2_trapezoid=round(best_chi2_trap, 4),
        delta_chi2=round(delta_chi2, 4),
        delta_bic=round(delta_bic, 4),
        best_ingress_fraction=round(best_frac, 3),
        preferred_model=preferred,
        flag="OK",
    )


def format_trapezoid_box_result(result: TrapezoidBoxResult) -> str:
    """Format trapezoid/box comparison result as Markdown."""
    lines = [
        "## Trapezoid vs Box Model Comparison",
        "",
        f"- Period: {result.period_days:.4f} days",
        f"- Duration: {result.duration_hours:.2f} hours",
        f"- Depth: {result.depth_ppm:.1f} ppm",
        f"- χ² (box): {result.chi2_box:.2f}",
        f"- χ² (trapezoid): {result.chi2_trapezoid:.2f}",
        f"- Δχ²: {result.delta_chi2:.2f}",
        f"- ΔBIC: {result.delta_bic:.2f}",
        f"- Best ingress fraction: {result.best_ingress_fraction:.2f}",
        f"- Preferred model: **{result.preferred_model}**",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="trapezoid_box_comparator",
        description="Compare box vs trapezoid transit model via ΔBIC.",
    )
    parser.add_argument("period_days", type=float)
    parser.add_argument("epoch_bjd", type=float)
    parser.add_argument("--duration-hours", type=float, default=2.0)
    parser.add_argument("--depth-ppm", type=float, default=1000.0)
    args = parser.parse_args(argv)

    result = compare_trapezoid_box(
        [], [], args.period_days, args.epoch_bjd,
        duration_hours=args.duration_hours,
        depth_ppm=args.depth_ppm,
    )
    print(format_trapezoid_box_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
