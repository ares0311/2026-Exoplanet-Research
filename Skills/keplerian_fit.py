"""Fit a symmetric trapezoidal transit model to phase-folded flux.

Uses scipy.optimize.minimize (Nelder-Mead) to find the best-fit depth,
duration, and ingress fraction.

Public API
----------
fit_trapezoid(phase, flux, *, period_days, n_points) -> TrapezoidFit
trapezoid_model(phase, depth_ppm, duration_phase, ingress_frac) -> ndarray
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrapezoidFit:
    depth_ppm: float
    duration_hours: float
    ingress_fraction: float
    chi2_reduced: float
    converged: bool
    period_days: float


def trapezoid_model(
    phase: np.ndarray,
    depth_ppm: float,
    duration_phase: float,
    ingress_frac: float,
) -> np.ndarray:
    """Evaluate a symmetric trapezoid transit model.

    The model is flat at 1.0 outside the transit, linearly falls during ingress
    and rises during egress, and is flat at ``1 - depth`` during the flat
    bottom.

    Args:
        phase: Phase array in [−0.5, 0.5).
        depth_ppm: Transit depth in parts per million.
        duration_phase: Total transit duration as a fraction of period.
        ingress_frac: Fraction of the transit that is ingress/egress (each).
            Clipped to [0.001, 0.499].

    Returns:
        Relative flux array (mean = 1.0 out of transit).
    """
    depth = depth_ppm * 1e-6
    half_dur = duration_phase / 2.0
    ingress_frac = max(0.001, min(0.499, ingress_frac))
    ingress_dur = ingress_frac * duration_phase

    model = np.ones_like(phase, dtype=float)
    abs_ph = np.abs(phase)

    in_full = abs_ph <= half_dur - ingress_dur
    in_ingress = (abs_ph > half_dur - ingress_dur) & (abs_ph <= half_dur)

    model[in_full] = 1.0 - depth
    t_norm = (abs_ph[in_ingress] - (half_dur - ingress_dur)) / ingress_dur
    model[in_ingress] = 1.0 - depth * (1.0 - t_norm)

    return model


def fit_trapezoid(
    phase: np.ndarray,
    flux: np.ndarray,
    *,
    period_days: float,
    flux_err: np.ndarray | None = None,
) -> TrapezoidFit:
    """Fit a symmetric trapezoidal transit model to phase-folded data.

    Args:
        phase: Phase array in [−0.5, 0.5); typically from ``phase_fold()``.
        flux: Relative flux array (mean ≈ 1.0).
        period_days: Orbital period used to convert phase units to hours.
        flux_err: Per-point flux uncertainties.  If ``None``, all points
            are weighted equally.

    Returns:
        :class:`TrapezoidFit` with best-fit parameters.

    Raises:
        ValueError: If ``phase`` and ``flux`` have different lengths.
    """
    from scipy.optimize import minimize  # noqa: PLC0415

    phase = np.asarray(phase, dtype=float)
    flux = np.asarray(flux, dtype=float)

    if phase.shape != flux.shape:
        raise ValueError("phase and flux must have the same shape")

    if flux_err is not None:
        weights = 1.0 / np.maximum(np.asarray(flux_err, dtype=float), 1e-10) ** 2
    else:
        weights = np.ones_like(flux)

    depth_guess = max(1.0 - float(np.min(flux)), 1e-6) * 1e6  # ppm
    in_transit = flux < (1.0 + float(np.min(flux))) / 2.0
    dur_guess = float(np.sum(in_transit)) / len(phase) if np.any(in_transit) else 0.05

    x0 = [depth_guess, max(0.005, dur_guess), 0.20]

    def _residuals(x: list[float]) -> float:
        d, dur, ing = x
        if d <= 0 or dur <= 0 or dur >= 0.5 or ing <= 0 or ing >= 0.5:
            return 1e10
        model = trapezoid_model(phase, d, dur, ing)
        return float(np.sum(weights * (flux - model) ** 2))

    result = minimize(
        _residuals,
        x0,
        method="Nelder-Mead",
        options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 5000},
    )

    d, dur, ing = result.x
    d = max(0.0, d)
    dur_hours = dur * period_days * 24.0
    model_best = trapezoid_model(phase, d, dur, ing)
    n = len(flux)
    dof = max(1, n - 3)
    chi2 = float(np.sum(weights * (flux - model_best) ** 2))
    chi2_red = chi2 / dof

    return TrapezoidFit(
        depth_ppm=d,
        duration_hours=dur_hours,
        ingress_fraction=ing,
        chi2_reduced=chi2_red,
        converged=result.success,
        period_days=period_days,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(
        prog="keplerian_fit",
        description="Fit a trapezoidal transit model to a phase-folded candidate.",
    )
    parser.add_argument("input", type=Path, metavar="FILE",
                        help="Pipeline JSON output containing phase-folded flux.")
    parser.add_argument("--period", type=float, required=True, metavar="DAYS",
                        help="Orbital period in days.")
    args = parser.parse_args(argv)

    data = json.loads(args.input.read_text())
    rows = data if isinstance(data, list) else [data]
    for row in rows:
        cid = row.get("candidate_id", "unknown")
        phase = row.get("phase")
        flux = row.get("flux")
        if phase is None or flux is None:
            print(f"{cid}: no phase/flux data — skipping")
            continue
        fit = fit_trapezoid(np.array(phase), np.array(flux), period_days=args.period)
        print(f"{cid}: depth={fit.depth_ppm:.1f} ppm  "
              f"dur={fit.duration_hours:.2f} h  "
              f"ingress={fit.ingress_fraction:.2f}  "
              f"chi2r={fit.chi2_reduced:.3f}  "
              f"converged={fit.converged}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
