"""Fit a limb-darkened transit model to phase-folded flux.

Uses a quadratic limb-darkening trapezoid approximation that avoids
requiring batman/PyTransit while still capturing the key morphological
differences between a genuine planet transit (curved limb-darkened
profile) and a box-shaped instrumental dip.

Public API
----------
TransitModelResult(period_days, epoch_bjd, depth_ppm, duration_hours,
                   impact_param, ld_u1, ld_u2, chi2_reduced, converged, rms_residual)
fit_transit_model(time, flux, period, epoch, *, flux_err, duration_days,
                  u1, u2, model_fn) -> TransitModelResult
transit_model(phase, depth, duration_days, period, *, u1, u2) -> np.ndarray
format_model_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TransitModelResult:
    period_days: float
    epoch_bjd: float
    depth_ppm: float
    duration_hours: float
    impact_param: float         # 0 = central, 1 = grazing
    ld_u1: float                # quadratic LD coefficient u1
    ld_u2: float                # quadratic LD coefficient u2
    chi2_reduced: float
    converged: bool
    rms_residual: float         # RMS of (data − model)


def _ld_correction(mu: float, u1: float, u2: float) -> float:
    """Quadratic limb-darkening correction factor at cos(theta) = mu."""
    return 1.0 - u1 * (1.0 - mu) - u2 * (1.0 - mu) ** 2


def transit_model(
    phase: list[float] | Any,
    depth: float,
    duration_days: float,
    period: float,
    *,
    u1: float = 0.4,
    u2: float = 0.2,
    ingress_fraction: float = 0.15,
) -> list[float]:
    """Limb-darkened trapezoid transit model.

    Args:
        phase: Phase array in days relative to mid-transit (centred on 0).
        depth: Fractional depth (e.g. 0.01 = 1%).
        duration_days: Full transit duration (first to last contact).
        period: Orbital period (days); not used for shape, kept for API clarity.
        u1, u2: Quadratic limb-darkening coefficients.
        ingress_fraction: Ingress/egress as fraction of duration.

    Returns:
        Flux array (normalised; 1.0 out-of-transit, < 1.0 in-transit).
    """
    try:
        import numpy as np
        ph = np.asarray(phase, dtype=float)
    except ImportError:
        ph = phase  # type: ignore[assignment]

    half_dur = duration_days / 2.0
    ing = ingress_fraction * duration_days

    result = []
    for p in ph if hasattr(ph, "__iter__") else [ph]:
        p = float(p)
        abs_p = abs(p)
        if abs_p >= half_dur:
            result.append(1.0)
        elif abs_p <= half_dur - ing:
            # Full eclipse — limb-darkening peaks at centre (mu=1)
            mu = math.sqrt(max(0.0, 1.0 - (p / (half_dur - ing)) ** 2))
            ld = _ld_correction(mu, u1, u2)
            result.append(1.0 - depth * ld)
        else:
            # Ingress / egress ramp
            frac = (half_dur - abs_p) / max(ing, 1e-9)
            result.append(1.0 - depth * frac)
    return result


def fit_transit_model(
    time: list[float],
    flux: list[float],
    period: float,
    epoch: float,
    *,
    flux_err: list[float] | None = None,
    duration_days: float = 0.1,
    u1: float = 0.4,
    u2: float = 0.2,
) -> TransitModelResult:
    """Fit a limb-darkened trapezoid transit model using scipy Nelder-Mead.

    Args:
        time: BJD time array.
        flux: Normalised flux array.
        period: Fixed orbital period (days).
        epoch: Fixed mid-transit epoch (BJD).
        flux_err: Per-cadence uncertainties.  If None, uniform weights.
        duration_days: Initial guess for transit duration.
        u1, u2: Fixed quadratic LD coefficients.

    Returns:
        :class:`TransitModelResult`.
    """
    import numpy as np
    from scipy.optimize import minimize  # type: ignore[import]

    t = np.asarray(time, dtype=float)
    f = np.asarray(flux, dtype=float)
    w = np.ones_like(f) if flux_err is None else 1.0 / np.clip(
        np.asarray(flux_err, dtype=float), 1e-9, None
    )

    # Phase-fold
    phase = (t - epoch) % period
    phase[phase > period / 2] -= period

    def _model(depth: float, dur: float, b: float) -> np.ndarray:
        dur = max(1e-4, min(dur, period * 0.5))
        m = np.array(
            transit_model(phase.tolist(), depth, dur, period, u1=u1, u2=u2),
            dtype=float,
        )
        return m

    def _chi2(params: np.ndarray) -> float:
        depth, dur, b = params
        if depth < 0 or dur < 0 or not (0 <= b <= 1):
            return 1e10
        m = _model(depth, dur, b)
        res = f - m
        return float(np.sum((res * w) ** 2))

    init_depth = max(1.0 - float(np.min(f)), 1e-4)
    x0 = np.array([init_depth, duration_days, 0.1])
    res = minimize(_chi2, x0, method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 5000})

    depth_fit, dur_fit, b_fit = res.x
    depth_fit = abs(depth_fit)
    dur_fit   = abs(dur_fit)

    model_flux = np.array(
        transit_model(phase.tolist(), depth_fit, dur_fit, period, u1=u1, u2=u2),
        dtype=float,
    )
    residuals = f - model_flux
    dof = max(len(f) - 3, 1)
    chi2_r = float(np.sum((residuals * w) ** 2)) / dof
    rms = float(np.std(residuals))

    return TransitModelResult(
        period_days=period,
        epoch_bjd=epoch,
        depth_ppm=depth_fit * 1e6,
        duration_hours=dur_fit * 24.0,
        impact_param=float(np.clip(b_fit, 0.0, 1.0)),
        ld_u1=u1,
        ld_u2=u2,
        chi2_reduced=chi2_r,
        converged=bool(res.success),
        rms_residual=rms,
    )


def format_model_result(result: TransitModelResult) -> str:
    """Format transit model fit as Markdown."""
    lines = [
        "## Transit Model Fit",
        "",
        f"- Period: {result.period_days:.4f} d",
        f"- Epoch (BJD): {result.epoch_bjd:.4f}",
        f"- Depth: {result.depth_ppm:.0f} ppm",
        f"- Duration: {result.duration_hours:.2f} h",
        f"- Impact param: {result.impact_param:.3f}",
        f"- LD (u1, u2): ({result.ld_u1:.2f}, {result.ld_u2:.2f})",
        f"- χ²_red: {result.chi2_reduced:.3f}",
        f"- RMS residual: {result.rms_residual:.6f}",
        f"- Converged: {result.converged}",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(
        prog="transit_modeler",
        description="Fit a limb-darkened transit model to phase-folded flux.",
    )
    parser.add_argument("--lc", required=True, metavar="JSON",
                        help="JSON file with 'time' and 'flux' keys.")
    parser.add_argument("--period", type=float, required=True)
    parser.add_argument("--epoch", type=float, required=True)
    parser.add_argument("--duration", type=float, default=0.1,
                        help="Initial duration guess in days.")
    parser.add_argument("--u1", type=float, default=0.4)
    parser.add_argument("--u2", type=float, default=0.2)
    args = parser.parse_args(argv)

    lc = json.loads(Path(args.lc).read_text())
    result = fit_transit_model(
        lc["time"], lc["flux"],
        args.period, args.epoch,
        duration_days=args.duration,
        u1=args.u1,
        u2=args.u2,
    )
    print(format_model_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
