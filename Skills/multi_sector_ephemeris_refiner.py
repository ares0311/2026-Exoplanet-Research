"""Refine transit period and epoch from multi-sector transit midpoint measurements."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class EphemerisRefinementResult:
    n_transits: int
    initial_period_days: float
    refined_period_days: float
    refined_epoch_bjd: float
    period_uncertainty_days: float
    epoch_uncertainty_days: float
    rms_oc_minutes: float
    max_oc_minutes: float
    flag: str


def refine_ephemeris(
    midpoints_bjd: list[float],
    initial_period_days: float,
    initial_epoch_bjd: float | None = None,
    midpoint_errors_days: list[float] | None = None,
) -> EphemerisRefinementResult:
    """
    Refine transit ephemeris using a weighted linear least-squares O-C fit.

    Model: T(n) = T0 + n * P  where n is the transit epoch integer.
    Solves for refined T0 and P via weighted least-squares.

    Parameters
    ----------
    midpoints_bjd:         Observed transit midpoints in BJD.
    initial_period_days:   Initial period estimate (used to assign epoch numbers).
    initial_epoch_bjd:     Initial epoch T0; if None, uses first midpoint.
    midpoint_errors_days:  Measurement errors; if None, uses uniform weights.
    """
    n_t = len(midpoints_bjd)
    if n_t < 2:
        return EphemerisRefinementResult(n_t, initial_period_days, float("nan"),
                                         float("nan"), float("nan"), float("nan"),
                                         float("nan"), float("nan"), "INSUFFICIENT_TRANSITS")
    if not math.isfinite(initial_period_days) or initial_period_days <= 0:
        return EphemerisRefinementResult(n_t, initial_period_days, float("nan"),
                                         float("nan"), float("nan"), float("nan"),
                                         float("nan"), float("nan"), "INVALID_PERIOD")

    t0_init = initial_epoch_bjd if initial_epoch_bjd is not None else midpoints_bjd[0]

    ns = [round((t - t0_init) / initial_period_days) for t in midpoints_bjd]

    if midpoint_errors_days is not None and len(midpoint_errors_days) == n_t:
        weights = [1.0 / (e ** 2) if math.isfinite(e) and e > 0 else 1.0
                   for e in midpoint_errors_days]
    else:
        weights = [1.0] * n_t

    sw = sum(weights)
    swx = sum(w * n for w, n in zip(weights, ns, strict=True))
    swy = sum(w * t for w, t in zip(weights, midpoints_bjd, strict=True))
    swxx = sum(w * n ** 2 for w, n in zip(weights, ns, strict=True))
    swxy = sum(w * n * t for w, n, t in zip(weights, ns, midpoints_bjd, strict=True))

    denom = sw * swxx - swx ** 2
    if abs(denom) < 1e-15:
        return EphemerisRefinementResult(n_t, initial_period_days, float("nan"),
                                         float("nan"), float("nan"), float("nan"),
                                         float("nan"), float("nan"), "DEGENERATE_FIT")

    refined_p = (sw * swxy - swx * swy) / denom
    refined_t0 = (swy - refined_p * swx) / sw

    residuals_min = [
        (t - (refined_t0 + n * refined_p)) * 1440.0
        for t, n in zip(midpoints_bjd, ns, strict=True)
    ]
    rms_oc = math.sqrt(sum(r ** 2 for r in residuals_min) / n_t)
    max_oc = max(abs(r) for r in residuals_min)

    var = sum(w * (t - (refined_t0 + n * refined_p)) ** 2
              for w, t, n in zip(weights, midpoints_bjd, ns, strict=True))
    s2 = var / max(1, n_t - 2)
    sigma_p = math.sqrt(s2 * sw / denom)
    sigma_t0 = math.sqrt(s2 * swxx / denom)

    return EphemerisRefinementResult(
        n_transits=n_t,
        initial_period_days=initial_period_days,
        refined_period_days=round(refined_p, 8),
        refined_epoch_bjd=round(refined_t0, 8),
        period_uncertainty_days=round(sigma_p, 8),
        epoch_uncertainty_days=round(sigma_t0, 8),
        rms_oc_minutes=round(rms_oc, 4),
        max_oc_minutes=round(max_oc, 4),
        flag="OK",
    )


def format_ephemeris_refinement_result(r: EphemerisRefinementResult) -> str:
    def _f(v: float, fmt: str = ".8f") -> str:
        return format(v, fmt) if math.isfinite(v) else "N/A"

    return (
        f"| Parameter | Value |\n|---|---|\n"
        f"| N transits | {r.n_transits} |\n"
        f"| Initial period (days) | {_f(r.initial_period_days)} |\n"
        f"| Refined period (days) | {_f(r.refined_period_days)} |\n"
        f"| Period uncertainty (days) | {_f(r.period_uncertainty_days)} |\n"
        f"| Refined epoch (BJD) | {_f(r.refined_epoch_bjd)} |\n"
        f"| Epoch uncertainty (days) | {_f(r.epoch_uncertainty_days)} |\n"
        f"| RMS O-C (min) | {_f(r.rms_oc_minutes, '.4f')} |\n"
        f"| Max O-C (min) | {_f(r.max_oc_minutes, '.4f')} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Refine transit ephemeris from multi-sector midpoints.")
    p.add_argument("initial_period_days", type=float)
    p.add_argument("midpoints_bjd", help="Comma-separated BJD transit midpoints")
    p.add_argument("--epoch-bjd", type=float, default=None)
    args = p.parse_args()
    mids = [float(x) for x in args.midpoints_bjd.split(",")]
    r = refine_ephemeris(mids, args.initial_period_days, initial_epoch_bjd=args.epoch_bjd)
    print(format_ephemeris_refinement_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
