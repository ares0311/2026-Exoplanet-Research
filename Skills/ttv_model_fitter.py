"""Fit a sinusoidal TTV model to O-C residuals."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TTVFitResult:
    n_points: int
    amplitude_minutes: float
    period_transits: float
    phase_rad: float
    offset_minutes: float
    rms_residual_minutes: float
    reduced_chi2: float
    flag: str


def fit_ttv_sinusoid(
    oc_minutes: list[float],
    transit_numbers: list[int] | None = None,
    noise_minutes: float = 1.0,
) -> TTVFitResult:
    """
    Fit O-C = A·sin(2π·n/P_ttv + φ) + c to transit timing residuals.

    Uses a grid search over TTV period then linear least-squares for amplitude/phase/offset.
    """
    n = len(oc_minutes)
    if n < 4:
        return TTVFitResult(
            n_points=n, amplitude_minutes=float("nan"), period_transits=float("nan"),
            phase_rad=float("nan"), offset_minutes=float("nan"),
            rms_residual_minutes=float("nan"), reduced_chi2=float("nan"),
            flag="INSUFFICIENT_DATA",
        )
    if noise_minutes <= 0.0:
        return TTVFitResult(
            n_points=n, amplitude_minutes=float("nan"), period_transits=float("nan"),
            phase_rad=float("nan"), offset_minutes=float("nan"),
            rms_residual_minutes=float("nan"), reduced_chi2=float("nan"),
            flag="INVALID_NOISE",
        )

    ns = list(transit_numbers) if transit_numbers is not None else list(range(n))
    if len(ns) != n:
        return TTVFitResult(
            n_points=n, amplitude_minutes=float("nan"), period_transits=float("nan"),
            phase_rad=float("nan"), offset_minutes=float("nan"),
            rms_residual_minutes=float("nan"), reduced_chi2=float("nan"),
            flag="LENGTH_MISMATCH",
        )

    # Grid search: TTV period from 2 to N/2 transits
    best_chi2 = float("inf")
    best_amp = 0.0
    best_period = float(n)
    best_phase = 0.0
    best_offset = sum(oc_minutes) / n

    p_min = 2.0
    p_max = max(float(n), p_min + 1.0)
    n_grid = 200
    dp = (p_max - p_min) / n_grid

    for i in range(n_grid + 1):
        p_try = p_min + i * dp
        omega = 2.0 * math.pi / p_try
        # Build design matrix [sin, cos, 1] for least-squares
        sn = [math.sin(omega * float(k)) for k in ns]
        cs = [math.cos(omega * float(k)) for k in ns]
        # Normal equations for [a, b, c] where fit = a*sin + b*cos + c
        ss = sum(x * x for x in sn)
        cc = sum(x * x for x in cs)
        sc = sum(sn[j] * cs[j] for j in range(n))
        s1 = sum(sn)
        c1 = sum(cs)
        sy = sum(oc_minutes[j] * sn[j] for j in range(n))
        cy = sum(oc_minutes[j] * cs[j] for j in range(n))
        y1 = sum(oc_minutes)
        mat = [
            [ss, sc, s1],
            [sc, cc, c1],
            [s1, c1, float(n)],
        ]
        rhs = [sy, cy, y1]
        try:
            a_coef, b_coef, c_coef = _solve3(mat, rhs)
        except ZeroDivisionError:
            continue
        res = [
            oc_minutes[j] - (a_coef * sn[j] + b_coef * cs[j] + c_coef)
            for j in range(n)
        ]
        chi2 = sum(r * r for r in res) / noise_minutes**2
        if chi2 < best_chi2:
            best_chi2 = chi2
            best_amp = math.sqrt(a_coef**2 + b_coef**2)
            best_period = p_try
            best_phase = math.atan2(b_coef, a_coef)
            best_offset = c_coef

    dof = max(n - 3, 1)
    reduced_chi2 = best_chi2 / dof

    omega_best = 2.0 * math.pi / best_period
    sn = [math.sin(omega_best * float(k)) for k in ns]
    cs = [math.cos(omega_best * float(k)) for k in ns]
    a_coef = best_amp * math.cos(best_phase)
    b_coef = best_amp * math.sin(best_phase)
    res = [
        oc_minutes[j] - (a_coef * sn[j] + b_coef * cs[j] + best_offset)
        for j in range(n)
    ]
    rms = math.sqrt(sum(r * r for r in res) / n)

    return TTVFitResult(
        n_points=n,
        amplitude_minutes=round(best_amp, 4),
        period_transits=round(best_period, 4),
        phase_rad=round(best_phase, 4),
        offset_minutes=round(best_offset, 4),
        rms_residual_minutes=round(rms, 4),
        reduced_chi2=round(reduced_chi2, 4),
        flag="OK",
    )


def _solve3(mat: list[list[float]], rhs: list[float]) -> tuple[float, float, float]:
    a, b, c = mat[0]
    d, e, f = mat[1]
    g, h, i = mat[2]
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(det) < 1e-30:
        raise ZeroDivisionError
    x = (rhs[0] * (e * i - f * h) - b * (rhs[1] * i - f * rhs[2])
         + c * (rhs[1] * h - e * rhs[2])) / det
    y = (a * (rhs[1] * i - f * rhs[2]) - rhs[0] * (d * i - f * g)
         + c * (d * rhs[2] - rhs[1] * g)) / det
    z = (a * (e * rhs[2] - rhs[1] * h) - b * (d * rhs[2] - rhs[1] * g)
         + rhs[0] * (d * h - e * g)) / det
    return x, y, z


def format_ttv_fit_result(r: TTVFitResult) -> str:
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| N transits | {r.n_points} |\n"
        f"| TTV amplitude (min) | {r.amplitude_minutes:.4f} |\n"
        f"| TTV period (transits) | {r.period_transits:.4f} |\n"
        f"| Phase (rad) | {r.phase_rad:.4f} |\n"
        f"| Offset (min) | {r.offset_minutes:.4f} |\n"
        f"| RMS residual (min) | {r.rms_residual_minutes:.4f} |\n"
        f"| Reduced χ² | {r.reduced_chi2:.4f} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Fit sinusoidal TTV model to O-C data.")
    p.add_argument("oc_minutes", type=float, nargs="+", help="O-C residuals in minutes")
    p.add_argument("--noise", type=float, default=1.0, help="Timing noise in minutes")
    args = p.parse_args()
    r = fit_ttv_sinusoid(args.oc_minutes, noise_minutes=args.noise)
    print(format_ttv_fit_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
