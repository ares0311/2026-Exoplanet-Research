"""Compute O-C transit timing residuals with linear/quadratic fit and TTV flag."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class OcPoint:
    index: int          # transit number (integer)
    midpoint_bjd: float
    predicted_bjd: float
    oc_minutes: float   # (observed - calculated) in minutes


@dataclass(frozen=True)
class OcResidualResult:
    n_transits: int
    period_days: float
    epoch_bjd: float
    rms_oc_minutes: float
    linear_slope_min_per_epoch: float    # dO-C/dN from linear fit
    quadratic_coeff_min_per_epoch2: float | None
    ttv_flag: bool                        # RMS > ttv_threshold_minutes
    ttv_threshold_minutes: float
    points: tuple[OcPoint, ...]
    flag: str


def compute_oc_residuals(
    midpoints_bjd: list[float],
    period_days: float,
    epoch_bjd: float,
    ttv_threshold_minutes: float = 5.0,
    fit_quadratic: bool = False,
) -> OcResidualResult:
    """
    Compute O-C transit timing residuals.

    For each observed midpoint t_i, finds the nearest predicted transit
    T_pred = epoch + N * period, computes residual (t_i - T_pred) in minutes,
    then fits a linear (and optionally quadratic) trend to the O-C values.

    Parameters
    ----------
    midpoints_bjd:       List of observed transit midpoint times (BJD).
    period_days:         Best-fit orbital period (days).
    epoch_bjd:           Reference transit epoch (BJD).
    ttv_threshold_minutes: RMS above this triggers ttv_flag.
    fit_quadratic:       If True, also fit a quadratic to the O-C residuals.
    """
    if not math.isfinite(period_days) or period_days <= 0:
        return OcResidualResult(
            n_transits=0, period_days=period_days, epoch_bjd=epoch_bjd,
            rms_oc_minutes=float("nan"), linear_slope_min_per_epoch=float("nan"),
            quadratic_coeff_min_per_epoch2=None, ttv_flag=False,
            ttv_threshold_minutes=ttv_threshold_minutes, points=(), flag="INVALID_PERIOD",
        )
    if not math.isfinite(epoch_bjd):
        return OcResidualResult(
            n_transits=0, period_days=period_days, epoch_bjd=epoch_bjd,
            rms_oc_minutes=float("nan"), linear_slope_min_per_epoch=float("nan"),
            quadratic_coeff_min_per_epoch2=None, ttv_flag=False,
            ttv_threshold_minutes=ttv_threshold_minutes, points=(), flag="INVALID_EPOCH",
        )

    finite_pts = [t for t in midpoints_bjd if math.isfinite(t)]
    if len(finite_pts) < 2:
        return OcResidualResult(
            n_transits=len(finite_pts), period_days=period_days, epoch_bjd=epoch_bjd,
            rms_oc_minutes=float("nan"), linear_slope_min_per_epoch=float("nan"),
            quadratic_coeff_min_per_epoch2=None, ttv_flag=False,
            ttv_threshold_minutes=ttv_threshold_minutes, points=(), flag="INSUFFICIENT_TRANSITS",
        )

    points: list[OcPoint] = []
    for t in sorted(finite_pts):
        n = round((t - epoch_bjd) / period_days)
        t_pred = epoch_bjd + n * period_days
        oc_min = (t - t_pred) * 1440.0
        points.append(OcPoint(index=n, midpoint_bjd=t, predicted_bjd=t_pred, oc_minutes=oc_min))

    ns = [p.index for p in points]
    ocs = [p.oc_minutes for p in points]

    # Linear fit: O-C = a + b*N  via least squares
    n_pts = len(ns)
    sum_n = sum(ns)
    sum_n2 = sum(ni ** 2 for ni in ns)
    sum_oc = sum(ocs)
    sum_noc = sum(ni * oc for ni, oc in zip(ns, ocs, strict=True))
    denom = n_pts * sum_n2 - sum_n ** 2
    b_lin = 0.0 if denom == 0 else (n_pts * sum_noc - sum_n * sum_oc) / denom

    oc_mean = sum_oc / n_pts
    residuals_lin = [
        oc - (oc_mean + b_lin * (ni - sum_n / n_pts))
        for ni, oc in zip(ns, ocs, strict=True)
    ]
    rms = math.sqrt(sum(r ** 2 for r in residuals_lin) / n_pts)

    # Quadratic fit (optional): O-C = a + b*N + c*N²
    quad_c: float | None = None
    if fit_quadratic and n_pts >= 3:
        sum_n3 = sum(ni ** 3 for ni in ns)
        sum_n4 = sum(ni ** 4 for ni in ns)
        sum_n2oc = sum(ni ** 2 * oc for ni, oc in zip(ns, ocs, strict=True))
        # Normal equations via Cramer's rule (3×3 system)
        mat = [
            [float(n_pts), float(sum_n), float(sum_n2)],
            [float(sum_n), float(sum_n2), float(sum_n3)],
            [float(sum_n2), float(sum_n3), float(sum_n4)],
        ]
        rhs = [float(sum_oc), float(sum_noc), float(sum_n2oc)]
        # Gaussian elimination
        for col in range(3):
            pivot = mat[col][col]
            if abs(pivot) < 1e-15:
                break
            for row in range(col + 1, 3):
                factor = mat[row][col] / pivot
                for k in range(3):
                    mat[row][k] -= factor * mat[col][k]
                rhs[row] -= factor * rhs[col]
        # Back-substitution
        coeffs = [0.0, 0.0, 0.0]
        for i in range(2, -1, -1):
            s = rhs[i]
            for j in range(i + 1, 3):
                s -= mat[i][j] * coeffs[j]
            if abs(mat[i][i]) > 1e-15:
                coeffs[i] = s / mat[i][i]
        quad_c = round(coeffs[2], 8)

    return OcResidualResult(
        n_transits=n_pts,
        period_days=period_days,
        epoch_bjd=epoch_bjd,
        rms_oc_minutes=round(rms, 4),
        linear_slope_min_per_epoch=round(b_lin, 6),
        quadratic_coeff_min_per_epoch2=quad_c,
        ttv_flag=rms > ttv_threshold_minutes,
        ttv_threshold_minutes=ttv_threshold_minutes,
        points=tuple(points),
        flag="OK",
    )


def format_oc_result(r: OcResidualResult) -> str:
    def _f(v: float | None, fmt: str = ".4f") -> str:
        if v is None:
            return "N/A"
        return format(v, fmt) if math.isfinite(v) else "N/A"

    lines = [
        "| Parameter | Value |\n|---|---|",
        f"| N transits | {r.n_transits} |",
        f"| Period (d) | {_f(r.period_days, '.6f')} |",
        f"| RMS O-C (min) | {_f(r.rms_oc_minutes)} |",
        f"| Linear slope (min/epoch) | {_f(r.linear_slope_min_per_epoch, '.6f')} |",
        f"| Quadratic coeff | {_f(r.quadratic_coeff_min_per_epoch2, '.8f')} |",
        f"| TTV flag | {r.ttv_flag} |",
        f"| Flag | {r.flag} |",
        "",
        "| Transit # | Midpoint (BJD) | Predicted (BJD) | O-C (min) |",
        "|---|---|---|---|",
    ]
    for pt in r.points:
        lines.append(
            f"| {pt.index} | {pt.midpoint_bjd:.6f} | {pt.predicted_bjd:.6f} | "
            f"{pt.oc_minutes:.4f} |"
        )
    return "\n".join(lines)


def _cli() -> int:
    p = argparse.ArgumentParser(description="Compute transit O-C residuals.")
    p.add_argument("period_days", type=float)
    p.add_argument("epoch_bjd", type=float)
    p.add_argument("midpoints", type=float, nargs="+")
    p.add_argument("--ttv-threshold-minutes", type=float, default=5.0)
    p.add_argument("--quadratic", action="store_true")
    args = p.parse_args()
    r = compute_oc_residuals(
        args.midpoints, args.period_days, args.epoch_bjd,
        ttv_threshold_minutes=args.ttv_threshold_minutes,
        fit_quadratic=args.quadratic,
    )
    print(format_oc_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
