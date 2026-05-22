"""Fit and subtract a polynomial (or piecewise-constant) trend from a light curve.

Complements ``detrending_comparator.py`` which uses Savitzky-Golay windows.
This module fits a global or per-segment polynomial to the out-of-transit
flux and subtracts it, returning residuals normalised around 1.0.

Public API
----------
DetrenderResult(degree, n_segments, coefficients, rms_before, rms_after,
                detrended_flux, flag)
fit_polynomial_trend(time, flux, *, degree, n_segments,
                     mask) -> DetrenderResult
apply_detrend(time, flux, result) -> list[float]
format_detrend_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DetrenderResult:
    degree: int
    n_segments: int
    coefficients: tuple[tuple[float, ...], ...]   # one tuple per segment
    rms_before: float | None
    rms_after: float | None
    detrended_flux: tuple[float, ...]
    flag: str  # "OK" | "INSUFFICIENT" | "INVALID"


def _rms(values: list[float]) -> float:
    n = len(values)
    if n == 0:
        return 0.0
    m = sum(values) / n
    return math.sqrt(sum((v - m) ** 2 for v in values) / n)


def _polyfit(x: list[float], y: list[float], deg: int) -> list[float]:
    """Least-squares polynomial fit via normal equations."""
    n = len(x)
    if n == 0 or deg < 0:
        return [0.0]
    deg = min(deg, n - 1)
    # Build Vandermonde matrix A (n × (deg+1))
    p = deg + 1
    A = [[xi ** j for j in range(p)] for xi in x]
    # Normal equations: (A^T A) c = A^T y
    AtA = [[sum(A[i][j] * A[i][k] for i in range(n)) for k in range(p)] for j in range(p)]
    Aty = [sum(A[i][j] * y[i] for i in range(n)) for j in range(p)]
    # Gaussian elimination with partial pivoting
    mat = [row[:] + [Aty[r]] for r, row in enumerate(AtA)]
    for col in range(p):
        # Pivot
        max_row = max(range(col, p), key=lambda r: abs(mat[r][col]))
        mat[col], mat[max_row] = mat[max_row], mat[col]
        pivot = mat[col][col]
        if abs(pivot) < 1e-30:
            continue
        for row in range(col + 1, p):
            factor = mat[row][col] / pivot
            mat[row] = [mat[row][k] - factor * mat[col][k] for k in range(p + 1)]
    # Back-substitution
    coeffs = [0.0] * p
    for i in range(p - 1, -1, -1):
        if abs(mat[i][i]) < 1e-30:
            coeffs[i] = 0.0
        else:
            num = mat[i][p] - sum(mat[i][j] * coeffs[j] for j in range(i + 1, p))
            coeffs[i] = num / mat[i][i]
    return coeffs


def _polyval(coeffs: list[float], x: float) -> float:
    return sum(c * x ** j for j, c in enumerate(coeffs))


def fit_polynomial_trend(
    time: list[float],
    flux: list[float],
    *,
    degree: int = 2,
    n_segments: int = 1,
    mask: list[bool] | None = None,
) -> DetrenderResult:
    """Fit a polynomial trend and return detrended flux.

    Args:
        time: Time array.
        flux: Flux array, same length as time.
        degree: Polynomial degree (0=constant, 1=linear, 2=quadratic).
        n_segments: Number of equal-width time segments (1=global fit).
        mask: Boolean mask; True = include in fit (e.g. OOT cadences).
              Defaults to all True.

    Returns:
        :class:`DetrenderResult`.
    """
    n = len(time)
    if n < 2 or len(flux) != n:
        return DetrenderResult(degree, n_segments, (), None, None, (), "INVALID")
    if degree < 0 or n_segments < 1:
        return DetrenderResult(degree, n_segments, (), None, None, (), "INVALID")

    if mask is None:
        mask = [True] * n

    rms_before = _rms(list(flux))

    # Split into segments
    t_min = min(time)
    t_max = max(time)
    seg_width = (t_max - t_min) / n_segments if n_segments > 1 else 1.0

    all_coeffs: list[tuple[float, ...]] = []
    detrended = list(flux)

    for seg in range(n_segments):
        t_lo = t_min + seg * seg_width
        t_hi = t_min + (seg + 1) * seg_width + 1e-9

        seg_idx = [i for i in range(n) if t_lo <= time[i] < t_hi]
        fit_idx = [i for i in seg_idx if mask[i]]

        if len(fit_idx) <= degree:
            # Fall back to constant = segment mean
            vals = [flux[i] for i in seg_idx] or [1.0]
            mean_val = sum(vals) / len(vals)
            coeffs = [mean_val] + [0.0] * degree
        else:
            t_fit = [time[i] for i in fit_idx]
            f_fit = [flux[i] for i in fit_idx]
            # Normalise time to [-1, 1] for numerical stability
            t_cen = sum(t_fit) / len(t_fit)
            t_scale = max(max(t_fit) - t_cen, 1e-9)
            t_norm = [(t - t_cen) / t_scale for t in t_fit]
            coeffs = _polyfit(t_norm, f_fit, degree)
            # Store coefficients with offset/scale info
            all_coeffs.append(tuple(coeffs))
            # Apply detrend to segment
            for i in seg_idx:
                t_n = (time[i] - t_cen) / t_scale
                trend = _polyval(coeffs, t_n)
                t_norms = [_polyval(coeffs, (time[j] - t_cen) / t_scale) for j in fit_idx]
                mean_trend = sum(t_norms) / len(fit_idx)
                detrended[i] = flux[i] - trend + mean_trend
            continue

        all_coeffs.append(tuple(coeffs))
        for i in seg_idx:
            trend = _polyval(list(coeffs), time[i])
            mean_trend = coeffs[0]
            detrended[i] = flux[i] - trend + mean_trend

    rms_after = _rms(detrended)

    return DetrenderResult(
        degree=degree,
        n_segments=n_segments,
        coefficients=tuple(all_coeffs),
        rms_before=round(rms_before, 8),
        rms_after=round(rms_after, 8),
        detrended_flux=tuple(round(v, 8) for v in detrended),
        flag="OK",
    )


def apply_detrend(
    time: list[float],
    flux: list[float],
    result: DetrenderResult,
) -> list[float]:
    """Return the already-computed detrended flux from a result."""
    return list(result.detrended_flux)


def format_detrend_result(result: DetrenderResult) -> str:
    """Format detrend result as Markdown."""
    lines = [
        "## Polynomial Detrend",
        "",
        f"- Degree: {result.degree}",
        f"- Segments: {result.n_segments}",
        f"- RMS before: {result.rms_before}",
        f"- RMS after: {result.rms_after}",
        f"- Points detrended: {len(result.detrended_flux)}",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="polynomial_detrend",
        description="Fit and subtract polynomial trend from light curve.",
    )
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--segments", type=int, default=1)
    args = parser.parse_args(argv)

    result = fit_polynomial_trend([], [], degree=args.degree, n_segments=args.segments)
    print(format_detrend_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
