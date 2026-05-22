"""Detect long-term flux trends in out-of-transit (OOT) data via linear regression.

A significant positive or negative slope in the OOT flux suggests
instrumental systematics, stellar activity, or contamination from a nearby
variable star.  This module fits a straight line y = a + b*t and tests
whether the slope is significant relative to the scatter.

Public API
----------
FluxTrendResult(slope_per_day, slope_sigma, slope_snr, is_significant,
                intercept, rms_residual, flag)
detect_flux_trend(time, flux, *, flux_err, significance_threshold) -> FluxTrendResult
format_trend_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FluxTrendResult:
    slope_per_day: float          # linear slope b in [flux/day]
    slope_sigma: float | None     # standard error on b
    slope_snr: float | None       # |b| / sigma_b
    is_significant: bool
    intercept: float
    rms_residual: float
    flag: str  # "OK" | "FLAT" | "INSUFFICIENT" | "INVALID"


def _weighted_linreg(
    x: list[float],
    y: list[float],
    w: list[float],
) -> tuple[float, float, float, float]:
    """Weighted least-squares y = a + b*x.

    Returns (a, b, sigma_a, sigma_b).
    """
    sw = sum(w)
    if sw <= 0:
        return 0.0, 0.0, 0.0, 0.0
    swx = sum(wi * xi for wi, xi in zip(w, x, strict=False))
    swy = sum(wi * yi for wi, yi in zip(w, y, strict=False))
    swxx = sum(wi * xi * xi for wi, xi in zip(w, x, strict=False))
    swxy = sum(wi * xi * yi for wi, xi, yi in zip(w, x, y, strict=False))

    denom = sw * swxx - swx ** 2
    if abs(denom) < 1e-30:
        return swy / sw, 0.0, 0.0, 0.0

    b = (sw * swxy - swx * swy) / denom
    a = (swy - b * swx) / sw

    # Estimate parameter uncertainties from weighted residuals
    n = len(x)
    if n > 2:
        resid2 = sum(wi * (yi - a - b * xi) ** 2 for wi, xi, yi in zip(w, x, y, strict=False))
        s2 = resid2 / (n - 2)
        var_b = s2 * sw / denom
        var_a = s2 * swxx / denom
        sigma_b = math.sqrt(max(var_b, 0.0))
        sigma_a = math.sqrt(max(var_a, 0.0))
    else:
        sigma_b = 0.0
        sigma_a = 0.0

    return a, b, sigma_a, sigma_b


def detect_flux_trend(
    time: list[float],
    flux: list[float],
    *,
    flux_err: list[float] | None = None,
    significance_threshold: float = 3.0,
) -> FluxTrendResult:
    """Fit a linear trend to a flux time series and assess significance.

    Args:
        time: Time array (days).
        flux: Normalised flux array.
        flux_err: Per-point flux uncertainties (uniform 1.0 if None).
        significance_threshold: SNR threshold for ``is_significant``.

    Returns:
        :class:`FluxTrendResult`.
    """
    n = len(flux)
    if n < 3 or len(time) != n:
        return FluxTrendResult(0.0, None, None, False, 0.0, 0.0, "INVALID")

    errs = flux_err if (flux_err is not None and len(flux_err) == n) else [1.0] * n
    weights = [1.0 / max(e ** 2, 1e-30) for e in errs]

    # Centre time to improve numerical stability
    t0 = sum(time) / n
    t_centred = [t - t0 for t in time]

    a, b, _sigma_a, sigma_b = _weighted_linreg(t_centred, flux, weights)

    # RMS of residuals
    residuals = [flux[i] - (a + b * t_centred[i]) for i in range(n)]
    rms = math.sqrt(sum(r ** 2 for r in residuals) / n)

    snr: float | None = None
    if sigma_b > 1e-30:
        snr = round(abs(b) / sigma_b, 3)

    is_sig = (snr is not None) and (snr >= significance_threshold)

    flag = "OK" if is_sig else "FLAT"

    return FluxTrendResult(
        slope_per_day=round(b, 8),
        slope_sigma=round(sigma_b, 8) if sigma_b > 0 else None,
        slope_snr=snr,
        is_significant=is_sig,
        intercept=round(a, 8),
        rms_residual=round(rms, 8),
        flag=flag,
    )


def format_trend_result(result: FluxTrendResult) -> str:
    """Format flux trend result as Markdown."""
    lines = [
        "## Flux Trend Detection",
        "",
        f"- Slope: {result.slope_per_day:.3e} flux/day",
        f"- Slope σ: {result.slope_sigma}",
        f"- Slope SNR: {result.slope_snr}",
        f"- Significant: {'Yes' if result.is_significant else 'No'}",
        f"- Intercept: {result.intercept:.6f}",
        f"- RMS residual: {result.rms_residual:.6f}",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="flux_trend_detector",
        description="Detect long-term linear trends in a flux series.",
    )
    parser.add_argument("--significance-threshold", type=float, default=3.0)
    args = parser.parse_args(argv)

    result = detect_flux_trend(
        [], [],
        significance_threshold=args.significance_threshold,
    )
    print(format_trend_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
