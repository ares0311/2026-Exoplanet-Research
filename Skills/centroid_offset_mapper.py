"""Phase-fold centroid time series and detect in-transit centroid shifts.

An in-transit centroid offset that is inconsistent with the target star
position is strong evidence for a background eclipsing binary (bgEB).
This module phase-folds the centroid (x, y) pixel time series and computes
the mean in-transit vs out-of-transit centroid offset.

Public API
----------
CentroidOffsetResult(delta_x_pix, delta_y_pix, offset_arcsec,
                     offset_sigma, is_significant, flag)
map_centroid_offsets(time, x_cen, y_cen, period_days, epoch_bjd, *,
                     duration_hours, pixel_scale_arcsec,
                     sigma_threshold) -> CentroidOffsetResult
format_centroid_offset_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CentroidOffsetResult:
    delta_x_pix: float | None     # in-transit minus OOT x centroid (pixels)
    delta_y_pix: float | None
    offset_arcsec: float | None   # total offset magnitude in arcsec
    offset_sigma: float | None    # offset / uncertainty
    is_significant: bool
    flag: str  # "OK" | "INSUFFICIENT" | "INVALID"


def _phase_fold(time: list[float], epoch: float, period: float) -> list[float]:
    phases = []
    for t in time:
        ph = ((t - epoch) % period) / period
        phases.append(ph - 1.0 if ph >= 0.5 else ph)
    return phases


def map_centroid_offsets(
    time: list[float],
    x_cen: list[float],
    y_cen: list[float],
    period_days: float,
    epoch_bjd: float,
    *,
    duration_hours: float = 2.0,
    pixel_scale_arcsec: float = 21.0,
    sigma_threshold: float = 3.0,
) -> CentroidOffsetResult:
    """Compute mean in-transit vs OOT centroid offset.

    Args:
        time: Time array (same units as epoch_bjd).
        x_cen: Centroid x-position array (pixels), same length as time.
        y_cen: Centroid y-position array (pixels), same length as time.
        period_days: Orbital period in days.
        epoch_bjd: Reference mid-transit epoch.
        duration_hours: Transit duration in hours.
        pixel_scale_arcsec: Pixel scale in arcsec/pixel (TESS = 21.0).
        sigma_threshold: Significance threshold for ``is_significant``.

    Returns:
        :class:`CentroidOffsetResult`.
    """
    n = len(time)
    if n < 5 or len(x_cen) != n or len(y_cen) != n or period_days <= 0:
        return CentroidOffsetResult(None, None, None, None, False, "INVALID")

    half_width = (duration_hours / 24.0) / period_days / 2.0
    phases = _phase_fold(time, epoch_bjd, period_days)

    in_x: list[float] = []
    in_y: list[float] = []
    oot_x: list[float] = []
    oot_y: list[float] = []
    oot_half = min(3 * half_width, 0.4)

    for i, ph in enumerate(phases):
        ap = abs(ph)
        if ap <= half_width:
            in_x.append(x_cen[i])
            in_y.append(y_cen[i])
        elif half_width < ap <= oot_half:
            oot_x.append(x_cen[i])
            oot_y.append(y_cen[i])

    if len(in_x) < 2 or len(oot_x) < 2:
        return CentroidOffsetResult(None, None, None, None, False, "INSUFFICIENT")

    def _mean(v: list[float]) -> float:
        return sum(v) / len(v)

    def _std(v: list[float], m: float) -> float:
        if len(v) < 2:
            return 0.0
        return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))

    in_x_mean = _mean(in_x)
    in_y_mean = _mean(in_y)
    oot_x_mean = _mean(oot_x)
    oot_y_mean = _mean(oot_y)

    dx = in_x_mean - oot_x_mean
    dy = in_y_mean - oot_y_mean
    offset_pix = math.sqrt(dx ** 2 + dy ** 2)
    offset_arcsec = offset_pix * pixel_scale_arcsec

    # Uncertainty from OOT scatter / sqrt(n_in)
    sx_oot = _std(oot_x, oot_x_mean)
    sy_oot = _std(oot_y, oot_y_mean)
    sigma_offset_pix = math.sqrt(sx_oot ** 2 + sy_oot ** 2) / math.sqrt(len(in_x))

    offset_sig: float | None = None
    if sigma_offset_pix > 1e-12:
        offset_sig = round(offset_pix / sigma_offset_pix, 3)

    is_sig = offset_sig is not None and offset_sig >= sigma_threshold

    return CentroidOffsetResult(
        delta_x_pix=round(dx, 5),
        delta_y_pix=round(dy, 5),
        offset_arcsec=round(offset_arcsec, 3),
        offset_sigma=offset_sig,
        is_significant=is_sig,
        flag="OK",
    )


def format_centroid_offset_result(result: CentroidOffsetResult) -> str:
    """Format centroid offset result as Markdown."""
    lines = [
        "## Centroid Offset Map",
        "",
        f"- Δx (pixels): {result.delta_x_pix}",
        f"- Δy (pixels): {result.delta_y_pix}",
        f"- Offset (arcsec): {result.offset_arcsec}",
        f"- Offset significance (σ): {result.offset_sigma}",
        f"- Significant: {'Yes' if result.is_significant else 'No'}",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="centroid_offset_mapper",
        description="Detect in-transit centroid shifts.",
    )
    parser.add_argument("period_days", type=float)
    parser.add_argument("epoch_bjd", type=float)
    parser.add_argument("--duration-hours", type=float, default=2.0)
    parser.add_argument("--pixel-scale", type=float, default=21.0)
    args = parser.parse_args(argv)

    result = map_centroid_offsets(
        [], [], [], args.period_days, args.epoch_bjd,
        duration_hours=args.duration_hours,
        pixel_scale_arcsec=args.pixel_scale,
    )
    print(format_centroid_offset_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
