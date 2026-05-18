"""Check pixel-level centroid motion during transit to identify background EBs.

Computes the in-transit vs. out-of-transit centroid offset in row/column
pixel coordinates and flags significant shifts that indicate the transit
source is not centred on the target.

Public API
----------
CentroidCheckResult(n_in_transit, n_oot, delta_row_px, delta_col_px,
                    offset_arcsec, significance_sigma, flag)
check_pixel_centroid(time, flux, row_centroid, col_centroid,
                     period_days, epoch_bjd, *, duration_days,
                     pixel_scale_arcsec, sigma_threshold) -> CentroidCheckResult
format_centroid_check_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CentroidCheckResult:
    n_in_transit: int
    n_oot: int
    delta_row_px: float | None
    delta_col_px: float | None
    offset_arcsec: float | None
    significance_sigma: float | None
    flag: str  # "OK", "CENTROID_SHIFT", "INSUFFICIENT"


def _mean_std(values: list[float]) -> tuple[float, float]:
    """Return (mean, std) of a list."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    m = sum(values) / n
    if n == 1:
        return m, 0.0
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    return m, math.sqrt(var)


def check_pixel_centroid(
    time: list[float],
    flux: list[float],
    row_centroid: list[float],
    col_centroid: list[float],
    period_days: float,
    epoch_bjd: float,
    *,
    duration_days: float = 0.1,
    pixel_scale_arcsec: float = 21.0,
    sigma_threshold: float = 3.0,
) -> CentroidCheckResult:
    """Check for pixel-level centroid shift during transit.

    Args:
        time: Time array (BJD).
        flux: Normalised flux array (used only for sanity; not strictly needed).
        row_centroid: Per-cadence row centroid (pixels).
        col_centroid: Per-cadence column centroid (pixels).
        period_days: Orbital period in days.
        epoch_bjd: Reference epoch in BJD.
        duration_days: Transit duration in days.
        pixel_scale_arcsec: Pixel scale in arcsec/pixel (TESS ≈ 21 arcsec/px).
        sigma_threshold: Detection threshold in sigma.

    Returns:
        :class:`CentroidCheckResult`.
    """
    if (not time or not row_centroid or not col_centroid
            or len(time) != len(row_centroid) or period_days <= 0):
        return CentroidCheckResult(0, 0, None, None, None, None, "INSUFFICIENT")

    half = duration_days / 2.0
    in_rows: list[float] = []
    in_cols: list[float] = []
    oot_rows: list[float] = []
    oot_cols: list[float] = []

    for t, r, c in zip(time, row_centroid, col_centroid, strict=False):
        ph_abs = (t - epoch_bjd) % period_days
        if ph_abs > period_days / 2:
            ph_abs -= period_days
        if abs(ph_abs) <= half:
            in_rows.append(r)
            in_cols.append(c)
        elif abs(ph_abs) > half * 3:
            oot_rows.append(r)
            oot_cols.append(c)

    if not in_rows or not oot_rows:
        return CentroidCheckResult(
            len(in_rows), len(oot_rows), None, None, None, None, "INSUFFICIENT",
        )

    mean_in_r, std_in_r = _mean_std(in_rows)
    mean_in_c, std_in_c = _mean_std(in_cols)
    mean_oot_r, std_oot_r = _mean_std(oot_rows)
    mean_oot_c, std_oot_c = _mean_std(oot_cols)

    delta_row = mean_in_r - mean_oot_r
    delta_col = mean_in_c - mean_oot_c
    offset_px = math.sqrt(delta_row ** 2 + delta_col ** 2)
    offset_arcsec = offset_px * pixel_scale_arcsec

    # Significance: use OOT scatter as noise floor for in-transit uncertainty
    n_in = len(in_rows)
    n_oot = len(oot_rows)
    # Per-point uncertainty from OOT scatter (or minimum floor if OOT is flat)
    sigma_r = max(std_oot_r, std_in_r, 1e-6)
    sigma_c = max(std_oot_c, std_in_c, 1e-6)
    err_r = sigma_r / math.sqrt(n_in)
    err_c = sigma_c / math.sqrt(n_in)
    err_total = math.sqrt(err_r ** 2 + err_c ** 2)

    significance = offset_px / err_total if err_total > 0 else 0.0
    flag = "CENTROID_SHIFT" if significance >= sigma_threshold else "OK"

    return CentroidCheckResult(
        n_in_transit=n_in,
        n_oot=n_oot,
        delta_row_px=round(delta_row, 4),
        delta_col_px=round(delta_col, 4),
        offset_arcsec=round(offset_arcsec, 4),
        significance_sigma=round(significance, 3),
        flag=flag,
    )


def format_centroid_check_result(result: CentroidCheckResult) -> str:
    """Format pixel centroid check result as Markdown."""
    lines = [
        "## Pixel-Level Centroid Check",
        "",
        f"- In-transit cadences: {result.n_in_transit}",
        f"- OOT cadences: {result.n_oot}",
    ]
    if result.flag == "INSUFFICIENT":
        lines.append("- **Flag: INSUFFICIENT** — not enough data")
    else:
        lines += [
            f"- Δrow: {result.delta_row_px:.4f} px",
            f"- Δcol: {result.delta_col_px:.4f} px",
            f"- Offset: {result.offset_arcsec:.4f} arcsec",
            f"- Significance: {result.significance_sigma:.3f}σ",
            f"- **Flag: {result.flag}**",
        ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="pixel_level_centroid_checker",
        description="Check pixel-level centroid motion during transit.",
    )
    parser.add_argument("period_days", type=float)
    parser.add_argument("epoch_bjd", type=float)
    parser.add_argument("--duration-days", type=float, default=0.1)
    parser.add_argument("--pixel-scale-arcsec", type=float, default=21.0)
    parser.add_argument("--sigma-threshold", type=float, default=3.0)
    args = parser.parse_args(argv)

    result = check_pixel_centroid(
        [], [], [], [], args.period_days, args.epoch_bjd,
        duration_days=args.duration_days,
        pixel_scale_arcsec=args.pixel_scale_arcsec,
        sigma_threshold=args.sigma_threshold,
    )
    print(format_centroid_check_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
