"""Compute 2D period×depth detection sensitivity map from light curve noise model."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SensitivityMapResult:
    n_periods: int
    n_depths: int
    period_grid_days: tuple[float, ...]
    depth_grid_ppm: tuple[float, ...]
    snr_grid: tuple[tuple[float, ...], ...]  # [period_idx][depth_idx]
    detectable_grid: tuple[tuple[bool, ...], ...]
    n_detectable_cells: int
    flag: str


def compute_sensitivity_map(
    baseline_days: float,
    rms_ppm: float,
    n_transits_per_period: float | None = None,
    cadence_minutes: float = 2.0,
    snr_threshold: float = 7.5,
    period_grid_days: list[float] | None = None,
    depth_grid_ppm: list[float] | None = None,
) -> SensitivityMapResult:
    """
    Compute a 2D period × depth detection sensitivity map.

    SNR = depth / sigma_transit, where
    sigma_transit = rms / sqrt(N_in_transit), and
    N_in_transit = n_transits * transit_duration_points.

    Transit duration approximation: T14 ≈ 13h * (P/yr)^(1/3)  (circular, b=0, 1 Msun/Rsun)
    n_transits = floor(baseline / P) if not provided.

    Returns SNR grid and detectability (SNR > threshold) for each cell.
    """
    if not math.isfinite(baseline_days) or baseline_days <= 0.0:
        return SensitivityMapResult(
            n_periods=0, n_depths=0, period_grid_days=(), depth_grid_ppm=(),
            snr_grid=(), detectable_grid=(), n_detectable_cells=0,
            flag="INVALID_BASELINE",
        )
    if not math.isfinite(rms_ppm) or rms_ppm <= 0.0:
        return SensitivityMapResult(
            n_periods=0, n_depths=0, period_grid_days=(), depth_grid_ppm=(),
            snr_grid=(), detectable_grid=(), n_detectable_cells=0,
            flag="INVALID_RMS",
        )

    # Default grids
    if period_grid_days is None:
        period_grid_days = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    if depth_grid_ppm is None:
        depth_grid_ppm = [100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0]

    snr_rows: list[tuple[float, ...]] = []
    det_rows: list[tuple[bool, ...]] = []

    for period in period_grid_days:
        snr_cols: list[float] = []
        det_cols: list[bool] = []
        if period <= 0 or not math.isfinite(period):
            for _ in depth_grid_ppm:
                snr_cols.append(float("nan"))
                det_cols.append(False)
        else:
            # Transit duration (hours): T14 ≈ 13 * (P/365.25)^(1/3)
            t14_hours = 13.0 * (period / 365.25) ** (1.0 / 3.0)
            n_pts_per_transit = max(1, int(t14_hours * 60.0 / cadence_minutes))

            n_tr = n_transits_per_period if n_transits_per_period is not None else max(
                1, int(baseline_days / period)
            )
            total_pts = n_tr * n_pts_per_transit
            sigma_transit = rms_ppm / math.sqrt(total_pts) if total_pts > 0 else float("inf")

            for depth in depth_grid_ppm:
                snr = depth / sigma_transit if sigma_transit > 0 else 0.0
                snr_cols.append(round(snr, 2))
                det_cols.append(snr >= snr_threshold)

        snr_rows.append(tuple(snr_cols))
        det_rows.append(tuple(det_cols))

    n_det = sum(1 for row in det_rows for cell in row if cell)

    return SensitivityMapResult(
        n_periods=len(period_grid_days),
        n_depths=len(depth_grid_ppm),
        period_grid_days=tuple(period_grid_days),
        depth_grid_ppm=tuple(depth_grid_ppm),
        snr_grid=tuple(snr_rows),
        detectable_grid=tuple(det_rows),
        n_detectable_cells=n_det,
        flag="OK",
    )


def format_sensitivity_map(r: SensitivityMapResult) -> str:
    if r.flag != "OK":
        return f"No map (flag: {r.flag}).\n"
    header = "| Period (d) \\ Depth (ppm) | " + " | ".join(
        f"{int(d)}" for d in r.depth_grid_ppm
    ) + " |"
    sep = "|" + "|".join("---" for _ in range(r.n_depths + 1)) + "|"
    lines = [
        f"**Detection Sensitivity Map** — "
        f"{r.n_detectable_cells}/{r.n_periods * r.n_depths} cells detectable\n",
        header, sep,
    ]
    for i, period in enumerate(r.period_grid_days):
        row_vals = " | ".join(
            ("Y" if r.detectable_grid[i][j] else "n") + f"({r.snr_grid[i][j]:.1f})"
            for j in range(r.n_depths)
        )
        lines.append(f"| {period:.1f} | {row_vals} |")
    return "\n".join(lines)


def _cli() -> int:
    p = argparse.ArgumentParser(description="Compute period-depth detection sensitivity map.")
    p.add_argument("baseline_days", type=float)
    p.add_argument("rms_ppm", type=float)
    p.add_argument("--snr-threshold", type=float, default=7.5)
    p.add_argument("--cadence-minutes", type=float, default=2.0)
    args = p.parse_args()
    r = compute_sensitivity_map(
        args.baseline_days, args.rms_ppm,
        snr_threshold=args.snr_threshold, cadence_minutes=args.cadence_minutes,
    )
    print(format_sensitivity_map(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
