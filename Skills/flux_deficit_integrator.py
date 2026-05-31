"""Integrate the flux deficit over a transit window."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FluxDeficitResult:
    n_points: int
    deficit_ppm_hours: float
    mean_depth_ppm: float
    duration_hours: float
    asymmetry: float
    flag: str


def integrate_flux_deficit(
    time_days: list[float],
    flux_norm: list[float],
    t_ingress: float,
    t_egress: float,
) -> FluxDeficitResult:
    """
    Trapezoidal integration of (1 - flux) over the in-transit window [t_ingress, t_egress].

    Returns the integral in ppm·hours and an ingress/egress asymmetry index.
    """
    n = len(time_days)
    if n < 2 or len(flux_norm) != n:
        return FluxDeficitResult(
            n_points=n, deficit_ppm_hours=float("nan"),
            mean_depth_ppm=float("nan"), duration_hours=float("nan"),
            asymmetry=float("nan"), flag="INSUFFICIENT_DATA",
        )
    if not math.isfinite(t_ingress) or not math.isfinite(t_egress):
        return FluxDeficitResult(
            n_points=n, deficit_ppm_hours=float("nan"),
            mean_depth_ppm=float("nan"), duration_hours=float("nan"),
            asymmetry=float("nan"), flag="INVALID_WINDOW",
        )
    if t_egress <= t_ingress:
        return FluxDeficitResult(
            n_points=n, deficit_ppm_hours=float("nan"),
            mean_depth_ppm=float("nan"), duration_hours=float("nan"),
            asymmetry=float("nan"), flag="INVALID_WINDOW",
        )

    # Select in-transit points
    pairs = [
        (t, f) for t, f in zip(time_days, flux_norm, strict=False)
        if t_ingress <= t <= t_egress
    ]
    pairs.sort(key=lambda x: x[0])

    if len(pairs) < 2:
        return FluxDeficitResult(
            n_points=n, deficit_ppm_hours=float("nan"),
            mean_depth_ppm=float("nan"), duration_hours=float("nan"),
            asymmetry=float("nan"), flag="NO_IN_TRANSIT_POINTS",
        )

    # Trapezoidal integration of deficit = 1 - flux
    integral_days = 0.0
    for i in range(len(pairs) - 1):
        t0, f0 = pairs[i]
        t1, f1 = pairs[i + 1]
        d0 = (1.0 - f0) * 1e6  # ppm
        d1 = (1.0 - f1) * 1e6
        integral_days += 0.5 * (d0 + d1) * (t1 - t0)

    integral_hours = integral_days * 24.0
    duration_hours = (pairs[-1][0] - pairs[0][0]) * 24.0
    mean_depth = integral_hours / duration_hours if duration_hours > 0.0 else 0.0

    # Asymmetry: compare ingress half vs egress half
    t_mid = 0.5 * (pairs[0][0] + pairs[-1][0])
    ing = [(t, f) for t, f in pairs if t <= t_mid]
    egr = [(t, f) for t, f in pairs if t > t_mid]
    if len(ing) >= 2 and len(egr) >= 2:
        def _trapz(pts: list[tuple[float, float]]) -> float:
            s = 0.0
            for i in range(len(pts) - 1):
                t0, f0 = pts[i]
                t1, f1 = pts[i + 1]
                s += 0.5 * ((1 - f0) + (1 - f1)) * (t1 - t0)
            return s * 1e6
        ing_area = _trapz(ing)
        egr_area = _trapz(egr)
        total = ing_area + egr_area
        asymmetry = (ing_area - egr_area) / total if total > 0.0 else 0.0
    else:
        asymmetry = 0.0

    return FluxDeficitResult(
        n_points=len(pairs),
        deficit_ppm_hours=round(integral_hours, 4),
        mean_depth_ppm=round(mean_depth, 4),
        duration_hours=round(duration_hours, 4),
        asymmetry=round(asymmetry, 6),
        flag="OK",
    )


def format_flux_deficit_result(r: FluxDeficitResult) -> str:
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| In-transit points | {r.n_points} |\n"
        f"| Flux deficit (ppm·h) | {r.deficit_ppm_hours:.4f} |\n"
        f"| Mean depth (ppm) | {r.mean_depth_ppm:.4f} |\n"
        f"| Duration (h) | {r.duration_hours:.4f} |\n"
        f"| Asymmetry | {r.asymmetry:.6f} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Integrate flux deficit over transit window.")
    p.add_argument("t_ingress", type=float, help="Ingress time (BJD)")
    p.add_argument("t_egress", type=float, help="Egress time (BJD)")
    p.add_argument("--time", type=float, nargs="+", required=True)
    p.add_argument("--flux", type=float, nargs="+", required=True)
    args = p.parse_args()
    r = integrate_flux_deficit(args.time, args.flux, args.t_ingress, args.t_egress)
    print(format_flux_deficit_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
