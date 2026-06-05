"""Generate a full in-transit Rossiter-McLaughlin effect time series."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RmTimeSeriesResult:
    phases: tuple[float, ...]
    rv_anomaly_ms: tuple[float, ...]
    max_anomaly_ms: float
    min_anomaly_ms: float
    lambda_deg: float
    flag: str


def compute_rm_time_series(
    v_sin_i_ms: float,
    depth: float,
    impact_parameter: float = 0.0,
    lambda_deg: float = 0.0,
    n_points: int = 50,
    limb_darkening_u1: float = 0.4,
    limb_darkening_u2: float = 0.2,
) -> RmTimeSeriesResult:
    """Compute RM anomaly as a function of orbital phase during transit.

    Uses the Ohta, Taruya & Suto (2005) analytic approximation:
      ΔRV(t) ≈ -v·sin(i) × δ(t) × x_p(t)/R★ × f_LD(x_p, y_p)

    where x_p, y_p are the planet centre coordinates on the stellar disk.

    Args:
        v_sin_i_ms: projected stellar rotation velocity (m/s)
        depth: transit depth (Rp/Rs)²; must be in (0, 1)
        impact_parameter: transit impact parameter b
        lambda_deg: projected spin-orbit angle λ (degrees)
        n_points: number of phase points during transit
        limb_darkening_u1: linear limb-darkening coefficient
        limb_darkening_u2: quadratic limb-darkening coefficient
    """
    if v_sin_i_ms <= 0.0:
        return RmTimeSeriesResult((), (), float("nan"), float("nan"),
                                   lambda_deg, "INVALID_VSINI")
    if not (0.0 < depth < 1.0):
        return RmTimeSeriesResult((), (), float("nan"), float("nan"),
                                   lambda_deg, "INVALID_DEPTH")
    if not (0.0 <= impact_parameter < 1.0):
        return RmTimeSeriesResult((), (), float("nan"), float("nan"),
                                   lambda_deg, "INVALID_IMPACT")

    rp_rs = math.sqrt(depth)
    lambda_rad = math.radians(lambda_deg)
    cos_lam = math.cos(lambda_rad)
    sin_lam = math.sin(lambda_rad)

    transit_half_dur = math.sqrt((1.0 + rp_rs) ** 2 - impact_parameter**2)
    phases = [
        -transit_half_dur + i * 2.0 * transit_half_dur / (n_points - 1)
        for i in range(n_points)
    ]

    rv_anomalies = []
    for x_sky in phases:
        y_sky = impact_parameter
        # Planet position on sky rotated by λ
        x_p = x_sky * cos_lam - y_sky * sin_lam
        y_p = x_sky * sin_lam + y_sky * cos_lam
        r_p = math.sqrt(x_p**2 + y_p**2)

        if r_p > 1.0 + rp_rs:
            rv_anomalies.append(0.0)
            continue

        # Limb darkening at planet centre position
        mu = math.sqrt(max(1.0 - r_p**2, 0.0))
        I_mu = 1.0 - limb_darkening_u1 * (1.0 - mu) - limb_darkening_u2 * (1.0 - mu)**2
        I_mu = max(I_mu, 0.0)

        # OTS approximation: ΔRV ≈ -v*sin(i) * depth * x_p * I(μ)
        delta_rv = -v_sin_i_ms * depth * x_p * I_mu
        rv_anomalies.append(delta_rv)

    rv_tuple = tuple(rv_anomalies)
    return RmTimeSeriesResult(
        phases=tuple(phases),
        rv_anomaly_ms=rv_tuple,
        max_anomaly_ms=max(rv_tuple) if rv_tuple else float("nan"),
        min_anomaly_ms=min(rv_tuple) if rv_tuple else float("nan"),
        lambda_deg=lambda_deg,
        flag="OK",
    )


def format_rm_time_series_result(r: RmTimeSeriesResult) -> str:
    if r.flag != "OK":
        return f"RmTimeSeries | flag={r.flag}"
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| λ (spin-orbit angle) | {r.lambda_deg:.1f}° |\n"
        f"| Max RM anomaly | {r.max_anomaly_ms:.2f} m/s |\n"
        f"| Min RM anomaly | {r.min_anomaly_ms:.2f} m/s |\n"
        f"| N phase points | {len(r.phases)} |\n"
        f"| Flag | {r.flag} |"
    )


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="RM effect time series model")
    p.add_argument("v_sin_i_ms", type=float)
    p.add_argument("depth", type=float)
    p.add_argument("--b", type=float, default=0.0)
    p.add_argument("--lambda-deg", type=float, default=0.0)
    p.add_argument("--n", type=int, default=50)
    args = p.parse_args()
    r = compute_rm_time_series(args.v_sin_i_ms, args.depth,
                                impact_parameter=args.b, lambda_deg=args.lambda_deg,
                                n_points=args.n)
    print(format_rm_time_series_result(r))


if __name__ == "__main__":
    _cli()
