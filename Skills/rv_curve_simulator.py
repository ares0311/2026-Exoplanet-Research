"""Simulate a Keplerian RV curve.

Public API:
    RvCurveResult  -- frozen dataclass
    simulate_rv_curve(k_ms, period_days, *, e, omega_deg, n_points) -> RvCurveResult
    format_rv_curve(result) -> str
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RvCurveResult:
    phases: list[float]
    rv_ms: list[float]
    k_amplitude_ms: float
    period_days: float
    eccentricity: float
    flag: str


def _true_anomaly(mean_anomaly: float, e: float, tol: float = 1e-8) -> float:
    ea = mean_anomaly
    for _ in range(50):
        delta = (ea - e * math.sin(ea) - mean_anomaly) / (1.0 - e * math.cos(ea))
        ea -= delta
        if abs(delta) < tol:
            break
    return 2.0 * math.atan2(
        math.sqrt(1 + e) * math.sin(ea / 2.0),
        math.sqrt(1 - e) * math.cos(ea / 2.0),
    )


def simulate_rv_curve(
    k_ms: float,
    period_days: float,
    *,
    eccentricity: float = 0.0,
    omega_deg: float = 90.0,
    n_points: int = 100,
) -> RvCurveResult:
    if k_ms < 0:
        return RvCurveResult(
            phases=[], rv_ms=[], k_amplitude_ms=k_ms,
            period_days=period_days, eccentricity=eccentricity,
            flag="INVALID_K_AMPLITUDE",
        )
    if period_days <= 0:
        return RvCurveResult(
            phases=[], rv_ms=[], k_amplitude_ms=k_ms,
            period_days=period_days, eccentricity=eccentricity,
            flag="INVALID_PERIOD",
        )
    if not 0.0 <= eccentricity < 1.0:
        return RvCurveResult(
            phases=[], rv_ms=[], k_amplitude_ms=k_ms,
            period_days=period_days, eccentricity=eccentricity,
            flag="INVALID_ECCENTRICITY",
        )
    omega = omega_deg * math.pi / 180.0
    phases = [i / n_points for i in range(n_points)]
    rv_ms: list[float] = []
    for ph in phases:
        m = 2.0 * math.pi * ph
        nu = _true_anomaly(m, eccentricity)
        rv = k_ms * (math.cos(nu + omega) + eccentricity * math.cos(omega))
        rv_ms.append(rv)
    return RvCurveResult(
        phases=phases,
        rv_ms=rv_ms,
        k_amplitude_ms=k_ms,
        period_days=period_days,
        eccentricity=eccentricity,
        flag="OK",
    )


def format_rv_curve(result: RvCurveResult) -> str:
    lines = [
        "## RV Curve Simulation",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| K Amplitude (m/s) | {result.k_amplitude_ms:.4f} |",
        f"| Period (days) | {result.period_days:.4f} |",
        f"| Eccentricity | {result.eccentricity:.4f} |",
        f"| N Points | {len(result.phases)} |",
        f"| Flag | {result.flag} |",
    ]
    if result.rv_ms:
        lines.append(f"| RV Range (m/s) | "
                     f"{min(result.rv_ms):.2f} to {max(result.rv_ms):.2f} |")
    return "\n".join(lines)


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Simulate a Keplerian RV curve.")
    parser.add_argument("k_ms", type=float)
    parser.add_argument("period_days", type=float)
    parser.add_argument("--eccentricity", type=float, default=0.0)
    parser.add_argument("--omega-deg", type=float, default=90.0)
    parser.add_argument("--n-points", type=int, default=100)
    args = parser.parse_args()
    result = simulate_rv_curve(
        args.k_ms, args.period_days,
        eccentricity=args.eccentricity,
        omega_deg=args.omega_deg,
        n_points=args.n_points,
    )
    print(format_rv_curve(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
