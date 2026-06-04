"""Generate a theoretical Keplerian radial velocity curve."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RvCurveResult:
    n_points: int
    k_ms: float
    period_days: float
    eccentricity: float
    rv_min_ms: float
    rv_max_ms: float
    rv_semi_amplitude_ms: float
    phases: tuple[float, ...]
    rv_ms: tuple[float, ...]
    flag: str


def _solve_kepler(mean_anomaly: float, ecc: float, tol: float = 1e-8) -> float:
    """Solve Kepler's equation M = E - e*sin(E) via Newton-Raphson."""
    e = mean_anomaly
    for _ in range(50):
        de = (mean_anomaly - e + ecc * math.sin(e)) / (1.0 - ecc * math.cos(e))
        e += de
        if abs(de) < tol:
            break
    return e


def compute_rv_curve(
    k_ms: float,
    period_days: float,
    eccentricity: float = 0.0,
    omega_deg: float = 90.0,
    t0_bjd: float = 0.0,
    gamma_ms: float = 0.0,
    n_points: int = 100,
) -> RvCurveResult:
    """
    Compute a theoretical Keplerian RV curve.

    Parameters
    ----------
    k_ms:         RV semi-amplitude (m/s).
    period_days:  Orbital period.
    eccentricity: Orbital eccentricity [0, 1).
    omega_deg:    Argument of periastron (degrees).
    t0_bjd:       Reference epoch (BJD); phase 0 at transit mid-point.
    gamma_ms:     Systemic velocity offset (m/s).
    n_points:     Number of phase points to compute.
    """
    if not math.isfinite(k_ms) or k_ms < 0:
        return RvCurveResult(
            n_points=0, k_ms=k_ms, period_days=period_days,
            eccentricity=eccentricity, rv_min_ms=float("nan"),
            rv_max_ms=float("nan"), rv_semi_amplitude_ms=float("nan"),
            phases=(), rv_ms=(), flag="INVALID_K",
        )
    if not math.isfinite(period_days) or period_days <= 0:
        return RvCurveResult(
            n_points=0, k_ms=k_ms, period_days=period_days,
            eccentricity=eccentricity, rv_min_ms=float("nan"),
            rv_max_ms=float("nan"), rv_semi_amplitude_ms=float("nan"),
            phases=(), rv_ms=(), flag="INVALID_PERIOD",
        )
    if not math.isfinite(eccentricity) or eccentricity < 0 or eccentricity >= 1.0:
        return RvCurveResult(
            n_points=0, k_ms=k_ms, period_days=period_days,
            eccentricity=eccentricity, rv_min_ms=float("nan"),
            rv_max_ms=float("nan"), rv_semi_amplitude_ms=float("nan"),
            phases=(), rv_ms=(), flag="INVALID_ECCENTRICITY",
        )

    omega = math.radians(omega_deg)
    phases: list[float] = []
    rvs: list[float] = []

    for i in range(n_points):
        phase = i / n_points
        mean_anomaly = 2.0 * math.pi * phase
        if eccentricity == 0.0:
            true_anomaly = mean_anomaly
        else:
            ecc_anomaly = _solve_kepler(mean_anomaly, eccentricity)
            sin_ta = math.sqrt(1 - eccentricity**2) * math.sin(ecc_anomaly) / (
                1 - eccentricity * math.cos(ecc_anomaly)
            )
            cos_ta = (math.cos(ecc_anomaly) - eccentricity) / (
                1 - eccentricity * math.cos(ecc_anomaly)
            )
            true_anomaly = math.atan2(sin_ta, cos_ta)

        rv = gamma_ms + k_ms * (math.cos(true_anomaly + omega) + eccentricity * math.cos(omega))
        phases.append(round(phase, 6))
        rvs.append(round(rv, 4))

    rv_min = min(rvs)
    rv_max = max(rvs)
    semi_amp = (rv_max - rv_min) / 2.0

    return RvCurveResult(
        n_points=n_points,
        k_ms=k_ms,
        period_days=period_days,
        eccentricity=eccentricity,
        rv_min_ms=round(rv_min, 4),
        rv_max_ms=round(rv_max, 4),
        rv_semi_amplitude_ms=round(semi_amp, 4),
        phases=tuple(phases),
        rv_ms=tuple(rvs),
        flag="OK",
    )


def format_rv_curve_result(r: RvCurveResult) -> str:
    if r.flag != "OK":
        return f"No RV curve (flag: {r.flag}).\n"

    def _f(v: float) -> str:
        return f"{v:.4f}" if math.isfinite(v) else "N/A"

    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| K (m/s) | {_f(r.k_ms)} |\n"
        f"| Period (days) | {_f(r.period_days)} |\n"
        f"| Eccentricity | {_f(r.eccentricity)} |\n"
        f"| RV min (m/s) | {_f(r.rv_min_ms)} |\n"
        f"| RV max (m/s) | {_f(r.rv_max_ms)} |\n"
        f"| Semi-amplitude (m/s) | {_f(r.rv_semi_amplitude_ms)} |\n"
        f"| N points | {r.n_points} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Compute theoretical Keplerian RV curve.")
    p.add_argument("k_ms", type=float)
    p.add_argument("period_days", type=float)
    p.add_argument("--eccentricity", type=float, default=0.0)
    p.add_argument("--omega-deg", type=float, default=90.0)
    p.add_argument("--gamma-ms", type=float, default=0.0)
    p.add_argument("--n-points", type=int, default=100)
    args = p.parse_args()
    r = compute_rv_curve(
        args.k_ms, args.period_days, args.eccentricity,
        args.omega_deg, gamma_ms=args.gamma_ms, n_points=args.n_points,
    )
    print(format_rv_curve_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
