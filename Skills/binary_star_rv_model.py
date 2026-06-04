"""Model radial velocity curves for single-lined (SB1) and double-lined (SB2) binary stars."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BinaryRvResult:
    k1_ms: float
    k2_ms: float | None          # None for SB1
    period_days: float
    eccentricity: float
    mass_ratio: float | None     # M1/M2; None for SB1
    phases: tuple[float, ...]
    rv_primary_ms: tuple[float, ...]
    rv_secondary_ms: tuple[float, ...] | None
    rv_min_primary_ms: float
    rv_max_primary_ms: float
    flag: str


def _solve_kepler(mean_anomaly: float, eccentricity: float, tol: float = 1e-8) -> float:
    """Newton–Raphson solution of Kepler's equation M = E − e sin E."""
    e_anom = mean_anomaly
    for _ in range(50):
        f = e_anom - eccentricity * math.sin(e_anom) - mean_anomaly
        fp = 1.0 - eccentricity * math.cos(e_anom)
        delta = f / fp
        e_anom -= delta
        if abs(delta) < tol:
            break
    return e_anom


def compute_binary_rv_model(
    k1_ms: float,
    period_days: float,
    eccentricity: float = 0.0,
    omega_deg: float = 0.0,
    t0_bjd: float = 0.0,
    gamma_ms: float = 0.0,
    mass_ratio: float | None = None,
    n_points: int = 100,
) -> BinaryRvResult:
    """
    Compute SB1 or SB2 binary star RV curves.

    For SB1: only primary RV curve is returned.
    For SB2: mass_ratio = M1/M2 must be supplied; K2 = K1 * M1/M2.

    rv1 = γ + K1 * (cos(ν + ω) + e cos ω)
    rv2 = γ − K2 * (cos(ν + ω) + e cos ω)   (SB2 only)

    Parameters
    ----------
    k1_ms:       Primary RV semi-amplitude in m/s.
    period_days: Orbital period in days.
    eccentricity:Orbital eccentricity.
    omega_deg:   Argument of periastron (degrees).
    t0_bjd:      Time of periastron passage (BJD).
    gamma_ms:    Systemic velocity in m/s.
    mass_ratio:  M1/M2; supply for SB2 model (else SB1).
    n_points:    Number of phase points in output.
    """
    if not math.isfinite(period_days) or period_days <= 0:
        return BinaryRvResult(k1_ms, None, period_days, eccentricity, mass_ratio,
                              (), (), None, float("nan"), float("nan"), "INVALID_PERIOD")
    if not math.isfinite(eccentricity) or eccentricity < 0 or eccentricity >= 1:
        return BinaryRvResult(k1_ms, None, period_days, eccentricity, mass_ratio,
                              (), (), None, float("nan"), float("nan"), "INVALID_ECCENTRICITY")
    if not math.isfinite(k1_ms) or k1_ms <= 0:
        return BinaryRvResult(k1_ms, None, period_days, eccentricity, mass_ratio,
                              (), (), None, float("nan"), float("nan"), "INVALID_K1")

    k2_ms = k1_ms * mass_ratio if mass_ratio is not None else None
    omega_rad = math.radians(omega_deg)

    phases: list[float] = []
    rv1_list: list[float] = []
    rv2_list: list[float] = []

    for i in range(n_points):
        phase = i / n_points
        mean_anom = 2.0 * math.pi * phase
        ecc_anom = _solve_kepler(mean_anom, eccentricity)

        # True anomaly
        nu = 2.0 * math.atan2(
            math.sqrt(1.0 + eccentricity) * math.sin(ecc_anom / 2.0),
            math.sqrt(1.0 - eccentricity) * math.cos(ecc_anom / 2.0),
        )
        cos_term = math.cos(nu + omega_rad) + eccentricity * math.cos(omega_rad)
        rv1 = gamma_ms + k1_ms * cos_term
        phases.append(round(phase, 6))
        rv1_list.append(round(rv1, 4))
        if k2_ms is not None:
            rv2_list.append(round(gamma_ms - k2_ms * cos_term, 4))

    return BinaryRvResult(
        k1_ms=k1_ms,
        k2_ms=k2_ms,
        period_days=period_days,
        eccentricity=eccentricity,
        mass_ratio=mass_ratio,
        phases=tuple(phases),
        rv_primary_ms=tuple(rv1_list),
        rv_secondary_ms=tuple(rv2_list) if rv2_list else None,
        rv_min_primary_ms=round(min(rv1_list), 4),
        rv_max_primary_ms=round(max(rv1_list), 4),
        flag="OK",
    )


def format_binary_rv_result(r: BinaryRvResult) -> str:
    def _f(v: float | None, fmt: str = ".4f") -> str:
        if v is None:
            return "N/A"
        return format(v, fmt) if math.isfinite(v) else "N/A"

    return (
        f"| Parameter | Value |\n|---|---|\n"
        f"| K1 (m/s) | {_f(r.k1_ms)} |\n"
        f"| K2 (m/s) | {_f(r.k2_ms)} |\n"
        f"| Period (days) | {_f(r.period_days)} |\n"
        f"| Eccentricity | {_f(r.eccentricity)} |\n"
        f"| Mass ratio M1/M2 | {_f(r.mass_ratio)} |\n"
        f"| N points | {len(r.phases)} |\n"
        f"| RV1 min (m/s) | {_f(r.rv_min_primary_ms)} |\n"
        f"| RV1 max (m/s) | {_f(r.rv_max_primary_ms)} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Compute binary star RV model (SB1 or SB2).")
    p.add_argument("k1_ms", type=float, help="Primary RV semi-amplitude (m/s)")
    p.add_argument("period_days", type=float)
    p.add_argument("--eccentricity", type=float, default=0.0)
    p.add_argument("--omega-deg", type=float, default=0.0)
    p.add_argument("--gamma-ms", type=float, default=0.0)
    p.add_argument("--mass-ratio", type=float, default=None)
    p.add_argument("--n-points", type=int, default=100)
    args = p.parse_args()
    r = compute_binary_rv_model(
        args.k1_ms, args.period_days,
        eccentricity=args.eccentricity, omega_deg=args.omega_deg,
        gamma_ms=args.gamma_ms, mass_ratio=args.mass_ratio, n_points=args.n_points,
    )
    print(format_binary_rv_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
