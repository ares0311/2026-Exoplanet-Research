"""Check mutual inclination bound for multi-planet system from transit durations."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

# For a flat system (b=0): T14 ~ (2*R_star/v) = (P*R_star)/(pi*a)
# Ratio T14_i / T14_j = sqrt((a_j/a_i) * P_i/P_j) under Kepler 3rd law (M fixed)
# => T_ratio = (P_i/P_j)^(1/3)


@dataclass(frozen=True)
class CoplanarityResult:
    n_planets: int
    delta_i_max_deg: float   # upper bound on mutual inclination
    delta_i_min_deg: float   # lower bound on mutual inclination
    coplanar: bool
    flag: str


def check_coplanarity(
    periods_days: list[float],
    durations_hours: list[float],
    stellar_radius_rsun: float = 1.0,
    duration_errors_hours: list[float] | None = None,
    coplanar_threshold_deg: float = 5.0,
) -> CoplanarityResult:
    """
    Estimate mutual inclination bound from transit duration ratios.

    For two planets i, j assuming circular orbits and b=0:
    Expected T14_j / T14_i = (P_j / P_i)^(1/3)

    Deviation from expected ratio implies non-zero mutual inclination:
    sin(Δi) ≈ |1 - (T_obs/T_exp)| * (a/R_star)^{-1}

    Requires at least 2 planets with known periods and durations.
    stellar_radius_rsun: used to estimate a/R_star scale.
    """
    n = len(periods_days)
    if n < 2:
        return CoplanarityResult(
            n_planets=n, delta_i_max_deg=float("nan"), delta_i_min_deg=float("nan"),
            coplanar=True, flag="SINGLE_PLANET",
        )
    if len(durations_hours) != n:
        return CoplanarityResult(
            n_planets=n, delta_i_max_deg=float("nan"), delta_i_min_deg=float("nan"),
            coplanar=False, flag="LENGTH_MISMATCH",
        )

    valid = [(p, d) for p, d in zip(periods_days, durations_hours, strict=False)
             if math.isfinite(p) and p > 0 and math.isfinite(d) and d > 0]
    if len(valid) < 2:
        return CoplanarityResult(
            n_planets=n, delta_i_max_deg=float("nan"), delta_i_min_deg=float("nan"),
            coplanar=True, flag="INSUFFICIENT_VALID_PLANETS",
        )

    delta_i_list: list[float] = []
    r_star_au = stellar_radius_rsun * 0.00465047  # Rsun to AU

    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            pi, ti = valid[i]
            pj, tj = valid[j]

            # Expected duration ratio from Kepler 3rd law
            t_ratio_exp = (pj / pi) ** (1.0 / 3.0)
            t_ratio_obs = tj / ti

            deviation = abs(t_ratio_obs - t_ratio_exp) / t_ratio_exp

            # a_i / R_star ≈ (P_i / P_ref)^(2/3) * (a_ref / R_star)
            # For rough estimate: a_i (AU) ~ (P_i / 365.25)^(2/3) for 1 Msun
            a_i_au = (pi / 365.25) ** (2.0 / 3.0)
            a_over_r = a_i_au / r_star_au

            # sin(delta_i) ≈ deviation * (R_star / a)
            sin_di = deviation / a_over_r if a_over_r > 0 else deviation
            di_rad = math.asin(min(sin_di, 1.0))
            delta_i_list.append(math.degrees(di_rad))

    if not delta_i_list:
        return CoplanarityResult(
            n_planets=n, delta_i_max_deg=float("nan"), delta_i_min_deg=float("nan"),
            coplanar=True, flag="NO_PAIRS",
        )

    di_max = max(delta_i_list)
    di_min = min(delta_i_list)
    coplanar = di_max <= coplanar_threshold_deg

    return CoplanarityResult(
        n_planets=len(valid),
        delta_i_max_deg=round(di_max, 3),
        delta_i_min_deg=round(di_min, 3),
        coplanar=coplanar,
        flag="OK",
    )


def format_coplanarity_result(r: CoplanarityResult) -> str:
    def _f(v: float) -> str:
        return f"{v:.3f}" if math.isfinite(v) else "N/A"

    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| N planets | {r.n_planets} |\n"
        f"| Max mutual inclination (deg) | {_f(r.delta_i_max_deg)} |\n"
        f"| Min mutual inclination (deg) | {_f(r.delta_i_min_deg)} |\n"
        f"| Coplanar (< threshold) | {r.coplanar} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Check multi-planet coplanarity from durations.")
    p.add_argument("--stellar-radius-rsun", type=float, default=1.0)
    p.add_argument("--coplanar-threshold-deg", type=float, default=5.0)
    p.parse_args()
    import sys
    print("Usage: provide periods_json and durations_json as positional args.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(_cli())
