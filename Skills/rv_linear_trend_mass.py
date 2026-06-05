"""Estimate minimum mass of outer companion from RV linear trend."""
from __future__ import annotations

import math
from dataclasses import dataclass

_G = 6.674e-11
_MSUN_KG = 1.989e30
_MJUP_KG = 1.898e27
_AU_M = 1.495978707e11
_SEC_PER_YR = 3.156e7


@dataclass(frozen=True)
class RvLinearTrendResult:
    trend_ms_per_day: float
    baseline_days: float
    outer_period_lower_bound_yr: float
    min_companion_mass_mjup: float
    min_companion_mass_msun: float
    companion_class: str   # STELLAR / BROWN_DWARF / PLANET
    flag: str


def compute_rv_trend_companion_mass(
    trend_ms_per_day: float,
    baseline_days: float,
    stellar_mass_msun: float = 1.0,
    assumed_separation_au: float | None = None,
) -> RvLinearTrendResult:
    """Estimate outer companion minimum mass from a linear RV trend.

    A linear trend dv/dt implies a companion at a_c > v_baseline / (2π × dv/dt × P_c/2π).
    For a circular orbit: K_c = (2πG/P_c)^(1/3) × Mc / (Ms + Mc)^(2/3)
    and dv/dt ≈ K_c × (2π/P_c).

    Lower bound on Mc*sin(i) given separation a ≥ baseline_au:
      a_min = (G Ms (T/2)²/(4π²))^(1/3)  [orbit half-completed in baseline]
      Mc*sin(i) ≥ dv/dt × Ms^(2/3) × a_min / G × sqrt(G Ms a_min)

    Args:
        trend_ms_per_day: RV linear trend (m/s per day); positive = accelerating
        baseline_days: total RV baseline (days)
        stellar_mass_msun: stellar mass (solar masses)
        assumed_separation_au: if supplied, compute mass at this specific separation
    """
    if baseline_days <= 0.0:
        return RvLinearTrendResult(trend_ms_per_day, baseline_days, float("nan"),
                                    float("nan"), float("nan"), "UNKNOWN", "INVALID_BASELINE")
    if stellar_mass_msun <= 0.0:
        return RvLinearTrendResult(trend_ms_per_day, baseline_days, float("nan"),
                                    float("nan"), float("nan"), "UNKNOWN", "INVALID_STELLAR_MASS")
    if trend_ms_per_day == 0.0:
        return RvLinearTrendResult(trend_ms_per_day, baseline_days, float("nan"),
                                    float("nan"), float("nan"), "UNKNOWN", "ZERO_TREND")

    ms_kg = stellar_mass_msun * _MSUN_KG
    dvdt_ms2 = abs(trend_ms_per_day) / 86400.0

    if assumed_separation_au is not None:
        if assumed_separation_au <= 0.0:
            return RvLinearTrendResult(trend_ms_per_day, baseline_days, float("nan"),
                                        float("nan"), float("nan"), "UNKNOWN",
                                        "INVALID_SEPARATION")
        a_m = assumed_separation_au * _AU_M
    else:
        # Minimum separation: companion completes at most half an orbit during baseline
        t_half = baseline_days * 86400.0 / 2.0
        a_m = (_G * ms_kg * t_half**2 / (4.0 * math.pi**2)) ** (1.0 / 3.0)

    a_au = a_m / _AU_M
    p_yr = math.sqrt(a_au**3 / stellar_mass_msun)

    v_circ = math.sqrt(_G * ms_kg / a_m)
    mc_sini_kg = dvdt_ms2 * a_m / _G * math.sqrt(_G * ms_kg * a_m)
    mc_sini_kg = dvdt_ms2 * ms_kg * a_m / v_circ

    mc_mjup = mc_sini_kg / _MJUP_KG
    mc_msun = mc_sini_kg / _MSUN_KG

    if mc_msun > 0.08:
        companion_class = "STELLAR"
    elif mc_mjup > 13.0:
        companion_class = "BROWN_DWARF"
    else:
        companion_class = "PLANET"

    return RvLinearTrendResult(
        trend_ms_per_day=trend_ms_per_day,
        baseline_days=baseline_days,
        outer_period_lower_bound_yr=p_yr,
        min_companion_mass_mjup=mc_mjup,
        min_companion_mass_msun=mc_msun,
        companion_class=companion_class,
        flag="OK",
    )


def format_rv_trend_result(r: RvLinearTrendResult) -> str:
    if r.flag != "OK":
        return f"RvLinearTrend | flag={r.flag}"
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| RV trend | {r.trend_ms_per_day:+.4f} m/s/day |\n"
        f"| Baseline | {r.baseline_days:.0f} days |\n"
        f"| Period lower bound | {r.outer_period_lower_bound_yr:.1f} yr |\n"
        f"| Min Mc·sin(i) | {r.min_companion_mass_mjup:.2f} MJup |\n"
        f"| Min Mc·sin(i) | {r.min_companion_mass_msun:.4f} M☉ |\n"
        f"| Companion class | {r.companion_class} |\n"
        f"| Flag | {r.flag} |"
    )


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="RV trend companion mass estimator")
    p.add_argument("trend_ms_per_day", type=float)
    p.add_argument("baseline_days", type=float)
    p.add_argument("--mstar", type=float, default=1.0)
    p.add_argument("--sep-au", type=float, default=None)
    args = p.parse_args()
    r = compute_rv_trend_companion_mass(
        args.trend_ms_per_day, args.baseline_days,
        stellar_mass_msun=args.mstar, assumed_separation_au=args.sep_au,
    )
    print(format_rv_trend_result(r))


if __name__ == "__main__":
    _cli()
