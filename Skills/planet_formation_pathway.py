"""Assess likely planet formation pathway from bulk density and orbital period."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

# Bulk density thresholds (g/cm^3)
_RHO_ROCKY_MIN = 3.0    # ≥ 3 g/cm³ → rocky / iron-rich
_RHO_WATER_MAX = 3.0    # < 3 g/cm³ → volatile-rich / water world
_RHO_GAS_MAX = 1.3      # < 1.3 g/cm³ → gas giant
_RHO_ICE_MAX = 2.0      # 1.3–2 g/cm³ → ice giant


@dataclass(frozen=True)
class FormationPathwayResult:
    period_days: float
    bulk_density_gcc: float
    core_accretion_prob: float
    disk_instability_prob: float
    migration_prob: float
    in_situ_prob: float
    most_likely_pathway: str
    flag: str


def assess_formation_pathway(
    period_days: float,
    bulk_density_gcc: float,
    radius_rearth: float | None = None,
) -> FormationPathwayResult:
    """
    Estimate relative likelihood of formation pathways from density and period.

    Core accretion:    favoured for rocky/icy planets, any period
    Disk instability:  favoured for massive, low-density planets at wide orbits
    Migration:         favoured for hot/warm giants (originally at > 1 AU)
    In situ:           favoured for close-in super-Earths / sub-Neptunes

    Returns normalised probability weights (sum ≈ 1).
    Intended as a qualitative prior, not a rigorous Bayesian posterior.
    """
    if not math.isfinite(period_days) or period_days <= 0.0:
        return FormationPathwayResult(
            period_days=period_days, bulk_density_gcc=bulk_density_gcc,
            core_accretion_prob=float("nan"), disk_instability_prob=float("nan"),
            migration_prob=float("nan"), in_situ_prob=float("nan"),
            most_likely_pathway="UNKNOWN", flag="INVALID_PERIOD",
        )
    if not math.isfinite(bulk_density_gcc) or bulk_density_gcc <= 0.0:
        return FormationPathwayResult(
            period_days=period_days, bulk_density_gcc=bulk_density_gcc,
            core_accretion_prob=float("nan"), disk_instability_prob=float("nan"),
            migration_prob=float("nan"), in_situ_prob=float("nan"),
            most_likely_pathway="UNKNOWN", flag="INVALID_DENSITY",
        )

    rho = bulk_density_gcc
    p = period_days

    # --- Core accretion score ---
    # Highest for rocky (rho > 3) or icy (rho 1–3) at any period
    ca = 0.5
    if rho >= _RHO_ROCKY_MIN:
        ca = 0.80
    elif rho >= _RHO_ICE_MAX:
        ca = 0.65
    # Slightly penalised for close-in gas giants (hard to form in situ)
    if rho < _RHO_GAS_MAX and p < 10.0:
        ca = max(0.10, ca - 0.20)

    # --- Disk instability score ---
    # Favoured for: very low density (gas giant), wide orbit
    di = 0.05
    if rho < _RHO_GAS_MAX:
        di = 0.30
        if p > 100.0:
            di = 0.55

    # --- Migration score ---
    # Favoured for: hot/warm Jupiters (low density, close orbit)
    mg = 0.10
    if rho < _RHO_GAS_MAX and p < 100.0:
        mg = 0.50
        if p < 10.0:
            mg = 0.65

    # --- In situ score ---
    # Favoured for: small rocky/sub-Neptune at < 50 days
    ins = 0.05
    if p < 50.0 and rho > _RHO_ICE_MAX:
        ins = 0.35
    elif p < 10.0 and rho >= _RHO_ROCKY_MIN:
        ins = 0.45

    total = ca + di + mg + ins
    if total <= 0:
        total = 1.0

    ca /= total
    di /= total
    mg /= total
    ins /= total

    pathways = {"CORE_ACCRETION": ca, "DISK_INSTABILITY": di,
                "MIGRATION": mg, "IN_SITU": ins}
    most_likely = max(pathways, key=lambda k: pathways[k])

    return FormationPathwayResult(
        period_days=period_days,
        bulk_density_gcc=bulk_density_gcc,
        core_accretion_prob=round(ca, 3),
        disk_instability_prob=round(di, 3),
        migration_prob=round(mg, 3),
        in_situ_prob=round(ins, 3),
        most_likely_pathway=most_likely,
        flag="OK",
    )


def format_formation_result(r: FormationPathwayResult) -> str:
    def _f(v: float) -> str:
        return f"{v:.3f}" if math.isfinite(v) else "N/A"

    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Period (days) | {r.period_days:.3f} |\n"
        f"| Bulk density (g/cm³) | {r.bulk_density_gcc:.3f} |\n"
        f"| P(core accretion) | {_f(r.core_accretion_prob)} |\n"
        f"| P(disk instability) | {_f(r.disk_instability_prob)} |\n"
        f"| P(migration) | {_f(r.migration_prob)} |\n"
        f"| P(in situ) | {_f(r.in_situ_prob)} |\n"
        f"| Most likely pathway | {r.most_likely_pathway} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Assess planet formation pathway.")
    p.add_argument("period_days", type=float)
    p.add_argument("bulk_density_gcc", type=float)
    args = p.parse_args()
    r = assess_formation_pathway(args.period_days, args.bulk_density_gcc)
    print(format_formation_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
