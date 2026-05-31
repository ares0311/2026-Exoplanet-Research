"""Estimate eccentricity e·cos(ω) from secondary eclipse phase offset."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class EccentricityResult:
    secondary_phase: float
    e_cos_omega: float
    eccentricity_lower: float
    flag: str


def estimate_eccentricity(
    secondary_phase: float,
    phase_err: float = 0.01,
) -> EccentricityResult:
    """
    Estimate e·cos(ω) from the secondary eclipse phase offset.

    For a circular orbit the secondary occurs at phase 0.5.
    The offset δ = phase - 0.5 gives:
        e·cos(ω) ≈ π * δ / 2

    This is the leading-order approximation (Kopal 1959).
    Returns eccentricity_lower = |e·cos(ω)| as a lower bound on e.
    """
    if not math.isfinite(secondary_phase) or not (0.0 < secondary_phase < 1.0):
        return EccentricityResult(
            secondary_phase=secondary_phase,
            e_cos_omega=float("nan"),
            eccentricity_lower=float("nan"),
            flag="INVALID_SECONDARY_PHASE",
        )
    if not math.isfinite(phase_err) or phase_err < 0.0:
        return EccentricityResult(
            secondary_phase=secondary_phase,
            e_cos_omega=float("nan"),
            eccentricity_lower=float("nan"),
            flag="INVALID_PHASE_ERR",
        )

    delta = secondary_phase - 0.5
    e_cos_w = math.pi * delta / 2.0
    e_lower = abs(e_cos_w)

    # Significance: |δ| > 3 * phase_err → eccentricity detected
    sigma = abs(delta) / phase_err if phase_err > 0.0 else float("inf")
    if sigma < 3.0:
        flag = "CONSISTENT_CIRCULAR"
    elif e_lower > 0.9:
        flag = "IMPLAUSIBLE_ECCENTRICITY"
    else:
        flag = "ECCENTRIC_ORBIT"

    return EccentricityResult(
        secondary_phase=secondary_phase,
        e_cos_omega=round(e_cos_w, 5),
        eccentricity_lower=round(e_lower, 5),
        flag=flag,
    )


def format_eccentricity_result(r: EccentricityResult) -> str:
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Secondary phase | {r.secondary_phase:.5f} |\n"
        f"| e·cos(ω) | {r.e_cos_omega:.5f} |\n"
        f"| e lower bound | {r.eccentricity_lower:.5f} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Estimate e·cos(ω) from secondary eclipse timing.")
    p.add_argument("secondary_phase", type=float)
    p.add_argument("--phase-err", type=float, default=0.01)
    args = p.parse_args()
    r = estimate_eccentricity(args.secondary_phase, args.phase_err)
    print(format_eccentricity_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
