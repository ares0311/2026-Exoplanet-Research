"""Estimate Rossiter–McLaughlin sensitivity to spin-orbit misalignment."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SpinOrbitAlignmentResult:
    v_sin_i_ms: float
    depth: float
    impact_parameter: float
    rv_precision_ms: float
    max_rm_amplitude_ms: float      # peak RM for aligned system
    min_detectable_lambda_deg: float  # minimum resolvable obliquity (degrees)
    lambda_sensitivity_ms_per_deg: float  # how much RM changes per degree of obliquity
    measurement_feasible: bool      # True if max_rm_amplitude > 3 * rv_precision
    flag: str


def compute_spin_orbit_alignment(
    v_sin_i_ms: float,
    depth: float,
    impact_parameter: float = 0.0,
    rv_precision_ms: float = 1.0,
) -> SpinOrbitAlignmentResult:
    """
    Estimate spin-orbit alignment sensitivity from the Rossiter–McLaughlin effect.

    Maximum RM amplitude (aligned, λ=0):
      RM_max ≈ v*sin(i) * δ * sqrt(1 − b²)

    The RM signal scales with projected obliquity λ as cos(λ) to first order
    (it modifies the ratio of prograde to retrograde flux).
    Sensitivity: ΔRM ≈ RM_max * |cos(λ) − cos(λ + Δλ)| ≈ RM_max * sin(λ) * Δλ
    Minimum detectable Δλ ≈ 3 * σ_rv / (RM_max * sin(90°)) at λ = 90° (worst case).

    Parameters
    ----------
    v_sin_i_ms:       Stellar v*sin(i) in m/s.
    depth:            Transit depth (dimensionless).
    impact_parameter: Impact parameter (0 = central transit).
    rv_precision_ms:  Single-epoch RV precision in m/s.
    """
    if not math.isfinite(v_sin_i_ms) or v_sin_i_ms < 0:
        return SpinOrbitAlignmentResult(v_sin_i_ms, depth, impact_parameter, rv_precision_ms,
                                        float("nan"), float("nan"), float("nan"),
                                        False, "INVALID_VSINI")
    if not math.isfinite(depth) or depth <= 0 or depth >= 1:
        return SpinOrbitAlignmentResult(v_sin_i_ms, depth, impact_parameter, rv_precision_ms,
                                        float("nan"), float("nan"), float("nan"),
                                        False, "INVALID_DEPTH")
    if not math.isfinite(impact_parameter) or impact_parameter < 0 or impact_parameter >= 1:
        return SpinOrbitAlignmentResult(v_sin_i_ms, depth, impact_parameter, rv_precision_ms,
                                        float("nan"), float("nan"), float("nan"),
                                        False, "INVALID_IMPACT")

    max_rm = v_sin_i_ms * depth * math.sqrt(1.0 - impact_parameter ** 2)

    if max_rm > 0:
        sensitivity = max_rm / 57.296  # dRM/dλ at λ=0 is RM_max in m/s per radian → /rad * π/180
        min_lambda_rad = (3.0 * rv_precision_ms) / max_rm
        min_lambda_deg = math.degrees(min_lambda_rad)
    else:
        sensitivity = 0.0
        min_lambda_deg = float("inf")

    feasible = max_rm >= 3.0 * rv_precision_ms

    return SpinOrbitAlignmentResult(
        v_sin_i_ms=v_sin_i_ms,
        depth=depth,
        impact_parameter=impact_parameter,
        rv_precision_ms=rv_precision_ms,
        max_rm_amplitude_ms=round(max_rm, 4),
        min_detectable_lambda_deg=(
            round(min_lambda_deg, 2) if math.isfinite(min_lambda_deg) else float("inf")
        ),
        lambda_sensitivity_ms_per_deg=round(sensitivity, 6),
        measurement_feasible=feasible,
        flag="OK",
    )


def format_spin_orbit_result(r: SpinOrbitAlignmentResult) -> str:
    def _f(v: float, fmt: str = ".4f") -> str:
        return format(v, fmt) if math.isfinite(v) else "N/A"

    return (
        f"| Parameter | Value |\n|---|---|\n"
        f"| v sin i (m/s) | {_f(r.v_sin_i_ms, '.2f')} |\n"
        f"| Max RM amplitude (m/s) | {_f(r.max_rm_amplitude_ms)} |\n"
        f"| Min detectable λ (deg) | {_f(r.min_detectable_lambda_deg, '.2f')} |\n"
        f"| RM sensitivity (m/s/deg) | {_f(r.lambda_sensitivity_ms_per_deg)} |\n"
        f"| Measurement feasible | {r.measurement_feasible} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(
        description="Estimate spin-orbit alignment measurement sensitivity."
    )
    p.add_argument("v_sin_i_ms", type=float)
    p.add_argument("depth", type=float)
    p.add_argument("--impact-parameter", type=float, default=0.0)
    p.add_argument("--rv-precision-ms", type=float, default=1.0)
    args = p.parse_args()
    r = compute_spin_orbit_alignment(
        args.v_sin_i_ms, args.depth,
        impact_parameter=args.impact_parameter,
        rv_precision_ms=args.rv_precision_ms,
    )
    print(format_spin_orbit_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
