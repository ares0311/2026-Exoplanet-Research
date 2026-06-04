"""Estimate the Rossiter–McLaughlin (RM) effect amplitude for a transiting planet."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RmAmplitudeResult:
    v_sin_i_ms: float
    depth: float
    impact_parameter: float
    rm_amplitude_ms: float
    flag: str


def compute_rm_amplitude(
    v_sin_i_ms: float,
    depth: float,
    impact_parameter: float = 0.0,
) -> RmAmplitudeResult:
    """
    Estimate the Rossiter–McLaughlin amplitude.

    RM_amp ≈ v*sin(i) * δ * sqrt(1 − b²)   (Gaudi & Winn 2007, simplified)
    where δ is the transit depth (Rp/Rs)² and b is the impact parameter.

    Parameters
    ----------
    v_sin_i_ms:       Stellar projected rotation velocity in m/s.
    depth:            Transit depth (dimensionless, 0 < depth < 1).
    impact_parameter: Impact parameter b (0 = central transit, <1 = transiting).
    """
    if not math.isfinite(v_sin_i_ms) or v_sin_i_ms < 0:
        return RmAmplitudeResult(v_sin_i_ms, depth, impact_parameter, float("nan"), "INVALID_VSINI")
    if not math.isfinite(depth) or depth <= 0 or depth >= 1:
        return RmAmplitudeResult(v_sin_i_ms, depth, impact_parameter, float("nan"), "INVALID_DEPTH")
    if not math.isfinite(impact_parameter) or impact_parameter < 0 or impact_parameter >= 1:
        return RmAmplitudeResult(
            v_sin_i_ms, depth, impact_parameter, float("nan"), "INVALID_IMPACT"
        )

    rm = v_sin_i_ms * depth * math.sqrt(1.0 - impact_parameter ** 2)
    return RmAmplitudeResult(
        v_sin_i_ms=v_sin_i_ms,
        depth=depth,
        impact_parameter=impact_parameter,
        rm_amplitude_ms=round(rm, 4),
        flag="OK",
    )


def format_rm_amplitude_result(r: RmAmplitudeResult) -> str:
    def _f(v: float, fmt: str = ".4f") -> str:
        return format(v, fmt) if math.isfinite(v) else "N/A"

    return (
        f"| Parameter | Value |\n|---|---|\n"
        f"| v sin i (m/s) | {_f(r.v_sin_i_ms, '.2f')} |\n"
        f"| Transit depth | {_f(r.depth)} |\n"
        f"| Impact parameter | {_f(r.impact_parameter)} |\n"
        f"| RM amplitude (m/s) | {_f(r.rm_amplitude_ms, '.4f')} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Estimate Rossiter–McLaughlin amplitude.")
    p.add_argument("v_sin_i_ms", type=float, help="Stellar v*sin(i) in m/s")
    p.add_argument("depth", type=float, help="Transit depth (0–1)")
    p.add_argument("--impact-parameter", type=float, default=0.0)
    args = p.parse_args()
    r = compute_rm_amplitude(args.v_sin_i_ms, args.depth, args.impact_parameter)
    print(format_rm_amplitude_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
