"""Check the T14/T23 duration ratio for transit geometry plausibility."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DurationRatioResult:
    t14_hours: float
    t23_hours: float
    ratio: float
    impact_parameter_est: float
    ingress_fraction: float
    flag: str


def check_duration_ratio(
    t14_hours: float,
    t23_hours: float,
    depth_ppm: float = 1000.0,
) -> DurationRatioResult:
    """
    Check plausibility of T14/T23 transit duration ratio.

    T14 = total (first to fourth contact), T23 = flat bottom (second to third).
    Ingress fraction = (T14 - T23) / T14 = 1 - T23/T14.
    Estimate impact parameter b from depth and ratio via Seager & Mallén-Ornelas (2003).

    Physical constraints:
    - T23 <= T14 (flat bottom can't exceed total)
    - ratio = T14/T23 >= 1
    - Very high ratio → grazing transit
    """
    for name, val in [("t14_hours", t14_hours), ("t23_hours", t23_hours)]:
        if not math.isfinite(val) or val <= 0.0:
            return DurationRatioResult(
                t14_hours=t14_hours, t23_hours=t23_hours,
                ratio=float("nan"), impact_parameter_est=float("nan"),
                ingress_fraction=float("nan"), flag=f"INVALID_{name.upper()}",
            )

    if t23_hours > t14_hours:
        return DurationRatioResult(
            t14_hours=t14_hours, t23_hours=t23_hours,
            ratio=float("nan"), impact_parameter_est=float("nan"),
            ingress_fraction=float("nan"), flag="T23_EXCEEDS_T14",
        )

    ratio = t14_hours / t23_hours
    ingress_frac = 1.0 - t23_hours / t14_hours

    # Estimate impact parameter from ratio and depth
    # ratio^2 ≈ (1 + k)^2 - b^2(1 - (T23/T14)^2 * something) — simplified:
    # For b~0: ratio = (1+k)/(1-k) where k = sqrt(depth_ppm/1e6)
    k = math.sqrt(max(depth_ppm, 1.0) * 1e-6)
    if ratio > 0 and k < 1.0:
        # Approximate: b^2 ≈ 1 - (T23/T14)^2 * (1 - k^2) / (1 - (k/ratio)^2)
        r23 = t23_hours / t14_hours
        denom = 1.0 - (k / ratio) ** 2 if ratio > k else 1e-6
        b2 = max(0.0, 1.0 - r23**2 * (1.0 - k**2) / denom)
        b = math.sqrt(b2)
    else:
        b = 0.0

    flag: str
    if ratio > 5.0:
        flag = "GRAZING_TRANSIT"
    elif ingress_frac > 0.8:
        flag = "HIGH_INGRESS_FRACTION"
    else:
        flag = "OK"

    return DurationRatioResult(
        t14_hours=t14_hours,
        t23_hours=t23_hours,
        ratio=round(ratio, 4),
        impact_parameter_est=round(min(b, 1.0), 4),
        ingress_fraction=round(ingress_frac, 4),
        flag=flag,
    )


def format_duration_ratio_result(r: DurationRatioResult) -> str:
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| T14 (hours) | {r.t14_hours:.4f} |\n"
        f"| T23 (hours) | {r.t23_hours:.4f} |\n"
        f"| T14/T23 ratio | {r.ratio:.4f} |\n"
        f"| Ingress fraction | {r.ingress_fraction:.4f} |\n"
        f"| Impact parameter (est) | {r.impact_parameter_est:.4f} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Check T14/T23 transit duration ratio.")
    p.add_argument("t14_hours", type=float)
    p.add_argument("t23_hours", type=float)
    p.add_argument("--depth-ppm", type=float, default=1000.0)
    args = p.parse_args()
    r = check_duration_ratio(args.t14_hours, args.t23_hours, args.depth_ppm)
    print(format_duration_ratio_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
