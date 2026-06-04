"""Analyze RV bisector span to flag stellar activity contamination."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BisectorResult:
    n_points: int
    bis_span_ms: float         # bisector inverse slope (BIS) span in m/s
    bis_slope: float           # slope of BIS vs RV (Pearson r)
    bis_rms_ms: float          # RMS of BIS measurements
    activity_flagged: bool     # |correlation| > threshold
    flag: str


def analyze_bisector(
    rv_ms: list[float],
    bis_ms: list[float],
    correlation_threshold: float = 0.5,
) -> BisectorResult:
    """
    Analyze RV bisector inverse slope (BIS) for stellar activity contamination.

    A significant anti-correlation between BIS and RV (Pearson |r| > threshold)
    indicates the RV signal is due to stellar activity rather than a planet.

    Parameters
    ----------
    rv_ms:                 Radial velocity measurements (m/s).
    bis_ms:                Bisector inverse slope measurements (m/s).
    correlation_threshold: |Pearson r| above which activity is flagged.
    """
    n = len(rv_ms)
    if n < 3:
        return BisectorResult(
            n_points=n, bis_span_ms=float("nan"), bis_slope=float("nan"),
            bis_rms_ms=float("nan"), activity_flagged=False,
            flag="INSUFFICIENT_POINTS",
        )
    if len(bis_ms) != n:
        return BisectorResult(
            n_points=n, bis_span_ms=float("nan"), bis_slope=float("nan"),
            bis_rms_ms=float("nan"), activity_flagged=False,
            flag="LENGTH_MISMATCH",
        )

    valid = [
        (rv, bis) for rv, bis in zip(rv_ms, bis_ms, strict=True)
        if math.isfinite(rv) and math.isfinite(bis)
    ]
    if len(valid) < 3:
        return BisectorResult(
            n_points=n, bis_span_ms=float("nan"), bis_slope=float("nan"),
            bis_rms_ms=float("nan"), activity_flagged=False,
            flag="INSUFFICIENT_FINITE",
        )

    rvs = [v[0] for v in valid]
    bis = [v[1] for v in valid]
    nv = len(valid)

    # BIS span and RMS
    bis_span = max(bis) - min(bis)
    bis_mean = sum(bis) / nv
    bis_rms = math.sqrt(sum((b - bis_mean) ** 2 for b in bis) / nv)

    # Pearson r between RV and BIS
    rv_mean = sum(rvs) / nv
    cov = sum((rv - rv_mean) * (b - bis_mean) for rv, b in zip(rvs, bis, strict=True)) / nv
    rv_std = math.sqrt(sum((rv - rv_mean) ** 2 for rv in rvs) / nv)
    bis_std = math.sqrt(sum((b - bis_mean) ** 2 for b in bis) / nv)

    pearson_r = cov / (rv_std * bis_std) if rv_std > 0 and bis_std > 0 else 0.0

    activity_flagged = abs(pearson_r) > correlation_threshold

    return BisectorResult(
        n_points=nv,
        bis_span_ms=round(bis_span, 4),
        bis_slope=round(pearson_r, 4),
        bis_rms_ms=round(bis_rms, 4),
        activity_flagged=activity_flagged,
        flag="OK",
    )


def format_bisector_result(r: BisectorResult) -> str:
    def _f(v: float) -> str:
        return f"{v:.4f}" if math.isfinite(v) else "N/A"

    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| N points | {r.n_points} |\n"
        f"| BIS span (m/s) | {_f(r.bis_span_ms)} |\n"
        f"| BIS–RV correlation (Pearson r) | {_f(r.bis_slope)} |\n"
        f"| BIS RMS (m/s) | {_f(r.bis_rms_ms)} |\n"
        f"| Activity flagged | {r.activity_flagged} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Analyze RV bisector for stellar activity.")
    p.add_argument("rv_json", help="JSON array of RV values (m/s)")
    p.add_argument("bis_json", help="JSON array of BIS values (m/s)")
    p.add_argument("--correlation-threshold", type=float, default=0.5)
    args = p.parse_args()
    import json
    rv = json.loads(args.rv_json)
    bis = json.loads(args.bis_json)
    r = analyze_bisector(rv, bis, args.correlation_threshold)
    print(format_bisector_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
