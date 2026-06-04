"""Validate a transit signal's consistency across multiple TESS sectors."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CrossSectorValidationResult:
    n_sectors: int
    period_consistent: bool
    depth_consistent: bool
    duration_consistent: bool
    n_inconsistent: int
    overall: str  # CONSISTENT | INCONSISTENT | PARTIAL
    flag: str


def _weighted_mean_std(values: list[float], errors: list[float]) -> tuple[float, float]:
    weights = [1.0 / (e**2) for e in errors if e > 0]
    if not weights:
        mean = sum(values) / len(values)
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
        return mean, std
    w_sum = sum(weights)
    mean = sum(w * v for w, v in zip(weights, values, strict=False)) / w_sum
    std = math.sqrt(1.0 / w_sum)
    return mean, std


def validate_cross_sector(
    sector_measurements: list[dict],
    period_rtol: float = 0.01,
    depth_rtol: float = 0.30,
    duration_rtol: float = 0.20,
) -> CrossSectorValidationResult:
    """
    Validate transit consistency across sectors.

    Each measurement dict may contain:
    - period_days, period_err_days
    - depth_ppm, depth_err_ppm
    - duration_hours, duration_err_hours
    - sector (int, optional)

    Consistency: each value within rtol of the weighted mean.
    """
    n = len(sector_measurements)
    if n < 2:
        return CrossSectorValidationResult(
            n_sectors=n, period_consistent=True, depth_consistent=True,
            duration_consistent=True, n_inconsistent=0,
            overall="CONSISTENT", flag="SINGLE_SECTOR",
        )

    def _check_param(key: str, err_key: str, rtol: float) -> bool:
        vals = [m[key] for m in sector_measurements if key in m and math.isfinite(m[key])]
        errs = [
            m.get(err_key, abs(m[key]) * 0.05) for m in sector_measurements
            if key in m and math.isfinite(m[key])
        ]
        if len(vals) < 2:
            return True
        mean, _ = _weighted_mean_std(vals, errs)
        if mean == 0:
            return True
        return all(abs(v - mean) / abs(mean) <= rtol for v in vals)

    period_ok = _check_param("period_days", "period_err_days", period_rtol)
    depth_ok = _check_param("depth_ppm", "depth_err_ppm", depth_rtol)
    dur_ok = _check_param("duration_hours", "duration_err_hours", duration_rtol)

    n_inconsistent = sum(1 for ok in [period_ok, depth_ok, dur_ok] if not ok)

    if n_inconsistent == 0:
        overall = "CONSISTENT"
    elif n_inconsistent == 3:
        overall = "INCONSISTENT"
    else:
        overall = "PARTIAL"

    return CrossSectorValidationResult(
        n_sectors=n,
        period_consistent=period_ok,
        depth_consistent=depth_ok,
        duration_consistent=dur_ok,
        n_inconsistent=n_inconsistent,
        overall=overall,
        flag="OK",
    )


def format_cross_sector_result(r: CrossSectorValidationResult) -> str:
    def _yn(b: bool) -> str:
        return "YES" if b else "NO"

    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| N sectors | {r.n_sectors} |\n"
        f"| Period consistent | {_yn(r.period_consistent)} |\n"
        f"| Depth consistent | {_yn(r.depth_consistent)} |\n"
        f"| Duration consistent | {_yn(r.duration_consistent)} |\n"
        f"| N inconsistent | {r.n_inconsistent} |\n"
        f"| Overall | {r.overall} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Validate cross-sector transit consistency.")
    p.add_argument("measurements_json", help="JSON array of sector measurement dicts or @file")
    args = p.parse_args()
    raw = args.measurements_json
    if raw.startswith("@"):
        with open(raw[1:]) as f:
            measurements = json.load(f)
    else:
        measurements = json.loads(raw)
    r = validate_cross_sector(measurements)
    print(format_cross_sector_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
