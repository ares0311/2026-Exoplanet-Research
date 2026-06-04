"""Track RMS noise floor trend across TESS sectors and flag degradation."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class NoiseTrendResult:
    n_sectors: int
    mean_rms_ppm: float
    rms_trend_ppm_per_sector: float   # positive = worsening
    trend_significance: float
    degradation_detected: bool
    worst_sector: int | None
    worst_rms_ppm: float
    flag: str


def track_noise_trend(
    sector_numbers: list[int],
    rms_ppm_values: list[float],
    degradation_threshold_sigma: float = 3.0,
) -> NoiseTrendResult:
    """
    Track RMS noise floor across sectors and flag upward degradation trend.

    Fits a linear trend to RMS vs sector number.
    degradation_detected when trend slope > 0 with significance > threshold.

    Parameters
    ----------
    sector_numbers:  TESS sector identifiers (integers).
    rms_ppm_values:  Per-sector RMS scatter in ppm.
    degradation_threshold_sigma: Significance threshold for degradation flag.
    """
    n = len(sector_numbers)
    if n < 2:
        return NoiseTrendResult(
            n_sectors=n, mean_rms_ppm=float("nan"),
            rms_trend_ppm_per_sector=float("nan"),
            trend_significance=float("nan"), degradation_detected=False,
            worst_sector=None, worst_rms_ppm=float("nan"),
            flag="INSUFFICIENT_SECTORS",
        )
    if len(rms_ppm_values) != n:
        return NoiseTrendResult(
            n_sectors=n, mean_rms_ppm=float("nan"),
            rms_trend_ppm_per_sector=float("nan"),
            trend_significance=float("nan"), degradation_detected=False,
            worst_sector=None, worst_rms_ppm=float("nan"),
            flag="LENGTH_MISMATCH",
        )

    valid = [(s, r) for s, r in zip(sector_numbers, rms_ppm_values, strict=False)
             if math.isfinite(r) and r > 0]
    if len(valid) < 2:
        return NoiseTrendResult(
            n_sectors=n, mean_rms_ppm=float("nan"),
            rms_trend_ppm_per_sector=float("nan"),
            trend_significance=float("nan"), degradation_detected=False,
            worst_sector=None, worst_rms_ppm=float("nan"),
            flag="INSUFFICIENT_FINITE_VALUES",
        )

    xs = [float(v[0]) for v in valid]
    ys = [v[1] for v in valid]
    nv = len(xs)

    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x**2 for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys, strict=False))
    det = nv * sxx - sx**2
    slope = (nv * sxy - sx * sy) / det if abs(det) > 1e-12 else 0.0
    intercept = (sy - slope * sx) / nv

    residuals = [y - (slope * x + intercept) for x, y in zip(xs, ys, strict=False)]
    rms_res = math.sqrt(sum(r**2 for r in residuals) / nv) if nv > 0 else 0.0
    noise_floor = rms_res / math.sqrt(nv) if nv > 0 else float("inf")
    significance = abs(slope) / noise_floor if noise_floor > 0 else 0.0

    mean_rms = sum(ys) / len(ys)
    worst_idx = max(range(len(valid)), key=lambda i: valid[i][1])
    worst_sec = valid[worst_idx][0]
    worst_rms = valid[worst_idx][1]

    degradation = slope > 0 and significance > degradation_threshold_sigma

    return NoiseTrendResult(
        n_sectors=nv,
        mean_rms_ppm=round(mean_rms, 2),
        rms_trend_ppm_per_sector=round(slope, 4),
        trend_significance=round(significance, 3),
        degradation_detected=degradation,
        worst_sector=int(worst_sec),
        worst_rms_ppm=round(worst_rms, 2),
        flag="DEGRADATION_DETECTED" if degradation else "OK",
    )


def format_noise_trend_result(r: NoiseTrendResult) -> str:
    def _f(v: float, fmt: str = ".3f") -> str:
        return format(v, fmt) if math.isfinite(v) else "N/A"

    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| N sectors | {r.n_sectors} |\n"
        f"| Mean RMS (ppm) | {_f(r.mean_rms_ppm, '.2f')} |\n"
        f"| Trend (ppm/sector) | {_f(r.rms_trend_ppm_per_sector, '.4f')} |\n"
        f"| Trend significance (σ) | {_f(r.trend_significance)} |\n"
        f"| Degradation detected | {r.degradation_detected} |\n"
        f"| Worst sector | {r.worst_sector if r.worst_sector is not None else 'N/A'} |\n"
        f"| Worst sector RMS (ppm) | {_f(r.worst_rms_ppm, '.2f')} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Track noise trend across TESS sectors.")
    p.add_argument("sectors_json", help="JSON array of sector numbers")
    p.add_argument("rms_json", help="JSON array of RMS values in ppm")
    p.add_argument("--degradation-threshold-sigma", type=float, default=3.0)
    args = p.parse_args()
    import json
    sectors = json.loads(args.sectors_json)
    rms_vals = json.loads(args.rms_json)
    r = track_noise_trend(sectors, rms_vals, args.degradation_threshold_sigma)
    print(format_noise_trend_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
