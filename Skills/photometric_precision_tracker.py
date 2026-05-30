"""
Tracks night-to-night photometric precision from a list of per-night RMS measurements.

Public API:
    PrecisionTrackResult  -- frozen dataclass holding precision statistics
    track_precision(night_rms_mmag) -> PrecisionTrackResult
    format_precision_track(result) -> str
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PrecisionTrackResult:
    n_nights: int
    mean_rms_mmag: float
    std_rms_mmag: float
    worst_night_idx: int
    best_night_idx: int
    flag: str


def track_precision(night_rms_mmag: list[float]) -> PrecisionTrackResult:
    n = len(night_rms_mmag)
    if n == 0:
        return PrecisionTrackResult(
            n_nights=0,
            mean_rms_mmag=0.0,
            std_rms_mmag=0.0,
            worst_night_idx=-1,
            best_night_idx=-1,
            flag="NO_DATA",
        )

    mean_rms = sum(night_rms_mmag) / n

    if n > 1:
        std_rms = math.sqrt(
            sum((v - mean_rms) ** 2 for v in night_rms_mmag) / (n - 1)
        )
    else:
        std_rms = 0.0

    worst_idx = night_rms_mmag.index(max(night_rms_mmag))
    best_idx = night_rms_mmag.index(min(night_rms_mmag))

    flag = "POOR_PRECISION" if mean_rms > 5.0 else "OK"

    return PrecisionTrackResult(
        n_nights=n,
        mean_rms_mmag=mean_rms,
        std_rms_mmag=std_rms,
        worst_night_idx=worst_idx,
        best_night_idx=best_idx,
        flag=flag,
    )


def format_precision_track(result: PrecisionTrackResult) -> str:
    lines = [
        "## Photometric Precision Tracker",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Nights | {result.n_nights} |",
        f"| Mean RMS (mmag) | {result.mean_rms_mmag:.3f} |",
        f"| Std RMS (mmag) | {result.std_rms_mmag:.3f} |",
        f"| Best night index | {result.best_night_idx} |",
        f"| Worst night index | {result.worst_night_idx} |",
        f"| Flag | {result.flag} |",
    ]
    return "\n".join(lines)


def _cli() -> int:
    parser = argparse.ArgumentParser(
        description="Track night-to-night photometric precision."
    )
    parser.add_argument(
        "--rms-values",
        required=True,
        help="JSON array of per-night RMS values in mmag, e.g. '[2.1, 3.4, 1.8]'",
    )
    args = parser.parse_args()

    values = json.loads(args.rms_values)
    result = track_precision(values)
    print(format_precision_track(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
