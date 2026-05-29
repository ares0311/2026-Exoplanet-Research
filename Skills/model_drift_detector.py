"""Detect when model prediction distributions drift from a baseline.

Compares a new batch of predictions against a stored baseline distribution
using mean shift and variance ratio as lightweight drift statistics.

Public API
----------
DriftResult(feature, baseline_mean, current_mean, mean_shift,
            baseline_std, current_std, std_ratio, drift_detected, flag)
BaselineStats(feature, mean, std, n, recorded_at)
detect_drift(current_scores, *, baseline_mean, baseline_std,
             mean_shift_threshold, std_ratio_threshold) -> DriftResult
compute_baseline_stats(scores, *, feature) -> BaselineStats
format_drift_report(results) -> str
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class BaselineStats:
    feature: str
    mean: float
    std: float
    n: int
    recorded_at: float  # Unix timestamp


@dataclass(frozen=True)
class DriftResult:
    feature: str
    baseline_mean: float
    current_mean: float
    mean_shift: float          # |current_mean - baseline_mean|
    baseline_std: float
    current_std: float
    std_ratio: float           # current_std / baseline_std (or inf if baseline_std=0)
    drift_detected: bool
    flag: str  # "OK" | "DRIFT" | "SEVERE_DRIFT" | "INSUFFICIENT_DATA"


def _mean_std(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mu = sum(values) / n
    if n < 2:
        return mu, 0.0
    variance = sum((x - mu) ** 2 for x in values) / (n - 1)
    return mu, math.sqrt(variance)


def compute_baseline_stats(
    scores: list[float],
    *,
    feature: str = "score",
) -> BaselineStats:
    """Compute baseline statistics from a reference score distribution.

    Args:
        scores: Reference score values.
        feature: Name of the feature/score being tracked.

    Returns:
        BaselineStats with mean, std, sample count, and timestamp.
    """
    mu, std = _mean_std(scores)
    return BaselineStats(
        feature=feature,
        mean=round(mu, 6),
        std=round(std, 6),
        n=len(scores),
        recorded_at=time.time(),
    )


def detect_drift(
    current_scores: list[float],
    *,
    baseline_mean: float,
    baseline_std: float,
    feature: str = "score",
    mean_shift_threshold: float = 0.05,
    std_ratio_threshold: float = 1.5,
    min_samples: int = 10,
) -> DriftResult:
    """Detect distribution drift relative to a baseline.

    Args:
        current_scores: New batch of prediction scores.
        baseline_mean: Mean of the reference distribution.
        baseline_std: Std of the reference distribution.
        feature: Name label for this score.
        mean_shift_threshold: Flag drift if |Δmean| > this.
        std_ratio_threshold: Flag drift if current_std/baseline_std > this.
        min_samples: Minimum current samples for a valid comparison.

    Returns:
        DriftResult with shift statistics and drift flag.
    """
    if len(current_scores) < min_samples:
        return DriftResult(
            feature=feature,
            baseline_mean=baseline_mean,
            current_mean=0.0,
            mean_shift=0.0,
            baseline_std=baseline_std,
            current_std=0.0,
            std_ratio=1.0,
            drift_detected=False,
            flag="INSUFFICIENT_DATA",
        )

    curr_mean, curr_std = _mean_std(current_scores)
    mean_shift = abs(curr_mean - baseline_mean)

    if baseline_std > 0:
        std_ratio = curr_std / baseline_std
    elif curr_std > 0:
        std_ratio = float("inf")
    else:
        std_ratio = 1.0

    drift_detected = mean_shift > mean_shift_threshold or std_ratio > std_ratio_threshold

    if not drift_detected:
        flag = "OK"
    elif mean_shift > 2 * mean_shift_threshold or std_ratio > 2 * std_ratio_threshold:
        flag = "SEVERE_DRIFT"
    else:
        flag = "DRIFT"

    return DriftResult(
        feature=feature,
        baseline_mean=round(baseline_mean, 6),
        current_mean=round(curr_mean, 6),
        mean_shift=round(mean_shift, 6),
        baseline_std=round(baseline_std, 6),
        current_std=round(curr_std, 6),
        std_ratio=round(std_ratio, 4) if math.isfinite(std_ratio) else float("inf"),
        drift_detected=drift_detected,
        flag=flag,
    )


def format_drift_report(results: list[DriftResult]) -> str:
    """Format drift detection results as Markdown.

    Args:
        results: List of DriftResult to format.

    Returns:
        Markdown string.
    """
    if not results:
        return "## Model Drift Report\n\n_No drift results._"

    n_drift = sum(1 for r in results if r.drift_detected)
    overall = "DRIFT DETECTED" if n_drift > 0 else "OK"
    lines = [
        "## Model Drift Report\n",
        f"**Overall**: `{overall}` | {n_drift}/{len(results)} features drifted\n",
        "",
        "| Feature | Baseline μ | Current μ | Shift | Baseline σ | Current σ | σ Ratio | Flag |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        std_ratio_str = (f"{r.std_ratio:.3f}" if math.isfinite(r.std_ratio) else "∞")
        lines.append(
            f"| {r.feature} | {r.baseline_mean:.4f} | {r.current_mean:.4f} | "
            f"{r.mean_shift:.4f} | {r.baseline_std:.4f} | {r.current_std:.4f} | "
            f"{std_ratio_str} | `{r.flag}` |"
        )
    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Detect model prediction drift.")
    parser.add_argument("--baseline", required=True, help="Baseline stats JSON.")
    parser.add_argument("--current", required=True, help="Current scores JSON (list of floats).")
    parser.add_argument("--mean-shift-threshold", type=float, default=0.05)
    parser.add_argument("--std-ratio-threshold", type=float, default=1.5)
    args = parser.parse_args(argv)

    from pathlib import Path
    baseline = json.loads(Path(args.baseline).read_text())
    current_scores = json.loads(Path(args.current).read_text())

    result = detect_drift(
        current_scores,
        baseline_mean=baseline["mean"],
        baseline_std=baseline["std"],
        feature=baseline.get("feature", "score"),
        mean_shift_threshold=args.mean_shift_threshold,
        std_ratio_threshold=args.std_ratio_threshold,
    )
    print(format_drift_report([result]))
    return 1 if result.drift_detected else 0


if __name__ == "__main__":
    raise SystemExit(_cli())
