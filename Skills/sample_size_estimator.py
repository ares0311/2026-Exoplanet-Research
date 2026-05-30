"""Estimate minimum labeled sample size for a target ML classifier performance.

Uses learning-curve power-law extrapolation: AUC(n) = AUC_max - A * n^(-b).

Public API:
    SampleSizeResult  -- frozen dataclass
    estimate_sample_size(known_points, target_auc, *, max_auc, exponent) -> SampleSizeResult
    format_sample_size_result(result) -> str
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SampleSizeResult:
    target_auc: float
    estimated_n: int
    current_best_auc: float
    achievable: bool
    flag: str


def estimate_sample_size(
    known_points: list[tuple[int, float]],
    target_auc: float,
    *,
    max_auc: float = 0.99,
    exponent: float = 0.5,
) -> SampleSizeResult:
    """
    known_points: list of (n_samples, auc) pairs from empirical learning curve.
    target_auc: desired AUC to achieve.
    max_auc: theoretical ceiling.
    exponent: power-law decay exponent b (default 0.5).
    """
    if not (0.0 < target_auc <= 1.0):
        return SampleSizeResult(
            target_auc=target_auc, estimated_n=0, current_best_auc=0.0,
            achievable=False, flag="INVALID_TARGET_AUC",
        )
    if target_auc >= max_auc:
        return SampleSizeResult(
            target_auc=target_auc, estimated_n=0, current_best_auc=0.0,
            achievable=False, flag="TARGET_EXCEEDS_MAX_AUC",
        )
    if len(known_points) < 1:
        return SampleSizeResult(
            target_auc=target_auc, estimated_n=0, current_best_auc=0.0,
            achievable=False, flag="INSUFFICIENT_DATA",
        )
    valid = [(n, a) for n, a in known_points if n > 0 and 0.0 < a <= 1.0]
    if not valid:
        return SampleSizeResult(
            target_auc=target_auc, estimated_n=0, current_best_auc=0.0,
            achievable=False, flag="INSUFFICIENT_DATA",
        )
    current_best = max(a for _, a in valid)
    if current_best >= target_auc:
        n_achieved = min(n for n, a in valid if a >= target_auc)
        return SampleSizeResult(
            target_auc=target_auc, estimated_n=n_achieved, current_best_auc=current_best,
            achievable=True, flag="ALREADY_ACHIEVED",
        )
    # Fit A from last known point: auc = max_auc - A * n^(-b) => A = (max_auc - auc) * n^b
    last_n, last_auc = max(valid, key=lambda x: x[0])
    a_coef = (max_auc - last_auc) * (last_n ** exponent)
    # Solve: target = max_auc - A * n^(-b) => n = (A / (max_auc - target))^(1/b)
    gap = max_auc - target_auc
    if gap <= 0 or a_coef <= 0:
        return SampleSizeResult(
            target_auc=target_auc, estimated_n=0, current_best_auc=current_best,
            achievable=False, flag="EXTRAPOLATION_FAILED",
        )
    estimated_n = int(math.ceil((a_coef / gap) ** (1.0 / exponent)))
    return SampleSizeResult(
        target_auc=target_auc,
        estimated_n=estimated_n,
        current_best_auc=current_best,
        achievable=True,
        flag="OK",
    )


def format_sample_size_result(result: SampleSizeResult) -> str:
    lines = [
        "## Sample Size Estimate",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Target AUC | {result.target_auc:.4f} |",
        f"| Estimated N | {result.estimated_n} |",
        f"| Current Best AUC | {result.current_best_auc:.4f} |",
        f"| Achievable | {result.achievable} |",
        f"| Flag | {result.flag} |",
    ]
    return "\n".join(lines)


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Estimate required labeled sample size.")
    parser.add_argument("target_auc", type=float)
    parser.add_argument("--point", nargs=2, action="append", metavar=("N", "AUC"),
                        type=float, dest="points", default=[])
    parser.add_argument("--max-auc", type=float, default=0.99)
    parser.add_argument("--exponent", type=float, default=0.5)
    args = parser.parse_args()
    known = [(int(p[0]), p[1]) for p in args.points]
    result = estimate_sample_size(
        known, args.target_auc, max_auc=args.max_auc, exponent=args.exponent,
    )
    print(format_sample_size_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
