"""
Analyzes a sequence of (n_samples, train_score, val_score) tuples from a learning curve.

Public API:
    LearningCurveResult  -- frozen dataclass holding curve diagnostics
    analyze_learning_curve(curve_points) -> LearningCurveResult
    format_learning_curve(result) -> str
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class LearningCurveResult:
    n_points: int
    final_train_score: float
    final_val_score: float
    gap: float
    converged: bool
    flag: str


def analyze_learning_curve(curve_points: list[dict]) -> LearningCurveResult:
    n = len(curve_points)
    if n == 0:
        return LearningCurveResult(
            n_points=0,
            final_train_score=0.0,
            final_val_score=0.0,
            gap=0.0,
            converged=False,
            flag="NO_DATA",
        )

    last = curve_points[-1]
    final_train = float(last["train_score"])
    final_val = float(last["val_score"])
    gap = final_train - final_val

    if n >= 2:
        prev_val = float(curve_points[-2]["val_score"])
        converged = abs(final_val - prev_val) < 0.01
    else:
        converged = True

    if final_val < 0.7:
        flag = "UNDERFITTING"
    elif gap > 0.1:
        flag = "OVERFITTING"
    else:
        flag = "OK"

    return LearningCurveResult(
        n_points=n,
        final_train_score=final_train,
        final_val_score=final_val,
        gap=gap,
        converged=converged,
        flag=flag,
    )


def format_learning_curve(result: LearningCurveResult) -> str:
    lines = [
        "## Learning Curve Analysis",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Points in curve | {result.n_points} |",
        f"| Final train score | {result.final_train_score:.4f} |",
        f"| Final val score | {result.final_val_score:.4f} |",
        f"| Train-val gap | {result.gap:.4f} |",
        f"| Converged | {result.converged} |",
        f"| Flag | {result.flag} |",
    ]
    return "\n".join(lines)


def _cli() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze a learning curve from a JSON file of curve points."
    )
    parser.add_argument(
        "curve_file",
        help="JSON file with list of {n_samples, train_score, val_score} dicts",
    )
    args = parser.parse_args()

    with open(args.curve_file) as fh:
        points = json.load(fh)

    result = analyze_learning_curve(points)
    print(format_learning_curve(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
