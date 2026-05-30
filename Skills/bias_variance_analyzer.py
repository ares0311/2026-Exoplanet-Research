"""
Bias-variance analyzer for model predictions from repeated CV folds.

Public API:
    BiasVarianceResult  -- frozen dataclass holding decomposition components
    analyze_bias_variance(true_label, fold_predictions) -> BiasVarianceResult
    format_bias_variance(result) -> str
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class BiasVarianceResult:
    mean_pred: float
    bias_sq: float
    variance: float
    total_error: float
    flag: str


def analyze_bias_variance(
    true_label: float,
    fold_predictions: list[float],
) -> BiasVarianceResult:
    n = len(fold_predictions)
    if n == 0:
        return BiasVarianceResult(
            mean_pred=0.0,
            bias_sq=0.0,
            variance=0.0,
            total_error=0.0,
            flag="NO_DATA",
        )

    mean_pred = sum(fold_predictions) / n
    bias_sq = (mean_pred - true_label) ** 2

    variance = sum((p - mean_pred) ** 2 for p in fold_predictions) / (n - 1) if n > 1 else 0.0

    total_error = bias_sq + variance
    flag = "HIGH_VARIANCE" if variance > 0.1 else "OK"

    return BiasVarianceResult(
        mean_pred=mean_pred,
        bias_sq=bias_sq,
        variance=variance,
        total_error=total_error,
        flag=flag,
    )


def format_bias_variance(result: BiasVarianceResult) -> str:
    lines = [
        "## Bias-Variance Analysis",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Mean prediction | {result.mean_pred:.4f} |",
        f"| Bias² | {result.bias_sq:.4f} |",
        f"| Variance | {result.variance:.4f} |",
        f"| Total error (Bias²+Var) | {result.total_error:.4f} |",
        f"| Flag | {result.flag} |",
    ]
    return "\n".join(lines)


def _cli() -> int:
    parser = argparse.ArgumentParser(
        description="Compute bias² and variance from CV fold predictions."
    )
    parser.add_argument("true_label", type=float, help="Ground-truth label value")
    parser.add_argument(
        "--predictions",
        required=True,
        help="JSON array of fold predictions, e.g. '[0.8, 0.9, 0.7]'",
    )
    args = parser.parse_args()

    preds = json.loads(args.predictions)
    result = analyze_bias_variance(args.true_label, preds)
    print(format_bias_variance(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
