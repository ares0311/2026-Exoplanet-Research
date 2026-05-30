"""
model_uncertainty_estimator.py

Public API:
    UncertaintyResult        -- frozen dataclass holding ensemble uncertainty metrics
    estimate_model_uncertainty(predictions) -> UncertaintyResult
    format_uncertainty(result) -> str
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class UncertaintyResult:
    n_models: int
    mean_pred: float
    std_pred: float
    iqr: float
    uncertainty_level: str
    flag: str


def estimate_model_uncertainty(predictions: list[float]) -> UncertaintyResult:
    n = len(predictions)
    if n == 0:
        return UncertaintyResult(
            n_models=0,
            mean_pred=0.0,
            std_pred=0.0,
            iqr=0.0,
            uncertainty_level="LOW",
            flag="NO_DATA",
        )

    mean_pred = sum(predictions) / n

    if n > 1:
        std_pred = (sum((p - mean_pred) ** 2 for p in predictions) / (n - 1)) ** 0.5
    else:
        std_pred = 0.0

    sorted_preds = sorted(predictions)
    # Q1 and Q3 via linear interpolation on the sorted list
    q1 = _percentile(sorted_preds, 25.0)
    q3 = _percentile(sorted_preds, 75.0)
    iqr = q3 - q1

    if std_pred < 0.05:
        uncertainty_level = "LOW"
    elif std_pred < 0.15:
        uncertainty_level = "MEDIUM"
    else:
        uncertainty_level = "HIGH"

    flag = "HIGH_UNCERTAINTY" if std_pred >= 0.15 else "OK"

    return UncertaintyResult(
        n_models=n,
        mean_pred=mean_pred,
        std_pred=std_pred,
        iqr=iqr,
        uncertainty_level=uncertainty_level,
        flag=flag,
    )


def _percentile(sorted_values: list[float], pct: float) -> float:
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    # Linear interpolation: index position for given percentile
    pos = pct / 100.0 * (n - 1)
    low = int(pos)
    high = low + 1
    if high >= n:
        return sorted_values[n - 1]
    frac = pos - low
    return sorted_values[low] + frac * (sorted_values[high] - sorted_values[low])


def format_uncertainty(result: UncertaintyResult) -> str:
    lines = [
        "## Model Uncertainty Estimate",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Models in ensemble | {result.n_models} |",
        f"| Mean prediction | {result.mean_pred:.4f} |",
        f"| Std dev | {result.std_pred:.4f} |",
        f"| IQR (Q3−Q1) | {result.iqr:.4f} |",
        f"| Uncertainty level | {result.uncertainty_level} |",
        f"| Flag | {result.flag} |",
    ]
    return "\n".join(lines)


def _cli() -> int:
    parser = argparse.ArgumentParser(
        description="Estimate prediction uncertainty from an ensemble of model outputs."
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="JSON array of model predictions, e.g. '[0.8, 0.75, 0.9]'",
    )
    args = parser.parse_args()

    try:
        preds = json.loads(args.predictions)
    except json.JSONDecodeError as exc:
        print(f"Error: invalid JSON for --predictions: {exc}")
        return 1

    result = estimate_model_uncertainty(preds)
    print(format_uncertainty(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
