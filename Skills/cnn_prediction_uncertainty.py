"""Monte Carlo dropout uncertainty estimation for CNN predictions.

Runs inference N times with dropout active to estimate prediction mean and
standard deviation — a proxy for model confidence. High std → uncertain
prediction → good candidate for active learning annotation.

Public API
----------
UncertaintyResult(tic_id, period_days, mean_score, std_score, n_samples,
                  is_uncertain, flag)
estimate_uncertainty(tic_id, period_days, snippet, *, model_fn, n_samples,
                     uncertainty_threshold) -> UncertaintyResult
batch_uncertainty(rows, *, model_fn, n_samples, uncertainty_threshold)
    -> list[UncertaintyResult]
format_uncertainty_report(results) -> str
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class UncertaintyResult:
    tic_id: int
    period_days: float
    mean_score: float
    std_score: float
    n_samples: int
    is_uncertain: bool   # std_score > uncertainty_threshold
    flag: str  # "OK" | "INVALID"


def estimate_uncertainty(
    tic_id: int,
    period_days: float,
    snippet: list[float],
    *,
    model_fn: Callable[[list[float]], float],
    n_samples: int = 30,
    uncertainty_threshold: float = 0.1,
) -> UncertaintyResult:
    """Estimate prediction uncertainty via repeated stochastic forward passes.

    Args:
        tic_id: TIC identifier.
        period_days: Folding period.
        snippet: Phase-folded, normalized flux array.
        model_fn: Callable that takes a snippet and returns a score in [0, 1].
            Must produce stochastic output (dropout active) on each call.
        n_samples: Number of forward passes.
        uncertainty_threshold: std_score above this → is_uncertain=True.

    Returns:
        UncertaintyResult with mean, std, and uncertainty flag.
    """
    if not snippet or n_samples < 1:
        return UncertaintyResult(
            tic_id=tic_id, period_days=period_days, mean_score=0.0,
            std_score=0.0, n_samples=0, is_uncertain=False, flag="INVALID",
        )

    try:
        scores = [model_fn(snippet) for _ in range(n_samples)]
    except Exception:
        return UncertaintyResult(
            tic_id=tic_id, period_days=period_days, mean_score=0.0,
            std_score=0.0, n_samples=0, is_uncertain=False, flag="INVALID",
        )

    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    std = math.sqrt(variance)
    is_uncertain = std > uncertainty_threshold

    return UncertaintyResult(
        tic_id=tic_id,
        period_days=period_days,
        mean_score=mean,
        std_score=std,
        n_samples=n_samples,
        is_uncertain=is_uncertain,
        flag="OK",
    )


def batch_uncertainty(
    rows: list[dict],
    *,
    model_fn: Callable[[list[float]], float],
    n_samples: int = 30,
    uncertainty_threshold: float = 0.1,
) -> list[UncertaintyResult]:
    """Run uncertainty estimation on a batch of snippet rows.

    Args:
        rows: List of dicts with keys ``tic_id``, ``period_days``, ``snippet``.
        model_fn: Stochastic inference function (snippet -> score).
        n_samples: Number of MC samples per snippet.
        uncertainty_threshold: Threshold for is_uncertain flag.

    Returns:
        List of UncertaintyResult, one per input row.
    """
    results = []
    for row in rows:
        try:
            tic_id = int(row["tic_id"])
            period = float(row.get("period_days") or 0.0)
            snippet = list(row["snippet"])
        except (KeyError, TypeError, ValueError):
            results.append(UncertaintyResult(
                tic_id=0, period_days=0.0, mean_score=0.0, std_score=0.0,
                n_samples=0, is_uncertain=False, flag="INVALID",
            ))
            continue
        results.append(estimate_uncertainty(
            tic_id, period, snippet,
            model_fn=model_fn,
            n_samples=n_samples,
            uncertainty_threshold=uncertainty_threshold,
        ))
    return results


def format_uncertainty_report(results: list[UncertaintyResult]) -> str:
    """Format a Markdown uncertainty report.

    Args:
        results: List of UncertaintyResult from batch_uncertainty.

    Returns:
        Markdown string.
    """
    n_ok = sum(1 for r in results if r.flag == "OK")
    n_uncertain = sum(1 for r in results if r.is_uncertain)
    lines = [
        "## MC Dropout Uncertainty Report\n",
        f"Total: {len(results)} | Valid: {n_ok} | Uncertain: {n_uncertain}\n",
        "",
        "| TIC ID | Period (d) | Mean Score | Std | Uncertain? |",
        "|---|---|---|---|---|",
    ]
    for r in results[:20]:   # cap table at 20 rows
        unc = "Yes" if r.is_uncertain else "No"
        lines.append(
            f"| {r.tic_id} | {r.period_days:.4f} | {r.mean_score:.4f}"
            f" | {r.std_score:.4f} | {unc} |"
        )
    if len(results) > 20:
        lines.append(f"| … | … | … | … | ({len(results) - 20} more) |")
    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Estimate CNN prediction uncertainty via MC dropout."
    )
    parser.add_argument(
        "--n-samples", type=int, default=30,
        help="Number of MC forward passes.",
    )
    parser.add_argument("--threshold", type=float, default=0.1)
    args = parser.parse_args(argv)

    import random
    rng = random.Random(0)

    def _demo_model(snippet: list[float]) -> float:
        return 0.5 + rng.gauss(0, 0.1)

    demo_rows = [
        {"tic_id": 100, "period_days": 5.0, "snippet": [1.0] * 64},
        {"tic_id": 200, "period_days": 10.0, "snippet": [0.99] * 64},
    ]
    results = batch_uncertainty(demo_rows, model_fn=_demo_model,
                                n_samples=args.n_samples, uncertainty_threshold=args.threshold)
    print(format_uncertainty_report(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
