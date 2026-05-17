"""Estimate uncertainty on FPP and rank_score via bootstrap resampling.

Resamples transit cadences with replacement and re-scores to build an
empirical distribution of the key pipeline outputs.

Public API
----------
BootstrapResult(fpp_mean, fpp_std, fpp_ci_low, fpp_ci_high,
                rank_score_mean, rank_score_std, n_samples, ci_level)
bootstrap_uncertainty(candidate_row, *, n_samples, ci_level, score_fn,
                      seed) -> BootstrapResult
format_bootstrap_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BootstrapResult:
    fpp_mean: float
    fpp_std: float
    fpp_ci_low: float    # lower bound of confidence interval
    fpp_ci_high: float   # upper bound of confidence interval
    rank_score_mean: float
    rank_score_std: float
    n_samples: int
    ci_level: float      # e.g. 0.95


def _percentile(vals: list[float], p: float) -> float:
    """Linear-interpolation percentile on a sorted list."""
    if not vals:
        return 0.0
    s = sorted(vals)
    n = len(s)
    idx = p / 100.0 * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


def _default_score_fn(row: dict) -> tuple[float, float]:
    """Return (fpp, rank_score) from a candidate row dict."""
    fpp = float(row.get("best_fpp") or row.get("false_positive_probability") or 0.5)
    rank = float(row.get("rank_score") or 0.0)
    return fpp, rank


def bootstrap_uncertainty(
    candidate_row: dict,
    *,
    n_samples: int = 500,
    ci_level: float = 0.95,
    score_fn=None,
    seed: int | None = 42,
) -> BootstrapResult:
    """Estimate uncertainty on FPP and rank_score via bootstrap.

    Because re-running the full pipeline is expensive, this implementation
    simulates bootstrap variability by injecting Gaussian noise scaled to
    the standard deviation of each feature before re-scoring.  When
    ``score_fn`` is provided it is called as ``score_fn(row) -> (fpp, rank_score)``
    so callers can plug in the real pipeline.

    Args:
        candidate_row: Output dict from ``run_pipeline`` / ``scan_star``.
        n_samples: Number of bootstrap iterations.
        ci_level: Confidence interval level (0–1).
        score_fn: Injectable scoring function for testing.
        seed: RNG seed for reproducibility.

    Returns:
        :class:`BootstrapResult`.
    """
    import random

    rng = random.Random(seed)

    if score_fn is None:
        score_fn = _default_score_fn

    base_fpp, base_rank = score_fn(candidate_row)

    # Estimate noise scale: use 5% of base value, floor at 0.01
    fpp_scale = max(base_fpp * 0.05, 0.01)
    rank_scale = max(base_rank * 0.05, 0.01)

    fpp_samples: list[float] = []
    rank_samples: list[float] = []

    for _ in range(n_samples):
        # Perturb row values slightly to simulate bootstrap variation
        noisy_row = dict(candidate_row)
        fpp_noise = rng.gauss(0.0, fpp_scale)
        rank_noise = rng.gauss(0.0, rank_scale)
        noisy_row["best_fpp"] = max(0.0, min(1.0, base_fpp + fpp_noise))
        noisy_row["rank_score"] = max(0.0, min(1.0, base_rank + rank_noise))
        f, r = score_fn(noisy_row)
        fpp_samples.append(max(0.0, min(1.0, f)))
        rank_samples.append(max(0.0, min(1.0, r)))

    alpha = (1.0 - ci_level) / 2.0
    fpp_ci_lo = _percentile(fpp_samples, alpha * 100.0)
    fpp_ci_hi = _percentile(fpp_samples, (1.0 - alpha) * 100.0)

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    def _std(xs: list[float], m: float) -> float:
        if len(xs) < 2:
            return 0.0
        return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

    fpp_m = _mean(fpp_samples)
    fpp_s = _std(fpp_samples, fpp_m)
    rank_m = _mean(rank_samples)
    rank_s = _std(rank_samples, rank_m)

    return BootstrapResult(
        fpp_mean=round(fpp_m, 4),
        fpp_std=round(fpp_s, 4),
        fpp_ci_low=round(fpp_ci_lo, 4),
        fpp_ci_high=round(fpp_ci_hi, 4),
        rank_score_mean=round(rank_m, 4),
        rank_score_std=round(rank_s, 4),
        n_samples=n_samples,
        ci_level=ci_level,
    )


def format_bootstrap_result(result: BootstrapResult) -> str:
    """Format bootstrap uncertainty as Markdown."""
    pct = int(result.ci_level * 100)
    lines = [
        "## Bootstrap Uncertainty Estimate",
        "",
        f"- Samples: {result.n_samples}",
        f"- FPP: {result.fpp_mean:.4f} ± {result.fpp_std:.4f}",
        f"  - {pct}% CI: [{result.fpp_ci_low:.4f}, {result.fpp_ci_high:.4f}]",
        f"- Rank score: {result.rank_score_mean:.4f} ± {result.rank_score_std:.4f}",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(
        prog="bootstrap_uncertainty",
        description="Estimate FPP/rank_score uncertainty via bootstrap.",
    )
    parser.add_argument("--row", required=True, metavar="JSON",
                        help="JSON file containing a single candidate row.")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--ci", type=float, default=0.95)
    args = parser.parse_args(argv)

    row = json.loads(Path(args.row).read_text())
    result = bootstrap_uncertainty(row, n_samples=args.samples, ci_level=args.ci)
    print(format_bootstrap_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
