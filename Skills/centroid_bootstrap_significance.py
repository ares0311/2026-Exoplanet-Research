"""Assess centroid offset significance via bootstrap resampling.

Pools in-transit and out-of-transit centroid positions, then repeatedly
re-partitions the pool to build a null distribution of centroid offsets.
The observed offset is compared to this null distribution to compute a
significance in units of bootstrap standard deviations.

Public API
----------
CentroidBootstrapResult(observed_offset, bootstrap_mean, bootstrap_std,
                        significance, is_significant, flag)
bootstrap_centroid_significance(in_transit_centroids, oot_centroids, *,
                                n_bootstrap, significance_threshold, seed)
    -> CentroidBootstrapResult
format_centroid_bootstrap(result) -> str
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class CentroidBootstrapResult:
    observed_offset: float
    bootstrap_mean: float
    bootstrap_std: float
    significance: float
    is_significant: bool
    flag: str  # "OK" | "SIGNIFICANT" | "INSUFFICIENT_DATA"


def _mean(values: list[float]) -> float:
    """Return the arithmetic mean of a non-empty list."""
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    """Return the population standard deviation of a list."""
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    variance = sum((v - mu) ** 2 for v in values) / len(values)
    return variance ** 0.5


def _lcg_shuffle_indices(n: int, seed: int) -> list[int]:
    """Return a shuffled list of indices 0..n-1 using a simple LCG."""
    state = seed
    indices = list(range(n))
    for i in range(n - 1, 0, -1):
        state = (state * 1664525 + 1013904223) % (2 ** 32)
        j = state % (i + 1)
        indices[i], indices[j] = indices[j], indices[i]
    return indices


def bootstrap_centroid_significance(
    in_transit_centroids: list[float],
    oot_centroids: list[float],
    *,
    n_bootstrap: int = 1000,
    significance_threshold: float = 3.0,
    seed: int = 42,
) -> CentroidBootstrapResult:
    """Assess centroid offset significance via bootstrap permutation.

    Parameters
    ----------
    in_transit_centroids:
        Centroid positions measured during transits.
    oot_centroids:
        Centroid positions measured out of transit.
    n_bootstrap:
        Number of bootstrap permutations.
    significance_threshold:
        Number of bootstrap sigma above which the offset is flagged.
    seed:
        Integer seed for the LCG random number generator.

    Returns
    -------
    CentroidBootstrapResult
    """
    if len(in_transit_centroids) < 2 or len(oot_centroids) < 2:
        return CentroidBootstrapResult(
            observed_offset=0.0,
            bootstrap_mean=0.0,
            bootstrap_std=0.0,
            significance=0.0,
            is_significant=False,
            flag="INSUFFICIENT_DATA",
        )

    n_in = len(in_transit_centroids)
    observed_offset = abs(_mean(in_transit_centroids) - _mean(oot_centroids))

    pool = list(in_transit_centroids) + list(oot_centroids)
    n_pool = len(pool)

    # LCG state — use a mutable container so the nested helper can update it
    lcg_state = [seed]

    def next_rand() -> float:
        lcg_state[0] = (lcg_state[0] * 1664525 + 1013904223) % (2 ** 32)
        return lcg_state[0] / (2 ** 32)

    def shuffle_pool() -> list[float]:
        indices = list(range(n_pool))
        for i in range(n_pool - 1, 0, -1):
            j = int(next_rand() * (i + 1))
            indices[i], indices[j] = indices[j], indices[i]
        return [pool[k] for k in indices]

    bootstrap_offsets: list[float] = []
    for _ in range(n_bootstrap):
        shuffled = shuffle_pool()
        fake_in = shuffled[:n_in]
        fake_oot = shuffled[n_in:]
        bootstrap_offsets.append(abs(_mean(fake_in) - _mean(fake_oot)))

    bootstrap_mean = _mean(bootstrap_offsets)
    bootstrap_std = _std(bootstrap_offsets)

    if bootstrap_std > 0.0:
        significance = (observed_offset - bootstrap_mean) / bootstrap_std
    else:
        significance = 0.0

    is_significant = significance > significance_threshold

    flag = "SIGNIFICANT" if is_significant else "OK"

    return CentroidBootstrapResult(
        observed_offset=observed_offset,
        bootstrap_mean=bootstrap_mean,
        bootstrap_std=bootstrap_std,
        significance=significance,
        is_significant=is_significant,
        flag=flag,
    )


def format_centroid_bootstrap(result: CentroidBootstrapResult) -> str:
    """Return a Markdown table summarising the centroid bootstrap result."""
    lines = [
        "| Metric | Value |",
        "| --- | --- |",
        f"| Observed Offset | {result.observed_offset:.6f} |",
        f"| Bootstrap Mean | {result.bootstrap_mean:.6f} |",
        f"| Bootstrap Std | {result.bootstrap_std:.6f} |",
        f"| Significance (sigma) | {result.significance:.2f} |",
        f"| Is Significant | {result.is_significant} |",
        f"| Flag | {result.flag} |",
    ]
    return "\n".join(lines)


def _cli() -> int:
    parser = argparse.ArgumentParser(
        description="Assess centroid offset significance via bootstrap permutation."
    )
    parser.add_argument(
        "--in-transit",
        nargs="+",
        type=float,
        required=True,
        metavar="VALUE",
        help="In-transit centroid positions.",
    )
    parser.add_argument(
        "--oot",
        nargs="+",
        type=float,
        required=True,
        metavar="VALUE",
        help="Out-of-transit centroid positions.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap permutations (default: 1000).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=3.0,
        help="Significance threshold in sigma (default: 3.0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="LCG random seed (default: 42).",
    )
    args = parser.parse_args()

    result = bootstrap_centroid_significance(
        args.in_transit,
        args.oot,
        n_bootstrap=args.n_bootstrap,
        significance_threshold=args.threshold,
        seed=args.seed,
    )
    print(format_centroid_bootstrap(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
