"""Assess impact of outliers on transit depth estimates.

Identifies outliers via MAD-based sigma clipping and compares depth
estimates with and without outliers to quantify their impact.

Public API
----------
OutlierImpactResult(n_outliers, outlier_indices, depth_with_ppm,
                    depth_without_ppm, depth_change_pct, flag)
assess_outlier_impact(flux, n_sigma) -> OutlierImpactResult
format_outlier_impact(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OutlierImpactResult:
    n_outliers: int
    outlier_indices: tuple[int, ...]
    depth_with_ppm: float
    depth_without_ppm: float
    depth_change_pct: float
    flag: str  # "OK", "OUTLIER_IMPACT"(change > 20%)


def _median(vals: list[float]) -> float:
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 == 1 else (s[mid - 1] + s[mid]) / 2.0


def assess_outlier_impact(flux: list[float], n_sigma: float = 3.0) -> OutlierImpactResult:
    """Assess outlier impact on transit depth estimate.

    Outliers: |flux - median| > n_sigma * MAD * 1.4826.
    Depth with outliers: (max - min) * 1e6 on original flux.
    Depth without: (max - min) * 1e6 on cleaned flux.

    Args:
        flux: Normalised flux values.
        n_sigma: Sigma clipping threshold.

    Returns:
        OutlierImpactResult.
    """
    if len(flux) < 2:
        return OutlierImpactResult(
            n_outliers=0,
            outlier_indices=(),
            depth_with_ppm=0.0,
            depth_without_ppm=0.0,
            depth_change_pct=0.0,
            flag="OK",
        )

    med = _median(flux)
    abs_dev = [abs(f - med) for f in flux]
    mad = _median(abs_dev)
    sigma = 1.4826 * mad

    outlier_indices: list[int] = []
    if sigma > 0:
        for i, f in enumerate(flux):
            if abs(f - med) > n_sigma * sigma:
                outlier_indices.append(i)

    # Depth with outliers (entire range)
    depth_with = (max(flux) - min(flux)) * 1e6

    # Depth without outliers
    cleaned = [f for i, f in enumerate(flux) if i not in outlier_indices]
    depth_without = (max(cleaned) - min(cleaned)) * 1e6 if len(cleaned) >= 2 else depth_with

    if depth_with > 0:
        depth_change_pct = abs(depth_with - depth_without) / depth_with * 100.0
    else:
        depth_change_pct = 0.0

    flag = "OUTLIER_IMPACT" if depth_change_pct > 20.0 else "OK"

    return OutlierImpactResult(
        n_outliers=len(outlier_indices),
        outlier_indices=tuple(outlier_indices),
        depth_with_ppm=round(depth_with, 2),
        depth_without_ppm=round(depth_without, 2),
        depth_change_pct=round(depth_change_pct, 2),
        flag=flag,
    )


def format_outlier_impact(result: OutlierImpactResult) -> str:
    """Format outlier impact assessment as Markdown.

    Args:
        result: OutlierImpactResult to format.

    Returns:
        Markdown string.
    """
    lines = [
        "## Outlier Impact Assessment",
        "",
        f"- Outliers detected: {result.n_outliers}",
        f"- Depth with outliers: {result.depth_with_ppm:.1f} ppm",
        f"- Depth without outliers: {result.depth_without_ppm:.1f} ppm",
        f"- Depth change: {result.depth_change_pct:.1f}%",
        f"- Status: `{result.flag}`",
    ]
    if result.outlier_indices:
        idx_str = ", ".join(str(i) for i in result.outlier_indices[:10])
        if result.n_outliers > 10:
            idx_str += f", ... (+{result.n_outliers - 10} more)"
        lines.append(f"- Outlier indices: {idx_str}")
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Assess outlier impact on depth.")
    parser.add_argument("flux_json", help="JSON file with list of flux values.")
    parser.add_argument("--sigma", type=float, default=3.0)
    args = parser.parse_args(argv)

    flux = json.loads(Path(args.flux_json).read_text())
    result = assess_outlier_impact(flux, args.sigma)
    print(format_outlier_impact(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
