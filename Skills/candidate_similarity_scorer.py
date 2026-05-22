"""Score similarity between two candidate transit signals.

Compares period, depth, duration, and shape to detect near-duplicates or
related signals (e.g. same planet detected in two sectors, or an alias pair).

Public API
----------
SimilarityResult(period_similarity, depth_similarity, duration_similarity,
                 composite_score, is_duplicate, relationship, flag)
score_similarity(cand_a, cand_b, *, period_tol_frac, depth_tol_frac,
                 duration_tol_frac, duplicate_threshold) -> SimilarityResult
format_similarity_result(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SimilarityResult:
    period_similarity: float | None    # 1 - |ΔP|/P, clipped to [0,1]
    depth_similarity: float | None     # 1 - |Δdepth|/depth, clipped [0,1]
    duration_similarity: float | None  # 1 - |Δdur|/dur, clipped [0,1]
    composite_score: float             # weighted mean of available scores
    is_duplicate: bool
    relationship: str  # "duplicate" | "alias" | "harmonic" | "unrelated"
    flag: str  # "OK" | "INVALID"


def _similarity_score(a: float, b: float, tol: float) -> float:
    """1 - relative difference, scaled so tol → 0.5."""
    if a <= 0 and b <= 0:
        return 1.0
    ref = max(abs(a), abs(b), 1e-20)
    diff = abs(a - b) / ref
    return max(0.0, 1.0 - diff / (2.0 * tol))


def score_similarity(
    cand_a: dict,
    cand_b: dict,
    *,
    period_tol_frac: float = 0.02,
    depth_tol_frac: float = 0.20,
    duration_tol_frac: float = 0.20,
    duplicate_threshold: float = 0.80,
) -> SimilarityResult:
    """Score similarity between two candidate signal dicts.

    Each dict should contain keys: ``period_days``, ``depth_ppm`` (optional),
    ``duration_hours`` (optional).

    Args:
        cand_a: First candidate dict.
        cand_b: Second candidate dict.
        period_tol_frac: Fractional period tolerance for similarity.
        depth_tol_frac: Fractional depth tolerance.
        duration_tol_frac: Fractional duration tolerance.
        duplicate_threshold: Composite score above which signals are duplicates.

    Returns:
        :class:`SimilarityResult`.
    """
    pa = cand_a.get("period_days")
    pb = cand_b.get("period_days")
    if pa is None or pb is None or pa <= 0 or pb <= 0:
        return SimilarityResult(None, None, None, 0.0, False, "unrelated", "INVALID")

    p_sim = _similarity_score(pa, pb, period_tol_frac)

    # Check alias relationship
    ratio = pa / pb if pa > pb else pb / pa
    nearest = round(ratio)
    alias_dev = abs(ratio - nearest) / max(nearest, 1) if nearest > 0 else 1.0
    is_alias = nearest in (2, 3, 4) and alias_dev < period_tol_frac * 2

    da = cand_a.get("depth_ppm")
    db = cand_b.get("depth_ppm")
    d_sim = _similarity_score(da, db, depth_tol_frac) if da and db else None

    dura = cand_a.get("duration_hours")
    durb = cand_b.get("duration_hours")
    dur_sim = _similarity_score(dura, durb, duration_tol_frac) if dura and durb else None

    scores = [s for s in [p_sim, d_sim, dur_sim] if s is not None]
    weights = [3.0, 2.0, 1.0][:len(scores)]
    composite = sum(s * w for s, w in zip(scores, weights, strict=False)) / sum(weights)

    if composite >= duplicate_threshold:
        relationship = "duplicate"
        is_dup = True
    elif is_alias:
        relationship = "alias"
        is_dup = False
    elif p_sim >= duplicate_threshold:
        relationship = "harmonic" if nearest > 1 else "duplicate"
        is_dup = nearest == 1
    else:
        relationship = "unrelated"
        is_dup = False

    return SimilarityResult(
        period_similarity=round(p_sim, 4),
        depth_similarity=round(d_sim, 4) if d_sim is not None else None,
        duration_similarity=round(dur_sim, 4) if dur_sim is not None else None,
        composite_score=round(composite, 4),
        is_duplicate=is_dup,
        relationship=relationship,
        flag="OK",
    )


def format_similarity_result(result: SimilarityResult) -> str:
    """Format similarity result as Markdown."""
    lines = [
        "## Candidate Similarity Score",
        "",
        f"- Period similarity: {result.period_similarity}",
        f"- Depth similarity: {result.depth_similarity}",
        f"- Duration similarity: {result.duration_similarity}",
        f"- **Composite score: {result.composite_score:.4f}**",
        f"- Is duplicate: {'Yes' if result.is_duplicate else 'No'}",
        f"- Relationship: {result.relationship}",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="candidate_similarity_scorer",
        description="Score similarity between two candidate signals.",
    )
    parser.add_argument("period_a", type=float)
    parser.add_argument("period_b", type=float)
    args = parser.parse_args(argv)

    a = {"period_days": args.period_a}
    b = {"period_days": args.period_b}
    result = score_similarity(a, b)
    print(format_similarity_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
