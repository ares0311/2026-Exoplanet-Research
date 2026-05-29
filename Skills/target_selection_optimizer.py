"""Optimize target selection for maximum science return per telescope hour.

Scores targets by expected science value accounting for observability,
stellar quality, and pipeline confidence.

Public API
----------
TargetScore(tic_id, science_value, observability, stellar_quality,
            pipeline_confidence, composite_score, rank, flag)
optimize_target_selection(targets, *, top_n, min_composite_score,
                          science_weight, obs_weight, stellar_weight,
                          pipeline_weight) -> list[TargetScore]
format_selection_result(results) -> str
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass


@dataclass(frozen=True)
class TargetScore:
    tic_id: int | None
    science_value: float         # [0,1]
    observability: float         # [0,1]
    stellar_quality: float       # [0,1]
    pipeline_confidence: float   # [0,1]
    composite_score: float       # [0,1]
    rank: int
    flag: str  # "RECOMMENDED" | "MARGINAL" | "SKIP"


def _safe_float(v: object, default: float = 0.5) -> float:
    with contextlib.suppress(TypeError, ValueError):
        return max(0.0, min(1.0, float(v)))  # type: ignore[arg-type]
    return default


def _science_value(row: dict) -> float:
    """Estimate science value from FPP, novelty, and pathway."""
    fpp = _safe_float(
        row.get("false_positive_probability")
        or (row.get("scores") or {}).get("false_positive_probability"),
        default=0.5,
    )
    novelty = _safe_float(
        row.get("novelty_score")
        or (row.get("scores") or {}).get("novelty_score"),
        default=0.5,
    )
    pathway = str(row.get("pathway") or "")
    pathway_bonus = 0.2 if "tfop_ready" in pathway else 0.0
    return max(0.0, min(1.0, (1.0 - fpp) * 0.6 + novelty * 0.3 + pathway_bonus))


def _stellar_quality(row: dict) -> float:
    """Estimate stellar quality from magnitude and Teff proximity to solar."""
    tmag = _safe_float(row.get("tmag") or row.get("stellar_tmag"), default=12.0)
    # Prefer Tmag 8–12 (bright but unsaturated)
    tmag_score = max(0.0, 1.0 - abs(tmag - 10.0) / 5.0)
    teff = _safe_float(row.get("stellar_teff_k") or row.get("teff"), default=5500.0)
    # Normalise: clip 3000–8000, prefer 4000–7000
    teff_norm = max(0.0, min(1.0, 1.0 - abs(teff - 5500.0) / 3000.0))
    return (tmag_score * 0.5 + teff_norm * 0.5)


def _pipeline_confidence(row: dict) -> float:
    """Estimate pipeline confidence from detection_confidence and provenance."""
    dc = _safe_float(
        row.get("detection_confidence")
        or (row.get("scores") or {}).get("detection_confidence"),
        default=0.5,
    )
    prov = _safe_float(row.get("provenance_score"), default=0.5)
    return dc * 0.6 + prov * 0.4


def optimize_target_selection(
    targets: list[dict],
    *,
    top_n: int | None = None,
    min_composite_score: float = 0.0,
    science_weight: float = 0.40,
    obs_weight: float = 0.25,
    stellar_weight: float = 0.20,
    pipeline_weight: float = 0.15,
) -> list[TargetScore]:
    """Rank targets by composite science-return score.

    Args:
        targets: List of pipeline output dicts.
        top_n: Return only this many top targets; None returns all.
        min_composite_score: Filter out targets below this score.
        science_weight: Weight for science value component.
        obs_weight: Weight for observability component.
        stellar_weight: Weight for stellar quality component.
        pipeline_weight: Weight for pipeline confidence component.

    Returns:
        List of TargetScore sorted descending by composite_score.
    """
    scored: list[tuple[float, dict, float, float, float, float]] = []
    for row in targets:
        sv = _science_value(row)
        obs = _safe_float(row.get("observability_score"), default=0.7)
        sq = _stellar_quality(row)
        pc = _pipeline_confidence(row)
        composite = (science_weight * sv + obs_weight * obs
                     + stellar_weight * sq + pipeline_weight * pc)
        composite = min(1.0, max(0.0, composite))
        if composite >= min_composite_score:
            scored.append((composite, row, sv, obs, sq, pc))

    scored.sort(key=lambda x: x[0], reverse=True)

    results: list[TargetScore] = []
    for rank, (composite, row, sv, obs, sq, pc) in enumerate(scored, 1):
        tic_id: int | None = None
        with contextlib.suppress(TypeError, ValueError):
            raw = row.get("tic_id")
            if raw is not None:
                tic_id = int(raw)

        if composite >= 0.65:
            flag = "RECOMMENDED"
        elif composite >= 0.45:
            flag = "MARGINAL"
        else:
            flag = "SKIP"

        results.append(TargetScore(
            tic_id=tic_id,
            science_value=round(sv, 4),
            observability=round(obs, 4),
            stellar_quality=round(sq, 4),
            pipeline_confidence=round(pc, 4),
            composite_score=round(composite, 4),
            rank=rank,
            flag=flag,
        ))

        if top_n is not None and rank >= top_n:
            break

    return results


def format_selection_result(results: list[TargetScore]) -> str:
    """Format target selection results as Markdown.

    Args:
        results: List of TargetScore from optimize_target_selection.

    Returns:
        Markdown string.
    """
    if not results:
        return "## Target Selection Optimizer\n\n_No targets to rank._"

    n_rec = sum(1 for r in results if r.flag == "RECOMMENDED")
    lines = [
        "## Target Selection Optimizer\n",
        f"**Targets evaluated**: {len(results)} | Recommended: {n_rec}\n",
        "",
        "| Rank | TIC ID | Science | Obs | Star | Pipeline | Composite | Flag |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        tic_str = str(r.tic_id) if r.tic_id is not None else "—"
        lines.append(
            f"| {r.rank} | {tic_str} | {r.science_value:.3f} | "
            f"{r.observability:.3f} | {r.stellar_quality:.3f} | "
            f"{r.pipeline_confidence:.3f} | {r.composite_score:.4f} | `{r.flag}` |"
        )
    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Optimise target selection.")
    parser.add_argument("input", help="Candidate JSON file.")
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument("--min-score", type=float, default=0.0)
    args = parser.parse_args(argv)

    from pathlib import Path
    rows = json.loads(Path(args.input).read_text())
    if not isinstance(rows, list):
        rows = [rows]
    results = optimize_target_selection(rows, top_n=args.top_n,
                                        min_composite_score=args.min_score)
    print(format_selection_result(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
