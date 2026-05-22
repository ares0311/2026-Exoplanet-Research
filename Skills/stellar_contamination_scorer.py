"""Compute a composite contamination score for a transit candidate.

Combines the TESS CROWDSAP contamination ratio, the number of nearby
contaminants, and a dilution factor estimate into a single [0, 1] score.
Higher scores indicate more contaminated apertures where the observed depth
is less reliable.

Public API
----------
ContaminationResult(crowdsap_score, neighbour_score, dilution_score,
                    composite_score, is_contaminated, flag)
score_contamination(contratio, n_contaminants, neighbour_flux_ratios, *,
                    contamination_threshold) -> ContaminationResult
format_contamination_result(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContaminationResult:
    crowdsap_score: float | None    # 1 - CROWDSAP; high = more contaminated
    neighbour_score: float | None   # saturates at 5 neighbours
    dilution_score: float | None    # 1 - dilution_factor (from flux ratios)
    composite_score: float
    is_contaminated: bool
    flag: str  # "OK" | "INSUFFICIENT" | "INVALID"


def score_contamination(
    contratio: float | None,
    n_contaminants: int | None,
    neighbour_flux_ratios: list[float] | None = None,
    *,
    contamination_threshold: float = 0.3,
) -> ContaminationResult:
    """Compute a composite aperture contamination score.

    Args:
        contratio: TESS CROWDSAP contamination ratio in [0, 1]; 0 = clean.
        n_contaminants: Number of nearby sources inside the aperture.
        neighbour_flux_ratios: List of F_neighbour / F_target flux ratios.
        contamination_threshold: Composite score above which ``is_contaminated``
            is True.

    Returns:
        :class:`ContaminationResult`.
    """
    scores: list[float] = []
    weights: list[float] = []

    cs: float | None = None
    ns: float | None = None
    ds: float | None = None

    if contratio is not None:
        if not (0.0 <= contratio <= 1.0):
            return ContaminationResult(None, None, None, 0.0, False, "INVALID")
        cs = contratio  # already a contamination fraction
        scores.append(cs)
        weights.append(0.5)

    if n_contaminants is not None:
        if n_contaminants < 0:
            return ContaminationResult(None, None, None, 0.0, False, "INVALID")
        ns = min(n_contaminants / 5.0, 1.0)
        scores.append(ns)
        weights.append(0.2)

    if neighbour_flux_ratios is not None:
        if any(r < 0 for r in neighbour_flux_ratios):
            return ContaminationResult(None, None, None, 0.0, False, "INVALID")
        if neighbour_flux_ratios:
            total_contam = sum(neighbour_flux_ratios)
            f_total = 1.0 + total_contam
            dilution = 1.0 / f_total
            ds = 1.0 - dilution
        else:
            ds = 0.0
        scores.append(ds)
        weights.append(0.3)

    if not scores:
        return ContaminationResult(None, None, None, 0.0, False, "INSUFFICIENT")

    sw = sum(weights)
    composite = sum(s * w for s, w in zip(scores, weights, strict=False)) / sw

    return ContaminationResult(
        crowdsap_score=round(cs, 4) if cs is not None else None,
        neighbour_score=round(ns, 4) if ns is not None else None,
        dilution_score=round(ds, 4) if ds is not None else None,
        composite_score=round(composite, 4),
        is_contaminated=composite >= contamination_threshold,
        flag="OK",
    )


def format_contamination_result(result: ContaminationResult) -> str:
    """Format contamination result as Markdown."""
    lines = [
        "## Stellar Contamination Score",
        "",
        f"- CROWDSAP score: {result.crowdsap_score}",
        f"- Neighbour score: {result.neighbour_score}",
        f"- Dilution score: {result.dilution_score}",
        f"- **Composite score: {result.composite_score:.4f}**",
        f"- Contaminated: {'Yes' if result.is_contaminated else 'No'}",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="stellar_contamination_scorer",
        description="Compute composite aperture contamination score.",
    )
    parser.add_argument("--contratio", type=float, default=None)
    parser.add_argument("--n-contaminants", type=int, default=None)
    parser.add_argument("--ratios", nargs="*", type=float, default=None)
    args = parser.parse_args(argv)

    result = score_contamination(args.contratio, args.n_contaminants, args.ratios)
    print(format_contamination_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
