"""Score the reliability of a Threshold Crossing Event (TCE).

Combines multiple detection statistics — MES (Multiple Event Statistic),
number of transits, SES/MES ratio, and fit residuals — into a composite
reliability score in [0, 1].

Public API
----------
TCEReliabilityResult(mes, n_transits, ses_mes_ratio, fit_residual_ppm,
                     composite_score, is_reliable, flag)
score_tce_reliability(mes, n_transits, *, ses_mes_ratio, fit_residual_ppm,
                       mes_threshold, n_transit_min,
                       reliability_threshold) -> TCEReliabilityResult
format_reliability_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TCEReliabilityResult:
    mes: float | None
    n_transits: int | None
    ses_mes_ratio: float | None    # max(SES) / MES — high = single-event dominated
    fit_residual_ppm: float | None
    composite_score: float
    is_reliable: bool
    flag: str  # "OK" | "INSUFFICIENT" | "INVALID"


def score_tce_reliability(
    mes: float | None,
    n_transits: int | None,
    *,
    ses_mes_ratio: float | None = None,
    fit_residual_ppm: float | None = None,
    mes_threshold: float = 7.1,
    n_transit_min: int = 3,
    reliability_threshold: float = 0.6,
) -> TCEReliabilityResult:
    """Compute a composite TCE reliability score.

    Args:
        mes: Multiple Event Statistic (combined transit SNR).
        n_transits: Number of individual transits detected.
        ses_mes_ratio: Ratio of max single-event statistic to MES.
            Values close to 1.0 indicate a single strong event dominates.
        fit_residual_ppm: RMS of fit residuals in ppm.
        mes_threshold: MES above which the signal is considered significant.
        n_transit_min: Minimum number of transits for a reliable detection.
        reliability_threshold: Composite score above which is reliable.

    Returns:
        :class:`TCEReliabilityResult`.
    """
    if mes is not None and mes < 0:
        return TCEReliabilityResult(mes, n_transits, ses_mes_ratio,
                                    fit_residual_ppm, 0.0, False, "INVALID")

    scores: list[float] = []
    weights: list[float] = []

    # MES score: sigmoid-like ramp from 0 at MES=0 to 1 at MES=15
    mes_score: float | None = None
    if mes is not None:
        mes_score = min(1.0, max(0.0, (mes - mes_threshold) / (15.0 - mes_threshold)))
        scores.append(mes_score)
        weights.append(0.40)

    # Transit count score: ramp from 0 at n=1 to 1 at n=10
    n_score: float | None = None
    if n_transits is not None:
        n_score = min(1.0, max(0.0, (n_transits - 1) / 9.0))
        scores.append(n_score)
        weights.append(0.30)

    # SES/MES score: low ratio = good; high = dominated by single event
    ses_score: float | None = None
    if ses_mes_ratio is not None:
        # Score 1 if ratio <= 1/sqrt(n_min), 0 if ratio = 1
        ideal = 1.0 / math.sqrt(max(n_transit_min, 1))
        ses_score = max(0.0, min(1.0, 1.0 - (ses_mes_ratio - ideal) / (1.0 - ideal)))
        scores.append(ses_score)
        weights.append(0.20)

    # Fit residual score: 1 if residuals < 100 ppm, 0 if > 5000 ppm
    res_score: float | None = None
    if fit_residual_ppm is not None:
        res_score = max(0.0, min(1.0, 1.0 - (fit_residual_ppm - 100.0) / 4900.0))
        scores.append(res_score)
        weights.append(0.10)

    if not scores:
        return TCEReliabilityResult(mes, n_transits, ses_mes_ratio,
                                    fit_residual_ppm, 0.0, False, "INSUFFICIENT")

    sw = sum(weights)
    composite = sum(s * w for s, w in zip(scores, weights, strict=False)) / sw

    # Hard gate: must pass MES and n_transits if provided
    if mes is not None and mes < mes_threshold:
        composite = min(composite, 0.4)
    if n_transits is not None and n_transits < n_transit_min:
        composite = min(composite, 0.3)

    return TCEReliabilityResult(
        mes=mes,
        n_transits=n_transits,
        ses_mes_ratio=ses_mes_ratio,
        fit_residual_ppm=fit_residual_ppm,
        composite_score=round(composite, 4),
        is_reliable=composite >= reliability_threshold,
        flag="OK",
    )


def format_reliability_result(result: TCEReliabilityResult) -> str:
    """Format TCE reliability result as Markdown."""
    lines = [
        "## TCE Reliability Score",
        "",
        f"- MES: {result.mes}",
        f"- N transits: {result.n_transits}",
        f"- SES/MES ratio: {result.ses_mes_ratio}",
        f"- Fit residual: {result.fit_residual_ppm} ppm",
        f"- **Composite score: {result.composite_score:.4f}**",
        f"- Reliable: {'Yes' if result.is_reliable else 'No'}",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="tce_reliability_scorer",
        description="Score TCE reliability from detection statistics.",
    )
    parser.add_argument("--mes", type=float, default=None)
    parser.add_argument("--n-transits", type=int, default=None)
    parser.add_argument("--ses-mes-ratio", type=float, default=None)
    args = parser.parse_args(argv)

    result = score_tce_reliability(args.mes, args.n_transits, ses_mes_ratio=args.ses_mes_ratio)
    print(format_reliability_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
