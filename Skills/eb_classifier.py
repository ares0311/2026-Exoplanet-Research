"""Classify a transit candidate as planet or eclipsing binary using simple heuristics.

Combines depth, odd/even asymmetry, secondary eclipse depth, and stellar density
ratio to produce a posterior probability and classification string.

Public API
----------
EBClassifierResult(depth_ppm, secondary_depth_ppm, odd_even_sigma,
                   rho_ratio, eb_probability, classification, reasons, flag)
classify_eb(depth_ppm, *, secondary_depth_ppm, odd_even_sigma,
            rho_ratio, secondary_threshold_ppm, odd_even_threshold_sigma,
            rho_threshold) -> EBClassifierResult
format_eb_result(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EBClassifierResult:
    depth_ppm: float
    secondary_depth_ppm: float | None
    odd_even_sigma: float | None
    rho_ratio: float | None          # ρ_photometric / ρ_spectroscopic
    eb_probability: float            # heuristic score in [0, 1]
    classification: str              # "planet_candidate" | "likely_eb" | "possible_eb"
    reasons: tuple[str, ...]
    flag: str                        # "OK" | "INSUFFICIENT"


def classify_eb(
    depth_ppm: float,
    *,
    secondary_depth_ppm: float | None = None,
    odd_even_sigma: float | None = None,
    rho_ratio: float | None = None,
    secondary_threshold_ppm: float = 500.0,
    odd_even_threshold_sigma: float = 3.0,
    rho_threshold: float = 2.0,
) -> EBClassifierResult:
    """Classify a transit candidate as planet or EB.

    Each diagnostic adds to an EB-evidence score in [0, 1]:
    - Depth > 50,000 ppm → strong EB indicator (+0.50)
    - Secondary eclipse depth > threshold → secondary detected (+0.30)
    - Odd/even sigma > threshold → depth asymmetry (+0.30)
    - Stellar density ratio > threshold → density inconsistency (+0.20)

    Args:
        depth_ppm: Transit depth in ppm.
        secondary_depth_ppm: Detected secondary eclipse depth (None = not measured).
        odd_even_sigma: Sigma asymmetry between odd and even transits (None = not measured).
        rho_ratio: Photometric/spectroscopic density ratio (None = not available).
        secondary_threshold_ppm: Secondary depth threshold for EB flag.
        odd_even_threshold_sigma: Odd/even asymmetry threshold in sigma.
        rho_threshold: Density ratio threshold for EB flag.

    Returns:
        :class:`EBClassifierResult`.
    """
    score = 0.0
    reasons: list[str] = []

    if depth_ppm <= 0:
        return EBClassifierResult(
            depth_ppm, secondary_depth_ppm, odd_even_sigma, rho_ratio,
            0.0, "planet_candidate", (), "INSUFFICIENT",
        )

    # Very deep transit (> 5% = 50,000 ppm) is a strong EB indicator
    if depth_ppm > 50000:
        score += 0.50
        reasons.append(f"very_deep_transit ({depth_ppm:.0f} ppm)")

    # Secondary eclipse detected
    if secondary_depth_ppm is not None and secondary_depth_ppm > secondary_threshold_ppm:
        score += 0.30
        reasons.append(f"secondary_eclipse ({secondary_depth_ppm:.0f} ppm)")

    # Odd/even depth asymmetry
    if odd_even_sigma is not None and odd_even_sigma > odd_even_threshold_sigma:
        score += 0.30
        reasons.append(f"odd_even_asymmetry ({odd_even_sigma:.1f}σ)")

    # Stellar density inconsistency
    if rho_ratio is not None and rho_ratio > rho_threshold:
        score += 0.20
        reasons.append(f"density_inconsistency (ratio={rho_ratio:.2f})")

    score = min(score, 1.0)

    if score >= 0.60:
        classification = "likely_eb"
    elif score >= 0.25:
        classification = "possible_eb"
    else:
        classification = "planet_candidate"

    return EBClassifierResult(
        depth_ppm=depth_ppm,
        secondary_depth_ppm=secondary_depth_ppm,
        odd_even_sigma=odd_even_sigma,
        rho_ratio=rho_ratio,
        eb_probability=round(score, 4),
        classification=classification,
        reasons=tuple(reasons),
        flag="OK",
    )


def format_eb_result(result: EBClassifierResult) -> str:
    """Format EB classification result as Markdown."""
    lines = [
        "## Eclipsing Binary Classifier",
        "",
        f"- Transit depth: {result.depth_ppm:.1f} ppm",
    ]
    if result.secondary_depth_ppm is not None:
        lines.append(f"- Secondary depth: {result.secondary_depth_ppm:.1f} ppm")
    if result.odd_even_sigma is not None:
        lines.append(f"- Odd/even asymmetry: {result.odd_even_sigma:.2f}σ")
    if result.rho_ratio is not None:
        lines.append(f"- Density ratio: {result.rho_ratio:.3f}")
    lines += [
        f"- EB probability: {result.eb_probability:.4f}",
        f"- Classification: **{result.classification}**",
        f"- Flag: **{result.flag}**",
    ]
    if result.reasons:
        lines.append("- EB evidence:")
        for r in result.reasons:
            lines.append(f"  - {r}")
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="eb_classifier",
        description="Classify transit candidate as planet or eclipsing binary.",
    )
    parser.add_argument("depth_ppm", type=float)
    parser.add_argument("--secondary-depth-ppm", type=float, default=None)
    parser.add_argument("--odd-even-sigma", type=float, default=None)
    parser.add_argument("--rho-ratio", type=float, default=None)
    parser.add_argument("--secondary-threshold-ppm", type=float, default=500.0)
    parser.add_argument("--odd-even-threshold-sigma", type=float, default=3.0)
    parser.add_argument("--rho-threshold", type=float, default=2.0)
    args = parser.parse_args(argv)

    result = classify_eb(
        args.depth_ppm,
        secondary_depth_ppm=args.secondary_depth_ppm,
        odd_even_sigma=args.odd_even_sigma,
        rho_ratio=args.rho_ratio,
        secondary_threshold_ppm=args.secondary_threshold_ppm,
        odd_even_threshold_sigma=args.odd_even_threshold_sigma,
        rho_threshold=args.rho_threshold,
    )
    print(format_eb_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
