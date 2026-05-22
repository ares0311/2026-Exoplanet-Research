"""Score the potential habitability of a planet candidate.

Combines equilibrium temperature, planet radius, and host stellar type into
a composite habitability score [0, 1].  Inspired by the Earth Similarity
Index (ESI) concept but simplified for rapid vetting use.  A high score
indicates Earth-like conditions; a low score indicates hot, cold, or
giant-planet conditions.

Public API
----------
HabitabilityResult(teq_score, radius_score, stellar_score, composite_score,
                   classification, flag)
score_habitability(teq_k, radius_earth, stellar_teff_k, *,
                   teq_weight, radius_weight, stellar_weight) -> HabitabilityResult
format_habitability_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class HabitabilityResult:
    teq_score: float | None      # [0, 1] — 1.0 at 255 K (Earth-like)
    radius_score: float | None   # [0, 1] — 1.0 at 1 R_Earth
    stellar_score: float | None  # [0, 1] — based on stellar type
    composite_score: float       # weighted combination
    classification: str          # "potentially_habitable" | "marginal" | "uninhabitable"
    flag: str  # "OK" | "INSUFFICIENT" | "INVALID"


def _teq_score(teq_k: float) -> float:
    """Score based on equilibrium temperature (peak at 255 K)."""
    if teq_k <= 0:
        return 0.0
    # Gaussian centred at 255 K, sigma = 100 K
    return math.exp(-0.5 * ((teq_k - 255.0) / 100.0) ** 2)


def _radius_score(radius_earth: float) -> float:
    """Score based on planet radius (peak at 1 R_Earth)."""
    if radius_earth <= 0:
        return 0.0
    # Penalise large planets (> 2 R_Earth → likely non-rocky)
    if radius_earth <= 1.5:
        return math.exp(-0.5 * ((radius_earth - 1.0) / 0.5) ** 2)
    # Rapid decay above 1.5 R_Earth
    decay = max(0.0, 1.0 - (radius_earth - 1.5) / 1.0)
    return math.exp(-0.5 * ((radius_earth - 1.0) / 0.5) ** 2) * decay


def _stellar_score(teff_k: float) -> float:
    """Score based on stellar type (K-dwarfs preferred, G ok, M marginal)."""
    if teff_k <= 0:
        return 0.5  # unknown — neutral
    if 3500 <= teff_k <= 5200:
        return 1.0  # K-dwarf sweet spot
    if 2700 <= teff_k < 3500:
        return 0.6  # M-dwarfs: habitable but flare risk
    if 5200 < teff_k <= 6000:
        return 0.8  # G-dwarfs
    if 6000 < teff_k <= 7000:
        return 0.4  # F-dwarfs: higher UV
    return 0.0     # very hot or very cool


def score_habitability(
    teq_k: float | None,
    radius_earth: float | None,
    stellar_teff_k: float | None,
    *,
    teq_weight: float = 0.50,
    radius_weight: float = 0.30,
    stellar_weight: float = 0.20,
) -> HabitabilityResult:
    """Score the habitability of a planet candidate.

    Args:
        teq_k: Equilibrium temperature in Kelvin (None if unknown).
        radius_earth: Planet radius in Earth radii (None if unknown).
        stellar_teff_k: Host star effective temperature in Kelvin (None if unknown).
        teq_weight: Weight for temperature sub-score.
        radius_weight: Weight for radius sub-score.
        stellar_weight: Weight for stellar-type sub-score.

    Returns:
        :class:`HabitabilityResult`.
    """
    total_w = teq_weight + radius_weight + stellar_weight
    if total_w <= 0:
        return HabitabilityResult(None, None, None, 0.0, "uninhabitable", "INVALID")

    ts: float | None = None
    rs: float | None = None
    ss: float | None = None
    used_w = 0.0
    score = 0.0

    if teq_k is not None:
        ts = _teq_score(teq_k)
        score += teq_weight * ts
        used_w += teq_weight

    if radius_earth is not None:
        rs = _radius_score(radius_earth)
        score += radius_weight * rs
        used_w += radius_weight

    if stellar_teff_k is not None:
        ss = _stellar_score(stellar_teff_k)
        score += stellar_weight * ss
        used_w += stellar_weight

    if used_w < 1e-9:
        return HabitabilityResult(None, None, None, 0.0, "uninhabitable", "INSUFFICIENT")

    # Renormalise to the actual weights used
    composite = score / used_w

    if composite >= 0.6:
        cls = "potentially_habitable"
    elif composite >= 0.3:
        cls = "marginal"
    else:
        cls = "uninhabitable"

    flag = "OK"

    return HabitabilityResult(
        teq_score=round(ts, 4) if ts is not None else None,
        radius_score=round(rs, 4) if rs is not None else None,
        stellar_score=round(ss, 4) if ss is not None else None,
        composite_score=round(composite, 4),
        classification=cls,
        flag=flag,
    )


def format_habitability_result(result: HabitabilityResult) -> str:
    """Format habitability result as Markdown."""
    lines = [
        "## Planet Habitability Score",
        "",
        f"- T_eq score: {result.teq_score}",
        f"- Radius score: {result.radius_score}",
        f"- Stellar-type score: {result.stellar_score}",
        f"- **Composite score: {result.composite_score:.4f}**",
        f"- Classification: {result.classification}",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="planet_habitability_scorer",
        description="Score habitability from T_eq, radius, and stellar type.",
    )
    parser.add_argument("--teq-k", type=float, default=None)
    parser.add_argument("--radius-earth", type=float, default=None)
    parser.add_argument("--stellar-teff", type=float, default=None)
    args = parser.parse_args(argv)

    result = score_habitability(args.teq_k, args.radius_earth, args.stellar_teff)
    print(format_habitability_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
