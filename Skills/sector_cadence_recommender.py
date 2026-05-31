"""Recommend TESS cadence mode based on target brightness and science goals."""
from __future__ import annotations

import argparse
from dataclasses import dataclass

_CADENCE_MODES = ("20s", "2min", "10min")

# Tmag brightness limits for each cadence mode
# Based on TESS saturation: ~7.5 mag at 2 min, ~4.5 mag at 20 s
_SATURATION_LIMITS: dict[str, float] = {
    "20s": 4.5,
    "2min": 7.5,
    "10min": 9.0,
}

# Science goals that require shorter cadence
_HIGH_CADENCE_GOALS = frozenset({
    "asteroseismology", "flares", "rapid_transients", "pulsations",
    "short_period_eb", "granulation",
})


@dataclass(frozen=True)
class CadenceRecommendationResult:
    tmag: float
    science_goal: str
    recommended_cadence: str
    reason: str
    saturated_modes: tuple[str, ...]
    flag: str


def recommend_cadence(
    tmag: float,
    science_goal: str = "transit",
    prefer_short: bool = False,
) -> CadenceRecommendationResult:
    """
    Recommend the optimal TESS cadence mode for a target.

    Rules:
    - Reject cadence modes where target saturates the detector.
    - Short-cadence science goals (asteroseismology, flares, etc.) prefer 20s or 2min.
    - Faint targets (Tmag > 13) benefit from 2min for noise budget; 20s rarely allocated.
    - Default for transit science: 2min if available, else 10min.
    """
    import math

    if not math.isfinite(tmag):
        return CadenceRecommendationResult(
            tmag=tmag, science_goal=science_goal,
            recommended_cadence="", reason="",
            saturated_modes=(), flag="INVALID_TMAG",
        )

    saturated = tuple(m for m, lim in _SATURATION_LIMITS.items() if tmag < lim)
    available = [m for m in _CADENCE_MODES if m not in saturated]

    if not available:
        return CadenceRecommendationResult(
            tmag=tmag, science_goal=science_goal,
            recommended_cadence="", reason="All cadence modes saturate at this brightness.",
            saturated_modes=saturated, flag="ALL_MODES_SATURATED",
        )

    goal = science_goal.lower().strip()
    requires_short = goal in _HIGH_CADENCE_GOALS or prefer_short

    # Select recommendation
    if requires_short:
        # Pick shortest available
        recommended = available[0]
        reason = (
            f"Science goal '{science_goal}' requires short cadence; "
            f"shortest available: {recommended}."
        )
    elif tmag > 13.0:
        # Faint star: 2min recommended over 20s (no benefit, saves quota)
        recommended = "2min" if "2min" in available else available[0]
        reason = f"Target is faint (Tmag={tmag:.1f}); 2min recommended to preserve 20s quota."
    elif tmag < 9.0 and "2min" in available:
        recommended = "2min"
        reason = f"Bright target (Tmag={tmag:.1f}); 2min provides good SNR without saturation."
    elif "2min" in available:
        recommended = "2min"
        reason = "Standard transit science; 2min cadence recommended."
    else:
        recommended = available[0]
        reason = f"2min not available (saturated); using {recommended}."

    return CadenceRecommendationResult(
        tmag=tmag,
        science_goal=science_goal,
        recommended_cadence=recommended,
        reason=reason,
        saturated_modes=saturated,
        flag="OK",
    )


def format_cadence_recommendation(r: CadenceRecommendationResult) -> str:
    sat_str = ", ".join(r.saturated_modes) if r.saturated_modes else "none"
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| T mag | {r.tmag:.2f} |\n"
        f"| Science goal | {r.science_goal} |\n"
        f"| Recommended cadence | {r.recommended_cadence} |\n"
        f"| Saturated modes | {sat_str} |\n"
        f"| Reason | {r.reason} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Recommend TESS cadence mode.")
    p.add_argument("tmag", type=float)
    p.add_argument("--science-goal", default="transit")
    p.add_argument("--prefer-short", action="store_true")
    args = p.parse_args()
    r = recommend_cadence(args.tmag, args.science_goal, args.prefer_short)
    print(format_cadence_recommendation(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
