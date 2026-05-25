"""Produce a ranked plain-English explanation of candidate posterior probabilities.

Maps known score and posterior keys to human-readable descriptions,
classifying each contributing factor as supporting or against planet candidacy.

Public API
----------
ScoreExplanationEntry
ScoreExplanationResult
explain_candidate_score(candidate) -> ScoreExplanationResult
format_score_explanation(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoreExplanationEntry:
    feature: str
    value: float | None
    direction: str   # "supports_planet" | "against_planet" | "neutral"
    strength: str    # "strong" | "moderate" | "weak"
    note: str


@dataclass(frozen=True)
class ScoreExplanationResult:
    tic_id: int | None
    period_days: float | None
    planet_posterior: float
    fpp: float
    pathway: str
    top_positive: tuple[ScoreExplanationEntry, ...]
    top_negative: tuple[ScoreExplanationEntry, ...]
    flag: str  # "OK" | "INVALID"


def _strength_order(s: str) -> int:
    return {"strong": 0, "moderate": 1, "weak": 2}.get(s, 3)


def explain_candidate_score(candidate: dict) -> ScoreExplanationResult:
    """Produce a plain-English explanation of candidate score drivers.

    Args:
        candidate: Pipeline output row dict containing 'scores', 'posterior',
                   and optional 'meta', 'tic_id', 'period_days', 'pathway'.

    Returns:
        :class:`ScoreExplanationResult`.
    """
    if not isinstance(candidate, dict):
        return ScoreExplanationResult(
            tic_id=None,
            period_days=None,
            planet_posterior=0.0,
            fpp=1.0,
            pathway="unknown",
            top_positive=(),
            top_negative=(),
            flag="INVALID",
        )

    scores = candidate.get("scores", {}) or {}
    posterior = candidate.get("posterior", {}) or {}

    tic_id = candidate.get("tic_id") or candidate.get("target_id")
    if isinstance(tic_id, str):
        try:
            tic_id = int(tic_id)
        except (ValueError, TypeError):
            tic_id = None

    period_days = candidate.get("period_days")
    pathway = candidate.get("pathway", "unknown") or "unknown"

    fpp = float(scores.get("false_positive_probability", 1.0))
    planet_posterior = float(posterior.get("planet_candidate", 0.0))

    entries: list[ScoreExplanationEntry] = []

    # FPP
    if "false_positive_probability" in scores:
        val = float(scores["false_positive_probability"])
        if val < 0.1:
            entries.append(ScoreExplanationEntry(
                "false_positive_probability", val,
                "supports_planet", "strong",
                f"Very low FPP ({val:.3f}) strongly favours planet candidacy",
            ))
        elif val < 0.3:
            entries.append(ScoreExplanationEntry(
                "false_positive_probability", val,
                "supports_planet", "moderate",
                f"Low FPP ({val:.3f}) supports planet candidacy",
            ))
        elif val < 0.5:
            entries.append(ScoreExplanationEntry(
                "false_positive_probability", val,
                "against_planet", "weak",
                f"Moderate FPP ({val:.3f}) weakly against planet",
            ))
        elif val < 0.7:
            entries.append(ScoreExplanationEntry(
                "false_positive_probability", val,
                "against_planet", "moderate",
                f"High FPP ({val:.3f}) against planet candidacy",
            ))
        else:
            entries.append(ScoreExplanationEntry(
                "false_positive_probability", val,
                "against_planet", "strong",
                f"Very high FPP ({val:.3f}) strongly against planet",
            ))

    # Detection confidence
    if "detection_confidence" in scores:
        val = float(scores["detection_confidence"])
        if val > 0.8:
            entries.append(ScoreExplanationEntry(
                "detection_confidence", val,
                "supports_planet", "moderate",
                f"High detection confidence ({val:.3f})",
            ))
        elif val < 0.4:
            entries.append(ScoreExplanationEntry(
                "detection_confidence", val,
                "against_planet", "weak",
                f"Low detection confidence ({val:.3f})",
            ))

    # EB posterior
    if "eclipsing_binary" in posterior:
        val = float(posterior["eclipsing_binary"])
        if val > 0.3:
            entries.append(ScoreExplanationEntry(
                "posterior_eclipsing_binary", val,
                "against_planet", "strong",
                f"High EB posterior ({val:.3f}) suggests eclipsing binary contamination",
            ))
        elif val > 0.15:
            entries.append(ScoreExplanationEntry(
                "posterior_eclipsing_binary", val,
                "against_planet", "moderate",
                f"Elevated EB posterior ({val:.3f})",
            ))

    # Instrumental artifact posterior
    if "instrumental_artifact" in posterior:
        val = float(posterior["instrumental_artifact"])
        if val > 0.3:
            entries.append(ScoreExplanationEntry(
                "posterior_instrumental_artifact", val,
                "against_planet", "moderate",
                f"Elevated instrumental artifact posterior ({val:.3f})",
            ))
        elif val > 0.15:
            entries.append(ScoreExplanationEntry(
                "posterior_instrumental_artifact", val,
                "against_planet", "weak",
                f"Mild instrumental artifact posterior ({val:.3f})",
            ))

    # Stellar variability posterior
    if "stellar_variability" in posterior:
        val = float(posterior["stellar_variability"])
        if val > 0.2:
            entries.append(ScoreExplanationEntry(
                "posterior_stellar_variability", val,
                "against_planet", "weak",
                f"Elevated stellar variability posterior ({val:.3f})",
            ))

    # Provenance score
    prov = candidate.get("provenance_score")
    if prov is not None:
        val = float(prov)
        if val > 0.8:
            entries.append(ScoreExplanationEntry(
                "provenance_score", val,
                "supports_planet", "weak",
                f"High provenance score ({val:.3f}): good data quality",
            ))
        elif val < 0.5:
            entries.append(ScoreExplanationEntry(
                "provenance_score", val,
                "against_planet", "weak",
                f"Low provenance score ({val:.3f}): reduced data quality",
            ))

    # Pathway
    if pathway == "tfop_ready":
        entries.append(ScoreExplanationEntry(
            "pathway", None,
            "supports_planet", "strong",
            "Pathway is tfop_ready: meets all TFOP submission criteria",
        ))
    elif pathway == "github_only_reproducibility":
        entries.append(ScoreExplanationEntry(
            "pathway", None,
            "against_planet", "moderate",
            "Pathway is github_only_reproducibility: does not meet follow-up criteria",
        ))

    # Planet posterior
    if planet_posterior > 0.6:
        entries.append(ScoreExplanationEntry(
            "posterior_planet_candidate", planet_posterior,
            "supports_planet", "strong",
            f"Planet candidate posterior is high ({planet_posterior:.3f})",
        ))
    elif planet_posterior < 0.2:
        entries.append(ScoreExplanationEntry(
            "posterior_planet_candidate", planet_posterior,
            "against_planet", "strong",
            f"Planet candidate posterior is low ({planet_posterior:.3f})",
        ))

    positive = sorted(
        [e for e in entries if e.direction == "supports_planet"],
        key=lambda e: _strength_order(e.strength),
    )
    negative = sorted(
        [e for e in entries if e.direction == "against_planet"],
        key=lambda e: _strength_order(e.strength),
    )

    return ScoreExplanationResult(
        tic_id=tic_id,
        period_days=period_days,
        planet_posterior=planet_posterior,
        fpp=fpp,
        pathway=pathway,
        top_positive=tuple(positive),
        top_negative=tuple(negative),
        flag="OK",
    )


def format_score_explanation(result: ScoreExplanationResult) -> str:
    """Format score explanation result as Markdown."""
    tic_str = str(result.tic_id) if result.tic_id is not None else "N/A"
    period_str = f"{result.period_days:.4f} d" if result.period_days is not None else "N/A"
    lines = [
        "## Candidate Score Explainer",
        "",
        f"- TIC: {tic_str}",
        f"- Period: {period_str}",
        f"- Planet posterior: {result.planet_posterior:.4f}",
        f"- FPP: {result.fpp:.4f}",
        f"- Pathway: {result.pathway}",
        "",
        "### Supporting Evidence (planet)",
    ]
    if result.top_positive:
        for e in result.top_positive:
            val_str = f"{e.value:.4f}" if e.value is not None else "—"
            lines.append(f"  - [{e.strength}] {e.feature} = {val_str}: {e.note}")
    else:
        lines.append("  - None")

    lines += ["", "### Evidence Against Planet"]
    if result.top_negative:
        for e in result.top_negative:
            val_str = f"{e.value:.4f}" if e.value is not None else "—"
            lines.append(f"  - [{e.strength}] {e.feature} = {val_str}: {e.note}")
    else:
        lines.append("  - None")

    lines += ["", f"**Flag: {result.flag}**"]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="candidate_score_explainer",
        description="Explain what drives a candidate's posterior probabilities.",
    )
    parser.add_argument("candidate_json", help="JSON file or inline JSON string.")
    args = parser.parse_args(argv)

    try:
        import pathlib
        p = pathlib.Path(args.candidate_json)
        candidate = json.loads(p.read_text()) if p.exists() else json.loads(args.candidate_json)
    except Exception as exc:
        print(f"Error: {exc}")
        return 1

    result = explain_candidate_score(candidate)
    print(format_score_explanation(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
