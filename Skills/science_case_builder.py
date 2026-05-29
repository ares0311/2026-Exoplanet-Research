"""Build a structured science case document for a transit candidate.

Assembles candidate parameters, scoring summary, and observational context
into a Markdown science case suitable for inclusion in follow-up proposals.

Public API
----------
ScienceCase(tic_id, period_days, depth_ppm, duration_hours,
            planet_radius_rearth, host_star_summary, scoring_summary,
            pathway, sections, flag)
build_science_case(row, *, planet_radius_rearth, host_star_summary,
                   extra_notes) -> ScienceCase
format_science_case(sc) -> str
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass


@dataclass(frozen=True)
class ScienceCase:
    tic_id: int | None
    period_days: float | None
    depth_ppm: float | None
    duration_hours: float | None
    planet_radius_rearth: float | None
    host_star_summary: str
    scoring_summary: str
    pathway: str
    sections: tuple[tuple[str, str], ...]  # (heading, body) pairs
    flag: str  # "OK" | "INCOMPLETE"


def _safe_float(v: object) -> float | None:
    with contextlib.suppress(TypeError, ValueError):
        return float(v)  # type: ignore[arg-type]
    return None


def _safe_int(v: object) -> int | None:
    with contextlib.suppress(TypeError, ValueError):
        return int(v)  # type: ignore[arg-type]
    return None


def build_science_case(
    row: dict,
    *,
    planet_radius_rearth: float | None = None,
    host_star_summary: str = "",
    extra_notes: str = "",
) -> ScienceCase:
    """Build a science case from a pipeline output row.

    Args:
        row: Pipeline output dict with keys period_days, depth_ppm, etc.
        planet_radius_rearth: Estimated planet radius in Earth radii.
        host_star_summary: Free-text description of the host star.
        extra_notes: Additional notes for the science case.

    Returns:
        ScienceCase with populated sections.
    """
    tic_id = _safe_int(row.get("tic_id"))
    period = _safe_float(row.get("period_days"))
    depth = _safe_float(row.get("depth_ppm"))
    duration = _safe_float(row.get("duration_hours"))
    scores = row.get("scores") or {}
    fpp = _safe_float(scores.get("false_positive_probability")
                      or row.get("false_positive_probability"))
    planet_post = _safe_float(
        (row.get("posterior") or {}).get("planet_candidate")
        or scores.get("planet_candidate")
    )
    pathway = str(row.get("pathway") or "unknown")

    # Scoring summary
    score_parts = []
    if fpp is not None:
        score_parts.append(f"FPP={fpp:.3f}")
    if planet_post is not None:
        score_parts.append(f"P(planet)={planet_post:.3f}")
    scoring_summary = ", ".join(score_parts) if score_parts else "not available"

    sections: list[tuple[str, str]] = []

    # Overview
    parts: list[str] = []
    if tic_id:
        parts.append(f"TIC {tic_id}")
    if period:
        parts.append(f"P = {period:.4f} d")
    if depth:
        parts.append(f"depth = {depth:.0f} ppm")
    if duration:
        parts.append(f"duration = {duration:.2f} h")
    sections.append(("Overview", " | ".join(parts) if parts else "Parameters unavailable."))

    # Host star
    sections.append(("Host Star", host_star_summary if host_star_summary
                     else "Stellar parameters not provided."))

    # Planet candidate parameters
    planet_parts: list[str] = []
    if planet_radius_rearth:
        planet_parts.append(f"Estimated radius: {planet_radius_rearth:.2f} R⊕")
    if period:
        planet_parts.append(f"Orbital period: {period:.4f} days")
    if depth:
        planet_parts.append(f"Transit depth: {depth:.0f} ppm ({depth / 10000:.2f}%)")
    sections.append(("Planet Candidate Parameters",
                     "\n".join(planet_parts) if planet_parts else "Not estimated."))

    # Vetting summary
    sections.append(("Vetting Summary", f"Scoring: {scoring_summary}\nPathway: `{pathway}`"))

    # Science motivation
    motivation_parts = [
        "This candidate signal requires ground-based follow-up to rule out false-positive "
        "scenarios including eclipsing binaries and background star contamination.",
    ]
    if fpp is not None and fpp < 0.10:
        motivation_parts.append(
            f"The low FPP ({fpp:.3f}) supports prioritisation for spectroscopic follow-up."
        )
    sections.append(("Science Motivation", " ".join(motivation_parts)))

    if extra_notes:
        sections.append(("Additional Notes", extra_notes))

    # Determine completeness
    has_core = period is not None and depth is not None
    flag = "OK" if has_core else "INCOMPLETE"

    return ScienceCase(
        tic_id=tic_id,
        period_days=period,
        depth_ppm=depth,
        duration_hours=duration,
        planet_radius_rearth=planet_radius_rearth,
        host_star_summary=host_star_summary,
        scoring_summary=scoring_summary,
        pathway=pathway,
        sections=tuple(sections),
        flag=flag,
    )


def format_science_case(sc: ScienceCase) -> str:
    """Format science case as a Markdown document.

    Args:
        sc: ScienceCase to format.

    Returns:
        Markdown string.
    """
    tic_str = str(sc.tic_id) if sc.tic_id is not None else "Unknown"
    lines = [
        f"# Science Case — TIC {tic_str}\n",
        f"**Pathway**: `{sc.pathway}` | **Status**: `{sc.flag}`\n",
    ]
    for heading, body in sc.sections:
        lines.append(f"\n## {heading}\n")
        lines.append(body)
    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Build science case for a candidate.")
    parser.add_argument("input", help="Candidate JSON file.")
    parser.add_argument("--radius", type=float, default=None)
    parser.add_argument("--star", default="", help="Host star description.")
    parser.add_argument("--notes", default="")
    args = parser.parse_args(argv)

    from pathlib import Path
    row = json.loads(Path(args.input).read_text())
    if isinstance(row, list):
        row = row[0]
    sc = build_science_case(row, planet_radius_rearth=args.radius,
                            host_star_summary=args.star, extra_notes=args.notes)
    print(format_science_case(sc))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
