"""Score and rank a list of candidates for follow-up priority.

Computes a composite priority score from FPP, detection confidence,
pathway, and provenance score, then ranks and labels each candidate.

Public API
----------
FollowUpEntry(tic_id, candidate_id, period_days, fpp, detection_confidence,
              pathway, provenance_score, priority_score, priority_rank, recommendation)
PrioritizerResult(n_candidates, n_urgent, n_high, entries, flag)
prioritize_followup(candidates, *, fpp_skip_threshold, fpp_urgent_threshold) -> PrioritizerResult
format_followup_priorities(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass

_PATHWAY_BONUS: dict[str, float] = {
    "tfop_ready": 1.0,
    "planet_hunters_discussion": 0.6,
    "kepler_archive_candidate": 0.7,
    "github_only_reproducibility": 0.2,
    "paper_or_preprint_candidate": 0.9,
    "known_object_annotation": 0.0,
}


@dataclass(frozen=True)
class FollowUpEntry:
    tic_id: int | None
    candidate_id: str
    period_days: float | None
    fpp: float | None
    detection_confidence: float | None
    pathway: str | None
    provenance_score: float | None
    priority_score: float
    priority_rank: int
    recommendation: str  # "urgent" | "high" | "medium" | "low" | "skip"


@dataclass(frozen=True)
class PrioritizerResult:
    n_candidates: int
    n_urgent: int
    n_high: int
    entries: tuple[FollowUpEntry, ...]
    flag: str  # "OK" | "EMPTY" | "INVALID"


def _extract_fpp(row: dict) -> float | None:
    if "scores" in row and isinstance(row["scores"], dict):
        v = row["scores"].get("false_positive_probability")
        if v is not None:
            return float(v)
    for key in ("fpp", "false_positive_probability"):
        v = row.get(key)
        if v is not None:
            return float(v)
    return None


def _extract_dc(row: dict) -> float | None:
    if "scores" in row and isinstance(row["scores"], dict):
        v = row["scores"].get("detection_confidence")
        if v is not None:
            return float(v)
    v = row.get("detection_confidence")
    if v is not None:
        return float(v)
    return None


def _extract_provenance(row: dict) -> float | None:
    v = row.get("provenance_score")
    if v is not None:
        return float(v)
    if "meta" in row and isinstance(row["meta"], dict):
        v = row["meta"].get("provenance_score")
        if v is not None:
            return float(v)
    return None


def _compute_priority(
    fpp: float | None,
    dc: float | None,
    pathway: str | None,
    provenance: float | None,
) -> float:
    """Compute priority score in [0, 1]."""
    fpp_term = (1.0 - fpp) if fpp is not None else 0.5
    dc_term = dc if dc is not None else 0.5
    pathway_bonus = _PATHWAY_BONUS.get(pathway or "", 0.3)
    prov_term = provenance if provenance is not None else 0.5
    return (
        0.40 * fpp_term
        + 0.30 * dc_term
        + 0.15 * pathway_bonus
        + 0.15 * prov_term
    )


def _recommend(
    priority_score: float,
    fpp: float | None,
    pathway: str | None,
    fpp_skip_threshold: float,
    fpp_urgent_threshold: float,
) -> str:
    if fpp is not None and fpp >= fpp_skip_threshold:
        return "skip"
    if fpp is not None and fpp < fpp_urgent_threshold and pathway == "tfop_ready":
        return "urgent"
    if priority_score >= 0.75:
        return "high"
    if priority_score >= 0.50:
        return "medium"
    return "low"


def prioritize_followup(
    candidates: list[dict],
    *,
    fpp_skip_threshold: float = 0.70,
    fpp_urgent_threshold: float = 0.10,
) -> PrioritizerResult:
    """Score and rank candidates for follow-up.

    Args:
        candidates: List of candidate result dicts from the pipeline.
        fpp_skip_threshold: FPP above this → "skip".
        fpp_urgent_threshold: FPP below this AND tfop_ready → "urgent".

    Returns:
        :class:`PrioritizerResult`.
    """
    if not isinstance(candidates, list):
        return PrioritizerResult(0, 0, 0, (), "INVALID")
    if not candidates:
        return PrioritizerResult(0, 0, 0, (), "EMPTY")

    scored: list[tuple[float, dict]] = []
    for row in candidates:
        if not isinstance(row, dict):
            continue
        fpp = _extract_fpp(row)
        dc = _extract_dc(row)
        pathway = row.get("pathway") or row.get("submission_pathway")
        provenance = _extract_provenance(row)
        ps = _compute_priority(fpp, dc, pathway, provenance)
        scored.append((ps, row))

    if not scored:
        return PrioritizerResult(0, 0, 0, (), "INVALID")

    # Sort descending by priority score
    scored.sort(key=lambda x: x[0], reverse=True)

    entries: list[FollowUpEntry] = []
    n_urgent = 0
    n_high = 0
    for rank, (ps, row) in enumerate(scored, start=1):
        fpp = _extract_fpp(row)
        dc = _extract_dc(row)
        pathway = row.get("pathway") or row.get("submission_pathway")
        provenance = _extract_provenance(row)
        tic_raw = row.get("tic_id")
        try:
            tic_id: int | None = int(tic_raw) if tic_raw is not None else None
        except (TypeError, ValueError):
            tic_id = None
        cid = str(row.get("candidate_id", f"signal_{rank}"))
        period = row.get("period_days")
        try:
            period_f: float | None = float(period) if period is not None else None
        except (TypeError, ValueError):
            period_f = None

        rec = _recommend(ps, fpp, pathway, fpp_skip_threshold, fpp_urgent_threshold)
        if rec == "urgent":
            n_urgent += 1
        elif rec == "high":
            n_high += 1

        entries.append(FollowUpEntry(
            tic_id=tic_id,
            candidate_id=cid,
            period_days=period_f,
            fpp=fpp,
            detection_confidence=dc,
            pathway=pathway,
            provenance_score=provenance,
            priority_score=round(ps, 4),
            priority_rank=rank,
            recommendation=rec,
        ))

    return PrioritizerResult(
        n_candidates=len(entries),
        n_urgent=n_urgent,
        n_high=n_high,
        entries=tuple(entries),
        flag="OK",
    )


def format_followup_priorities(result: PrioritizerResult) -> str:
    """Format followup priorities result as Markdown."""
    lines = [
        "## Candidate Follow-Up Prioritizer",
        "",
        f"- Total candidates: {result.n_candidates}",
        f"- Urgent: {result.n_urgent}",
        f"- High priority: {result.n_high}",
        f"- **Flag: {result.flag}**",
        "",
    ]
    if result.entries:
        lines.append("### Priority Table")
        lines.append("")
        lines.append("| Rank | Candidate | FPP | DC | Pathway | Score | Rec |")
        lines.append("|------|-----------|-----|----|---------|-------|-----|")
        for e in result.entries:
            fpp_s = f"{e.fpp:.3f}" if e.fpp is not None else "—"
            dc_s = f"{e.detection_confidence:.3f}" if e.detection_confidence is not None else "—"
            lines.append(
                f"| {e.priority_rank} | {e.candidate_id} | {fpp_s} | {dc_s} "
                f"| {e.pathway or '—'} | {e.priority_score:.4f} | {e.recommendation} |"
            )
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(
        prog="candidate_followup_prioritizer",
        description="Rank candidates by follow-up priority.",
    )
    parser.add_argument("path", help="Path to pipeline JSON result file")
    parser.add_argument("--fpp-skip", type=float, default=0.70)
    parser.add_argument("--fpp-urgent", type=float, default=0.10)
    args = parser.parse_args(argv)

    data = json.loads(Path(args.path).read_text())
    if isinstance(data, dict):
        data = [data]
    result = prioritize_followup(
        data,
        fpp_skip_threshold=args.fpp_skip,
        fpp_urgent_threshold=args.fpp_urgent,
    )
    print(format_followup_priorities(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
