"""Generate a comprehensive report card for a single transit candidate.

Integrates multiple vetting diagnostics into a single structured Markdown/JSON
report card for human review and archiving.

Public API
----------
ReportCardSection(title, status, details)
CandidateReportCard(tic_id, period_days, epoch_bjd, depth_ppm,
                    sections, overall_flag, recommendation, generated_at)
build_report_card(candidate_row, *, diagnostics) -> CandidateReportCard
format_report_card(card) -> str
save_report_card(card, path) -> Path
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class ReportCardSection:
    title: str
    status: str            # "PASS", "WARN", "FAIL", "SKIP"
    details: str


@dataclass(frozen=True)
class CandidateReportCard:
    tic_id: int
    period_days: float
    epoch_bjd: float
    depth_ppm: float
    sections: tuple[ReportCardSection, ...]
    overall_flag: str          # "PROMISING", "MARGINAL", "LIKELY_FP", "INSUFFICIENT"
    recommendation: str
    generated_at: str


_STATUS_RANK = {"PASS": 0, "WARN": 1, "FAIL": 2, "SKIP": 3}


def _overall_flag(sections: list[ReportCardSection]) -> tuple[str, str]:
    """Derive overall flag and recommendation from section statuses."""
    n_fail = sum(1 for s in sections if s.status == "FAIL")
    n_warn = sum(1 for s in sections if s.status == "WARN")
    n_pass = sum(1 for s in sections if s.status == "PASS")

    if n_fail >= 2:
        return "LIKELY_FP", "Probable false positive — do not prioritise follow-up."
    if n_fail == 1:
        return "MARGINAL", "At least one vetting check failed — review carefully before follow-up."
    if n_warn >= 3:
        return "MARGINAL", "Multiple warnings — proceed with caution."
    if n_pass >= 3 and n_fail == 0:
        return "PROMISING", "Candidate passes all active vetting checks — recommend follow-up."
    return "INSUFFICIENT", "Insufficient diagnostics available to make a determination."


def build_report_card(
    candidate_row: dict,
    *,
    diagnostics: list[dict] | None = None,
) -> CandidateReportCard:
    """Build a report card from a pipeline output row and optional diagnostics.

    Args:
        candidate_row: Output dict from ``run_pipeline`` containing at minimum
            ``tic_id``, ``period_days``, ``epoch_bjd``, ``depth_ppm``,
            ``false_positive_probability``, ``pathway``.
        diagnostics: Optional list of diagnostic result dicts, each with
            ``title``, ``status``, ``details``.

    Returns:
        :class:`CandidateReportCard`.
    """
    tic_id = int(candidate_row.get("tic_id", 0))
    period = float(candidate_row.get("period_days", 0.0))
    epoch = float(candidate_row.get("epoch_bjd", 0.0))
    depth = float(candidate_row.get("depth_ppm", 0.0))

    fpp = candidate_row.get("false_positive_probability")
    if fpp is None:
        scores = candidate_row.get("scores", {})
        fpp = scores.get("false_positive_probability")
    fpp_val = float(fpp) if fpp is not None else None

    pathway = str(candidate_row.get("pathway", "unknown"))
    n_transits = candidate_row.get("n_transits", candidate_row.get("transit_count"))

    # Build core sections
    sections: list[ReportCardSection] = []

    # FPP section
    if fpp_val is not None:
        if fpp_val < 0.1:
            fpp_status = "PASS"
        elif fpp_val < 0.5:
            fpp_status = "WARN"
        else:
            fpp_status = "FAIL"
        sections.append(ReportCardSection(
            title="False Positive Probability",
            status=fpp_status,
            details=f"FPP = {fpp_val:.3f}",
        ))
    else:
        sections.append(ReportCardSection("False Positive Probability", "SKIP", "Not computed"))

    # Transit count section
    if n_transits is not None:
        n = int(n_transits)
        if n >= 3:
            tc_status = "PASS"
        elif n >= 2:
            tc_status = "WARN"
        else:
            tc_status = "FAIL"
        sections.append(ReportCardSection(
            title="Transit Count",
            status=tc_status,
            details=f"{n} transits observed",
        ))

    # Pathway section
    good_pathways = {"tfop_ready", "kepler_archive_candidate"}
    warn_pathways = {"planet_hunters_discussion"}
    if pathway in good_pathways:
        pw_status = "PASS"
    elif pathway in warn_pathways or pathway == "github_only_reproducibility":
        pw_status = "WARN"
    else:
        pw_status = "SKIP"
    sections.append(ReportCardSection(
        title="Submission Pathway",
        status=pw_status,
        details=pathway.replace("_", " ").title(),
    ))

    # Inject any caller-supplied diagnostics
    if diagnostics:
        for d in diagnostics:
            sections.append(ReportCardSection(
                title=str(d.get("title", "Diagnostic")),
                status=str(d.get("status", "SKIP")),
                details=str(d.get("details", "")),
            ))

    overall, recommendation = _overall_flag(sections)

    return CandidateReportCard(
        tic_id=tic_id,
        period_days=period,
        epoch_bjd=epoch,
        depth_ppm=depth,
        sections=tuple(sections),
        overall_flag=overall,
        recommendation=recommendation,
        generated_at=datetime.now(UTC).isoformat(),
    )


def format_report_card(card: CandidateReportCard) -> str:
    """Format a report card as Markdown."""
    status_icon = {
        "PASS": "✓", "WARN": "⚠", "FAIL": "✗", "SKIP": "—",
        "PROMISING": "★", "MARGINAL": "?", "LIKELY_FP": "✗", "INSUFFICIENT": "?",
    }
    icon = status_icon.get(card.overall_flag, "?")
    lines = [
        f"# Candidate Report Card: TIC {card.tic_id}",
        "",
        f"**Overall:** {icon} {card.overall_flag}",
        "",
        f"> {card.recommendation}",
        "",
        "## Transit Parameters",
        "",
        "| Parameter | Value |",
        "|---|---|",
        f"| Period | {card.period_days:.6f} d |",
        f"| Epoch (BJD) | {card.epoch_bjd:.4f} |",
        f"| Depth | {card.depth_ppm:.1f} ppm |",
        "",
        "## Vetting Sections",
        "",
        "| Section | Status | Details |",
        "|---|---|---|",
    ]
    for sec in card.sections:
        icon_s = status_icon.get(sec.status, sec.status)
        lines.append(f"| {sec.title} | {icon_s} {sec.status} | {sec.details} |")

    lines += [
        "",
        f"*Generated: {card.generated_at}*",
        "",
    ]
    return "\n".join(lines)


def save_report_card(card: CandidateReportCard, path: Path) -> Path:
    """Save a report card as JSON to path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        **{k: v for k, v in asdict(card).items() if k != "sections"},
        "sections": [
            {"title": s.title, "status": s.status, "details": s.details}
            for s in card.sections
        ],
    }
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp, path)
    except Exception:
        import contextlib
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    return path


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="candidate_report_card",
        description="Generate a vetting report card for a transit candidate.",
    )
    parser.add_argument("--candidate", required=True, metavar="JSON")
    parser.add_argument("--output", metavar="PATH")
    args = parser.parse_args(argv)

    row = json.loads(Path(args.candidate).read_text())
    if isinstance(row, list):
        row = row[0]
    card = build_report_card(row)
    print(format_report_card(card))
    if args.output:
        p = save_report_card(card, Path(args.output))
        print(f"\nSaved to {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
