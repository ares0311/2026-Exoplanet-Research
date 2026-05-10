"""Report-readiness and conservative recommendation helpers."""

from __future__ import annotations

from html import escape
from pathlib import Path

from exo_toolkit.background.followup import report_is_ready
from exo_toolkit.background.schemas import (
    DraftReport,
    FollowUpTestRecord,
    KnownTessTarget,
    SubmissionRecommendation,
)


def build_draft_report(target: KnownTessTarget, tests: list[FollowUpTestRecord]) -> DraftReport:
    ready = report_is_ready(tests)
    blocking_issues = [
        test.rationale for test in tests if test.status.value in {"fail", "blocked", "uncertain"}
    ]
    sections = {
        "abstract": (
            f"{target.target_name} is a known TESS benchmark candidate signal used to validate "
            "background search automation. This report does not claim a new discovery."
        ),
        "target_context": (
            f"{target.target_id}; mission={target.mission}; "
            f"known_object={target.known_object}"
        ),
        "data_provenance": "; ".join(target.provenance),
        "methodology": "Deterministic fixture-only background run with no live catalog access.",
        "evidence_supporting_follow_up": "; ".join(target.positive_evidence),
        "negative_evidence": "; ".join(target.negative_evidence),
        "false_positive_analysis": (
            f"Fixture false-positive risk score is {target.false_positive_risk_score:.2f}; "
            "false positives remain the default hypothesis outside this benchmark fixture."
        ),
        "uncertainty_and_limitations": (
            "Fixture values are static development examples and should be refreshed only through "
            "an explicitly approved live-data workflow."
        ),
        "recommended_next_steps": (
            "Use this record for internal validation, benchmark review, and scheduler testing."
        ),
    }
    return DraftReport(
        target_id=target.target_id,
        ready=ready,
        sections=sections,
        blocking_issues=blocking_issues,
    )


def export_draft_report(
    report: DraftReport, run_id: str, export_dir: Path, formats: list[str]
) -> DraftReport:
    export_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{run_id}_{_safe_name(report.target_id)}"
    paths: dict[str, str] = {}
    if "markdown" in formats:
        markdown_path = export_dir / f"{stem}.md"
        markdown_path.write_text(render_markdown_report(report), encoding="utf-8")
        paths["markdown"] = str(markdown_path)
    if "html" in formats:
        html_path = export_dir / f"{stem}.html"
        html_path.write_text(render_html_report(report), encoding="utf-8")
        paths["html"] = str(html_path)
    return DraftReport(
        target_id=report.target_id,
        ready=report.ready,
        sections=report.sections,
        blocking_issues=report.blocking_issues,
        export_paths=paths,
    )


def render_markdown_report(report: DraftReport) -> str:
    lines = [
        f"# Background Search Report: {report.target_id}",
        "",
        "**Status:** Candidate signal review artifact. This report does not claim discovery, "
        "confirmation, or external validation.",
        "",
        f"**Report ready:** {str(report.ready).lower()}",
        "",
    ]
    for title, body in report.sections.items():
        lines.extend([f"## {_human_title(title)}", "", body, ""])
    if report.blocking_issues:
        lines.extend(["## Blocking Issues", ""])
        lines.extend(f"- {issue}" for issue in report.blocking_issues)
        lines.append("")
    return "\n".join(lines)


def render_html_report(report: DraftReport) -> str:
    sections = "\n".join(
        f"<section><h2>{escape(_human_title(title))}</h2><p>{escape(body)}</p></section>"
        for title, body in report.sections.items()
    )
    status = (
        "Candidate signal review artifact. This report does not claim discovery, "
        "confirmation, or external validation."
    )
    blocking = ""
    if report.blocking_issues:
        items = "\n".join(f"<li>{escape(issue)}</li>" for issue in report.blocking_issues)
        blocking = f"<section><h2>Blocking Issues</h2><ul>{items}</ul></section>"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Background Search Report: {escape(report.target_id)}</title>
</head>
<body>
  <h1>Background Search Report: {escape(report.target_id)}</h1>
  <p><strong>Status:</strong> {escape(status)}</p>
  <p><strong>Report ready:</strong> {str(report.ready).lower()}</p>
  {sections}
  {blocking}
</body>
</html>
"""


def build_submission_recommendations(target: KnownTessTarget) -> list[SubmissionRecommendation]:
    known_object_risk = "Known TESS benchmark; do not present as a new candidate."
    return [
        SubmissionRecommendation(
            destination="Internal project review",
            rank=1,
            suitability_rationale="Best first destination for inspecting benchmark evidence.",
            risks=[known_object_risk, "Fixture-only evidence is not live external validation."],
            prerequisites=["Human review of fixture provenance", "Validation-summary check"],
            recommended_action="internal_review",
        ),
        SubmissionRecommendation(
            destination="Project reproducibility log",
            rank=2,
            suitability_rationale="Appropriate for preserving deterministic benchmark run outputs.",
            risks=["Not an external submission path"],
            prerequisites=["SQLite ledger and exactly-one-outcome invariant"],
            recommended_action="record_for_reproducibility",
        ),
        SubmissionRecommendation(
            destination="External submission",
            rank=3,
            suitability_rationale="External submission is intentionally blocked by default.",
            risks=[known_object_risk, "Explicit human approval is required before any contact."],
            prerequisites=[
                "Explicit user approval",
                "Live-data provenance if external use is intended",
            ],
            recommended_action="do_not_submit_yet",
        ),
    ]


def _human_title(value: str) -> str:
    return value.replace("_", " ").title()


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value).strip("_")
