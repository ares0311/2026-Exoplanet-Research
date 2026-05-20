"""Build a static, conservative candidate review dashboard.

The dashboard is a local HTML artifact for reviewing existing pipeline,
batch-scan, or candidate-database rows. It does not query live services and it
does not create discovery or confirmation claims.

Public API
----------
DashboardCandidate(...)
load_dashboard_rows(paths) -> list[dict[str, Any]]
normalize_candidate(row) -> DashboardCandidate
build_dashboard(rows, *, title, generated_at, source_label) -> str
write_dashboard(rows, output_path, *, title, generated_at, source_label) -> Path
"""
from __future__ import annotations

import html
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DashboardCandidate:
    candidate_id: str
    target_id: str
    period_days: float | None
    depth_ppm: float | None
    snr: float | None
    false_positive_probability: float | None
    detection_confidence: float | None
    pathway: str
    risk_band: str
    status: str
    positive_evidence: tuple[str, ...]
    negative_evidence: tuple[str, ...]
    blocking_issues: tuple[str, ...]
    source_file: str | None


def _as_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _first_present(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _nested_score(row: dict[str, Any], key: str) -> Any:
    scores = row.get("scores")
    if isinstance(scores, dict):
        return scores.get(key)
    return None


def _text_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list | tuple):
        return tuple(str(item) for item in value if item is not None)
    return (str(value),)


def _evidence(row: dict[str, Any], key: str) -> tuple[str, ...]:
    explanation = row.get("explanation")
    if isinstance(explanation, dict):
        return _text_tuple(explanation.get(key))
    return _text_tuple(row.get(key))


def _target_id(row: dict[str, Any]) -> str:
    value = _first_present(row, "target_id", "tic_id", "target")
    if value is None:
        return "unknown"
    text = str(value)
    return text if text.upper().startswith("TIC") else f"TIC {text}"


def _risk_band(fpp: float | None, blocking_issues: tuple[str, ...]) -> str:
    if blocking_issues:
        return "blocked"
    if fpp is None:
        return "unknown"
    if fpp <= 0.10:
        return "low-fpp"
    if fpp <= 0.35:
        return "moderate-fpp"
    if fpp <= 0.70:
        return "elevated-fpp"
    return "high-fpp"


def load_dashboard_rows(paths: list[Path | str]) -> list[dict[str, Any]]:
    """Load candidate rows from one or more JSON files.

    Each file may contain a single dict, a list of dicts, or a dict with a
    top-level ``candidates`` or ``rows`` list. A ``_source_file`` key is added
    for provenance.
    """
    rows: list[dict[str, Any]] = []
    for path_like in paths:
        path = Path(path_like)
        data = json.loads(path.read_text())
        if isinstance(data, dict) and isinstance(data.get("candidates"), list):
            data = data["candidates"]
        elif isinstance(data, dict) and isinstance(data.get("rows"), list):
            data = data["rows"]
        elif isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            continue
        for row in data:
            if isinstance(row, dict):
                copied = dict(row)
                copied.setdefault("_source_file", str(path))
                rows.append(copied)
    return rows


def normalize_candidate(row: dict[str, Any]) -> DashboardCandidate:
    """Normalize a candidate-like row into a dashboard contract."""
    candidate_id = str(
        _first_present(row, "candidate_id", "id", "signal_id") or _target_id(row)
    )
    fpp = _as_float(
        _first_present(
            row,
            "false_positive_probability",
            "best_fpp",
            "fpp",
        )
    )
    if fpp is None:
        fpp = _as_float(_nested_score(row, "false_positive_probability"))

    detection_confidence = _as_float(
        _first_present(row, "detection_confidence", "best_detection_confidence")
    )
    if detection_confidence is None:
        detection_confidence = _as_float(_nested_score(row, "detection_confidence"))

    pathway = str(
        _first_present(row, "pathway", "best_pathway", "recommended_pathway") or "unknown"
    )
    blocking_issues = _evidence(row, "blocking_issues")
    return DashboardCandidate(
        candidate_id=candidate_id,
        target_id=_target_id(row),
        period_days=_as_float(_first_present(row, "period_days", "best_period_days")),
        depth_ppm=_as_float(_first_present(row, "depth_ppm", "best_depth_ppm")),
        snr=_as_float(_first_present(row, "snr", "best_snr")),
        false_positive_probability=fpp,
        detection_confidence=detection_confidence,
        pathway=pathway,
        risk_band=_risk_band(fpp, blocking_issues),
        status=str(_first_present(row, "status", "outcome") or "candidate_signal"),
        positive_evidence=_evidence(row, "positive_evidence"),
        negative_evidence=_evidence(row, "negative_evidence"),
        blocking_issues=blocking_issues,
        source_file=str(row.get("_source_file")) if row.get("_source_file") else None,
    )


def _fmt_float(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _fmt_period(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.5f}"


def _escape(text: Any) -> str:
    return html.escape(str(text), quote=True)


def _summary(candidates: list[DashboardCandidate]) -> dict[str, Any]:
    fpps = [
        candidate.false_positive_probability
        for candidate in candidates
        if candidate.false_positive_probability is not None
    ]
    by_risk: dict[str, int] = {}
    for candidate in candidates:
        by_risk[candidate.risk_band] = by_risk.get(candidate.risk_band, 0) + 1
    return {
        "n_candidates": len(candidates),
        "min_fpp": min(fpps) if fpps else None,
        "median_fpp": sorted(fpps)[len(fpps) // 2] if fpps else None,
        "blocked_count": by_risk.get("blocked", 0),
        "by_risk": by_risk,
    }


def _evidence_list(items: tuple[str, ...], fallback: str) -> str:
    if not items:
        return f"<li>{_escape(fallback)}</li>"
    return "".join(f"<li>{_escape(item)}</li>" for item in items)


def build_dashboard(
    rows: list[dict[str, Any]],
    *,
    title: str = "Candidate Review Dashboard",
    generated_at: str | None = None,
    source_label: str = "local JSON",
) -> str:
    """Build a static HTML dashboard from candidate-like rows."""
    generated_at = generated_at or datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    candidates = sorted(
        (normalize_candidate(row) for row in rows),
        key=lambda c: (
            c.false_positive_probability is None,
            c.false_positive_probability if c.false_positive_probability is not None else 1.0,
            c.candidate_id,
        ),
    )
    summary = _summary(candidates)

    parts = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="UTF-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{_escape(title)}</title>",
        "<style>",
        ":root { color-scheme: light; font-family: -apple-system, "
        "BlinkMacSystemFont, 'Segoe UI', sans-serif; }",
        "body { margin: 0; color: #1f2933; background: #f6f8fb; }",
        "header { padding: 24px 32px; background: #ffffff; "
        "border-bottom: 1px solid #d9e2ec; }",
        "main { padding: 24px 32px; max-width: 1280px; margin: 0 auto; }",
        "h1 { margin: 0 0 8px; font-size: 30px; }",
        "h2 { margin-top: 30px; font-size: 20px; }",
        ".meta { color: #52606d; margin: 0; }",
        ".notice { margin-top: 16px; padding: 12px 14px; "
        "border-left: 4px solid #8a6d3b; background: #fff8e6; }",
        ".metrics { display: grid; grid-template-columns: repeat(auto-fit, "
        "minmax(150px, 1fr)); gap: 12px; }",
        ".metric { background: #ffffff; border: 1px solid #d9e2ec; "
        "border-radius: 6px; padding: 14px; }",
        ".metric b { display: block; font-size: 24px; margin-top: 4px; }",
        "table { width: 100%; border-collapse: collapse; "
        "background: #ffffff; border: 1px solid #d9e2ec; }",
        "th, td { padding: 10px 12px; border-bottom: 1px solid #e4e7eb; "
        "text-align: left; vertical-align: top; }",
        "th { background: #e6eef8; font-weight: 700; }",
        ".candidate { background: #ffffff; border: 1px solid #d9e2ec; "
        "border-radius: 6px; padding: 16px; margin: 14px 0; }",
        ".tag { display: inline-block; padding: 2px 8px; "
        "border-radius: 999px; border: 1px solid #bcccdc; "
        "background: #f0f4f8; font-size: 12px; }",
        ".blocked { border-color: #b42318; color: #b42318; }",
        ".high-fpp, .elevated-fpp { border-color: #b54708; color: #b54708; }",
        ".low-fpp { border-color: #027a48; color: #027a48; }",
        "ul { margin-top: 8px; }",
        "@media (max-width: 760px) { header, main { padding: 18px; } table { font-size: 13px; } }",
        "</style>",
        "</head>",
        "<body>",
        "<header>",
        f"<h1>{_escape(title)}</h1>",
        f'<p class="meta">Generated { _escape(generated_at) } from { _escape(source_label) }.</p>',
        '<p class="notice">This local dashboard summarizes candidate signals '
        "and follow-up targets. It does not claim a confirmed planet or "
        "external validation.</p>",
        "</header>",
        "<main>",
        "<section>",
        "<h2>Overview</h2>",
        '<div class="metrics">',
        f'<div class="metric">Candidates<b>{summary["n_candidates"]}</b></div>',
        f'<div class="metric">Minimum FPP<b>{_fmt_float(summary["min_fpp"], 4)}</b></div>',
        f'<div class="metric">Median FPP<b>{_fmt_float(summary["median_fpp"], 4)}</b></div>',
        f'<div class="metric">Blocked<b>{summary["blocked_count"]}</b></div>',
        "</div>",
        "</section>",
    ]

    if not candidates:
        parts.extend(
            [
                "<section>",
                "<h2>Candidates</h2>",
                "<p>No candidate signals loaded.</p>",
                "</section>",
            ]
        )
    else:
        parts.extend(
            [
                "<section>",
                "<h2>Candidate Table</h2>",
                "<table>",
                "<thead><tr><th>Candidate</th><th>Target</th>"
                "<th>Period (d)</th><th>FPP</th><th>Detection</th>"
                "<th>Risk</th><th>Pathway</th></tr></thead>",
                "<tbody>",
            ]
        )
        for candidate in candidates:
            candidate_id = _escape(candidate.candidate_id)
            target_id = _escape(candidate.target_id)
            risk_band = _escape(candidate.risk_band)
            parts.append(
                "<tr>"
                f'<td><a href="#{candidate_id}">{candidate_id}</a></td>'
                f"<td>{target_id}</td>"
                f"<td>{_fmt_period(candidate.period_days)}</td>"
                f"<td>{_fmt_float(candidate.false_positive_probability, 4)}</td>"
                f"<td>{_fmt_float(candidate.detection_confidence, 3)}</td>"
                f'<td><span class="tag {risk_band}">{risk_band}</span></td>'
                f"<td>{_escape(candidate.pathway)}</td>"
                "</tr>"
            )
        parts.extend(
            [
                "</tbody>",
                "</table>",
                "</section>",
                "<section>",
                "<h2>Candidate Details</h2>",
            ]
        )
        for candidate in candidates:
            candidate_id = _escape(candidate.candidate_id)
            risk_band = _escape(candidate.risk_band)
            positive_items = _evidence_list(
                candidate.positive_evidence,
                "No positive evidence supplied in the source row.",
            )
            negative_items = _evidence_list(
                candidate.negative_evidence,
                "No false-positive evidence supplied in the source row.",
            )
            blocking_items = _evidence_list(
                candidate.blocking_issues,
                "No blocking issues supplied in the source row.",
            )
            parts.extend(
                [
                    f'<article class="candidate" id="{candidate_id}">',
                    f'<h3>{candidate_id} <span class="tag {risk_band}">'
                    f"{risk_band}</span></h3>",
                    f"<p><b>Target:</b> {_escape(candidate.target_id)} | "
                    f"<b>Status:</b> {_escape(candidate.status)} | "
                    f"<b>Pathway:</b> {_escape(candidate.pathway)}</p>",
                    f"<p><b>Period:</b> {_fmt_period(candidate.period_days)} d | "
                    f"<b>Depth:</b> {_fmt_float(candidate.depth_ppm, 1)} ppm | "
                    f"<b>SNR:</b> {_fmt_float(candidate.snr, 2)}</p>",
                    "<h4>Positive Evidence</h4>",
                    f"<ul>{positive_items}</ul>",
                    "<h4>False-Positive And Negative Evidence</h4>",
                    f"<ul>{negative_items}</ul>",
                    "<h4>Blocking Issues</h4>",
                    f"<ul>{blocking_items}</ul>",
                ]
            )
            if candidate.source_file is not None:
                parts.append(f"<p><b>Source:</b> {_escape(candidate.source_file)}</p>")
            parts.append("</article>")
        parts.extend(["</section>"])

    parts.extend(["</main>", "</body>", "</html>"])
    return "\n".join(parts) + "\n"


def write_dashboard(
    rows: list[dict[str, Any]],
    output_path: Path | str,
    *,
    title: str = "Candidate Review Dashboard",
    generated_at: str | None = None,
    source_label: str = "local JSON",
) -> Path:
    """Write a static dashboard HTML file and return its path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        build_dashboard(
            rows,
            title=title,
            generated_at=generated_at,
            source_label=source_label,
        )
    )
    return path


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="candidate_dashboard_export",
        description="Build a local static candidate review dashboard.",
    )
    parser.add_argument("files", nargs="+", type=Path, metavar="JSON")
    parser.add_argument("--output", type=Path, required=True, metavar="HTML")
    parser.add_argument("--title", default="Candidate Review Dashboard")
    parser.add_argument("--source-label", default="local JSON")
    args = parser.parse_args(argv)

    rows = load_dashboard_rows(args.files)
    write_dashboard(
        rows,
        args.output,
        title=args.title,
        source_label=args.source_label,
    )
    print(f"Dashboard written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
