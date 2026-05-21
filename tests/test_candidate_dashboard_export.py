"""Tests for Skills.candidate_dashboard_export."""
from __future__ import annotations

import json
from pathlib import Path

from Skills.candidate_dashboard_export import (
    DashboardCandidate,
    build_dashboard,
    load_dashboard_rows,
    normalize_candidate,
    write_dashboard,
)


def _row(
    *,
    candidate_id: str = "TIC1-s01",
    target_id: str = "TIC 1",
    fpp: float | None = 0.10,
    pathway: str = "planet_hunters_discussion",
) -> dict:
    return {
        "candidate_id": candidate_id,
        "target_id": target_id,
        "period_days": 3.5,
        "depth_ppm": 900.0,
        "snr": 12.0,
        "scores": {
            "false_positive_probability": fpp,
            "detection_confidence": 0.82,
        },
        "recommended_pathway": pathway,
        "explanation": {
            "positive_evidence": ["Repeated transit-like event"],
            "negative_evidence": ["No centroid check supplied"],
            "blocking_issues": [],
        },
    }


def test_normalize_candidate_returns_contract() -> None:
    candidate = normalize_candidate(_row())
    assert isinstance(candidate, DashboardCandidate)


def test_normalize_reads_nested_scores() -> None:
    candidate = normalize_candidate(_row(fpp=0.23))
    assert candidate.false_positive_probability == 0.23
    assert candidate.detection_confidence == 0.82


def test_normalize_reads_batch_scan_shape() -> None:
    candidate = normalize_candidate(
        {
            "tic_id": 123,
            "best_fpp": 0.42,
            "best_period_days": 5.0,
            "best_pathway": "github_only_reproducibility",
        }
    )
    assert candidate.target_id == "TIC 123"
    assert candidate.period_days == 5.0
    assert candidate.pathway == "github_only_reproducibility"


def test_normalize_reads_plot_path() -> None:
    candidate = normalize_candidate({**_row(), "phase_fold_plot_path": "plots/TIC1.png"})
    assert candidate.plot_path == "plots/TIC1.png"


def test_normalize_reads_nested_plot_artifact() -> None:
    candidate = normalize_candidate(
        {**_row(), "artifacts": {"phase_fold_plot_path": "plots/nested.webp"}}
    )
    assert candidate.plot_path == "plots/nested.webp"


def test_normalize_ignores_non_image_plot_path() -> None:
    candidate = normalize_candidate({**_row(), "plot_path": "javascript:alert(1)"})
    assert candidate.plot_path is None


def test_low_fpp_risk_band() -> None:
    assert normalize_candidate(_row(fpp=0.05)).risk_band == "low-fpp"


def test_high_fpp_risk_band() -> None:
    assert normalize_candidate(_row(fpp=0.90)).risk_band == "high-fpp"


def test_blocking_issue_overrides_risk_band() -> None:
    row = _row(fpp=0.01)
    row["explanation"]["blocking_issues"] = ["Missing provenance"]
    assert normalize_candidate(row).risk_band == "blocked"


def test_build_dashboard_contains_doctype() -> None:
    html = build_dashboard([_row()], generated_at="2026-05-20 00:00 UTC")
    assert "<!DOCTYPE html>" in html


def test_build_dashboard_contains_conservative_notice() -> None:
    html = build_dashboard([_row()], generated_at="2026-05-20 00:00 UTC")
    assert "does not claim a confirmed planet" in html


def test_build_dashboard_does_not_use_confirmation_claim_as_heading() -> None:
    html = build_dashboard([_row()], generated_at="2026-05-20 00:00 UTC")
    assert "Confirmed Planet" not in html


def test_build_dashboard_contains_false_positive_section() -> None:
    html = build_dashboard([_row()], generated_at="2026-05-20 00:00 UTC")
    assert "False-Positive And Negative Evidence" in html
    assert "No centroid check supplied" in html


def test_build_dashboard_escapes_html() -> None:
    html = build_dashboard(
        [_row(candidate_id="<script>alert(1)</script>")],
        generated_at="2026-05-20 00:00 UTC",
    )
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_build_dashboard_renders_phase_fold_plot() -> None:
    html = build_dashboard(
        [{**_row(), "plot_path": "plots/TIC1_phase_fold.png"}],
        generated_at="2026-05-20 00:00 UTC",
    )
    assert '<figure class="plot-preview">' in html
    assert 'src="plots/TIC1_phase_fold.png"' in html
    assert "Phase-fold plot artifact" in html


def test_build_dashboard_escapes_plot_path() -> None:
    html = build_dashboard(
        [{**_row(), "plot_path": 'plots/<bad "name">.png'}],
        generated_at="2026-05-20 00:00 UTC",
    )
    assert 'plots/<bad "name">.png' not in html
    assert "plots/&lt;bad &quot;name&quot;&gt;.png" in html


def test_build_dashboard_handles_empty_rows() -> None:
    html = build_dashboard([], generated_at="2026-05-20 00:00 UTC")
    assert "No candidate signals loaded" in html


def test_build_dashboard_sorts_by_fpp() -> None:
    html = build_dashboard(
        [
            _row(candidate_id="high", fpp=0.8),
            _row(candidate_id="low", fpp=0.05),
        ],
        generated_at="2026-05-20 00:00 UTC",
    )
    assert html.index('href="#low"') < html.index('href="#high"')


def test_write_dashboard_creates_parent_directory(tmp_path: Path) -> None:
    out = tmp_path / "reports" / "dashboard.html"
    path = write_dashboard([_row()], out, generated_at="2026-05-20 00:00 UTC")
    assert path == out
    assert out.exists()


def test_load_dashboard_rows_from_list_json(tmp_path: Path) -> None:
    path = tmp_path / "rows.json"
    path.write_text(json.dumps([_row(candidate_id="a"), _row(candidate_id="b")]))
    rows = load_dashboard_rows([path])
    assert [row["candidate_id"] for row in rows] == ["a", "b"]
    assert all(row["_source_file"] == str(path) for row in rows)


def test_load_dashboard_rows_from_candidates_key(tmp_path: Path) -> None:
    path = tmp_path / "wrapped.json"
    path.write_text(json.dumps({"candidates": [_row(candidate_id="wrapped")]}))
    rows = load_dashboard_rows([path])
    assert rows[0]["candidate_id"] == "wrapped"


def test_load_dashboard_rows_skips_non_rows(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps("not a row"))
    assert load_dashboard_rows([path]) == []


def test_missing_scores_are_rendered_as_unknown() -> None:
    row = {"candidate_id": "missing", "target_id": "TIC 9"}
    html = build_dashboard([row], generated_at="2026-05-20 00:00 UTC")
    assert "unknown" in html
    assert "n/a" in html
