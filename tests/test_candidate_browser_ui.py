"""Tests for Skills.candidate_browser_ui."""
from __future__ import annotations

import json
from pathlib import Path

from Skills.candidate_browser_ui import build_browser_ui, write_browser_ui


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
            "positive_evidence": ["Repeated candidate signal"],
            "negative_evidence": ["False-positive evidence not fully excluded"],
            "blocking_issues": [],
        },
    }


def _embedded_payload(html: str) -> dict:
    start = html.index('<script type="application/json" id="embedded-candidate-data">')
    start = html.index(">", start) + 1
    end = html.index("</script>", start)
    return json.loads(html[start:end])


def test_build_browser_ui_contains_doctype() -> None:
    html = build_browser_ui([_row()], generated_at="2026-05-20 00:00 UTC")
    assert "<!DOCTYPE html>" in html


def test_build_browser_ui_contains_filter_controls() -> None:
    html = build_browser_ui([_row()], generated_at="2026-05-20 00:00 UTC")
    assert 'id="search"' in html
    assert 'id="risk-filter"' in html
    assert 'id="pathway-filter"' in html


def test_embedded_data_contains_normalized_candidate() -> None:
    html = build_browser_ui([_row(candidate_id="A")], generated_at="2026-05-20 00:00 UTC")
    payload = _embedded_payload(html)
    assert payload["candidates"][0]["candidate_id"] == "A"


def test_embedded_data_contains_plot_path() -> None:
    html = build_browser_ui(
        [{**_row(candidate_id="A"), "plot_path": "plots/A_phase_fold.png"}],
        generated_at="2026-05-20 00:00 UTC",
    )
    payload = _embedded_payload(html)
    assert payload["candidates"][0]["plot_path"] == "plots/A_phase_fold.png"


def test_embedded_data_sorts_by_fpp() -> None:
    html = build_browser_ui(
        [_row(candidate_id="high", fpp=0.9), _row(candidate_id="low", fpp=0.02)],
        generated_at="2026-05-20 00:00 UTC",
    )
    payload = _embedded_payload(html)
    assert [row["candidate_id"] for row in payload["candidates"]] == ["low", "high"]


def test_api_mode_sets_api_base_and_omits_embedded_data() -> None:
    html = build_browser_ui(
        [_row()],
        api_base_url="http://127.0.0.1:8765",
        embed_data=False,
        generated_at="2026-05-20 00:00 UTC",
    )
    assert 'data-api-base="http://127.0.0.1:8765"' in html
    assert 'type="application/json" id="embedded-candidate-data"' not in html


def test_embedded_mode_can_still_include_api_base() -> None:
    html = build_browser_ui(
        [_row()],
        api_base_url="http://127.0.0.1:8765",
        embed_data=True,
        generated_at="2026-05-20 00:00 UTC",
    )
    assert 'data-api-base="http://127.0.0.1:8765"' in html
    assert "embedded-candidate-data" in html


def test_ui_contains_detail_panel() -> None:
    html = build_browser_ui([_row()], generated_at="2026-05-20 00:00 UTC")
    assert 'id="detail"' in html
    assert "False-Positive And Negative Evidence" in html


def test_ui_contains_plot_preview_renderer() -> None:
    html = build_browser_ui([_row()], generated_at="2026-05-20 00:00 UTC")
    assert "plotPreview" in html
    assert "Phase-fold plot artifact" in html


def test_ui_contains_summary_metrics() -> None:
    html = build_browser_ui([_row()], generated_at="2026-05-20 00:00 UTC")
    assert 'id="metric-count"' in html
    assert 'id="metric-min-fpp"' in html
    assert 'id="metric-blocked"' in html


def test_ui_conservative_notice_has_no_confirmation_claim() -> None:
    html = build_browser_ui([_row()], generated_at="2026-05-20 00:00 UTC")
    assert "No discovery or external validation claim" in html
    assert "confirmed planet" not in html.lower()


def test_missing_scores_remain_null_in_embedded_payload() -> None:
    html = build_browser_ui(
        [{"candidate_id": "missing", "target_id": "TIC 9"}],
        generated_at="2026-05-20 00:00 UTC",
    )
    payload = _embedded_payload(html)
    assert payload["candidates"][0]["false_positive_probability"] is None


def test_blocked_candidate_risk_is_embedded() -> None:
    row = _row(fpp=0.01)
    row["explanation"]["blocking_issues"] = ["Missing provenance"]
    html = build_browser_ui([row], generated_at="2026-05-20 00:00 UTC")
    payload = _embedded_payload(html)
    assert payload["candidates"][0]["risk_band"] == "blocked"


def test_html_escapes_title() -> None:
    html = build_browser_ui(
        [_row()],
        title="<script>bad()</script>",
        generated_at="2026-05-20 00:00 UTC",
    )
    assert "<script>bad()</script>" not in html
    assert "&lt;script&gt;" in html


def test_javascript_fetches_candidates_from_api() -> None:
    html = build_browser_ui([], embed_data=False, generated_at="2026-05-20 00:00 UTC")
    assert "/candidates" in html
    assert "fetch(" in html


def test_javascript_filters_by_risk_and_pathway() -> None:
    html = build_browser_ui([_row()], generated_at="2026-05-20 00:00 UTC")
    assert "risk-filter" in html
    assert "pathway-filter" in html
    assert "filteredCandidates" in html


def test_write_browser_ui_creates_file(tmp_path: Path) -> None:
    path = write_browser_ui(
        [_row()],
        tmp_path / "reports" / "browser.html",
        generated_at="2026-05-20 00:00 UTC",
    )
    assert path.exists()
    assert "Candidate Browser" in path.read_text()


def test_write_browser_ui_creates_parent_directory(tmp_path: Path) -> None:
    path = write_browser_ui(
        [_row()],
        tmp_path / "nested" / "browser.html",
        generated_at="2026-05-20 00:00 UTC",
    )
    assert path.parent.exists()


def test_empty_rows_render_zero_metrics() -> None:
    html = build_browser_ui([], generated_at="2026-05-20 00:00 UTC")
    payload = _embedded_payload(html)
    assert payload["summary"]["n_candidates"] == 0
    assert payload["candidates"] == []


def test_source_file_is_preserved_in_embedded_payload() -> None:
    row = _row()
    row["_source_file"] = "results.json"
    html = build_browser_ui([row], generated_at="2026-05-20 00:00 UTC")
    payload = _embedded_payload(html)
    assert payload["candidates"][0]["source_file"] == "results.json"
