"""Tests for Skills.candidate_api."""
from __future__ import annotations

import json

from Skills.candidate_api import (
    CandidateAPI,
    api_response,
    candidate_to_payload,
    summary_payload,
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
            "positive_evidence": ["Repeated candidate signal"],
            "negative_evidence": ["False-positive evidence not fully excluded"],
            "blocking_issues": [],
        },
    }


def _json_body(body: bytes) -> object:
    return json.loads(body.decode("utf-8"))


def test_candidate_api_sorts_by_fpp() -> None:
    api = CandidateAPI([_row(candidate_id="high", fpp=0.8), _row(candidate_id="low", fpp=0.02)])
    assert [candidate.candidate_id for candidate in api.candidates] == ["low", "high"]


def test_candidate_by_id_returns_candidate() -> None:
    api = CandidateAPI([_row(candidate_id="TIC1-s01")])
    candidate = api.candidate_by_id("TIC1-s01")
    assert candidate is not None
    assert candidate.target_id == "TIC 1"


def test_candidate_by_id_missing_returns_none() -> None:
    api = CandidateAPI([_row(candidate_id="TIC1-s01")])
    assert api.candidate_by_id("missing") is None


def test_candidate_payload_is_json_ready() -> None:
    api = CandidateAPI([_row()])
    payload = candidate_to_payload(api.candidates[0])
    assert payload["positive_evidence"] == ["Repeated candidate signal"]
    assert payload["language_guardrail"].startswith("candidate signal")


def test_summary_payload_marks_read_only_and_no_live_services() -> None:
    api = CandidateAPI([_row()])
    payload = summary_payload(api.candidates)
    assert payload["read_only"] is True
    assert payload["live_services"] is False
    assert payload["external_submission"] is False


def test_summary_payload_counts_blocked_rows() -> None:
    row = _row(fpp=0.01)
    row["explanation"]["blocking_issues"] = ["Missing provenance"]
    api = CandidateAPI([row])
    payload = summary_payload(api.candidates)
    assert payload["blocked_count"] == 1
    assert payload["risk_counts"]["blocked"] == 1


def test_health_endpoint() -> None:
    status, content_type, body = api_response(CandidateAPI([_row()]), "/health")
    payload = _json_body(body)
    assert status == 200
    assert content_type.startswith("application/json")
    assert payload["ok"] is True


def test_summary_endpoint() -> None:
    status, _, body = api_response(CandidateAPI([_row()]), "/summary")
    payload = _json_body(body)
    assert status == 200
    assert payload["n_candidates"] == 1


def test_candidates_endpoint() -> None:
    status, _, body = api_response(CandidateAPI([_row(candidate_id="A")]), "/candidates")
    payload = _json_body(body)
    assert status == 200
    assert payload[0]["candidate_id"] == "A"


def test_candidate_detail_endpoint() -> None:
    status, _, body = api_response(CandidateAPI([_row(candidate_id="A")]), "/candidates/A")
    payload = _json_body(body)
    assert status == 200
    assert payload["candidate_id"] == "A"


def test_candidate_detail_endpoint_decodes_url() -> None:
    api = CandidateAPI([_row(candidate_id="TIC 1/s01")])
    status, _, body = api_response(api, "/candidates/TIC%201%2Fs01")
    payload = _json_body(body)
    assert status == 200
    assert payload["candidate_id"] == "TIC 1/s01"


def test_missing_candidate_endpoint_returns_404() -> None:
    status, _, body = api_response(CandidateAPI([_row(candidate_id="A")]), "/candidates/B")
    payload = _json_body(body)
    assert status == 404
    assert payload["error"] == "candidate not found"


def test_dashboard_endpoint_returns_html() -> None:
    status, content_type, body = api_response(CandidateAPI([_row()]), "/dashboard")
    text = body.decode("utf-8")
    assert status == 200
    assert content_type.startswith("text/html")
    assert "does not claim a confirmed planet" in text


def test_root_endpoint_lists_endpoints() -> None:
    status, _, body = api_response(CandidateAPI([]), "/")
    payload = _json_body(body)
    assert status == 200
    assert "/candidates" in payload["endpoints"]


def test_unknown_route_returns_404() -> None:
    status, _, body = api_response(CandidateAPI([]), "/missing")
    payload = _json_body(body)
    assert status == 404
    assert payload["error"] == "not found"


def test_missing_scores_stay_none_not_zero() -> None:
    api = CandidateAPI([{"candidate_id": "missing", "target_id": "TIC 9"}])
    status, _, body = api_response(api, "/candidates/missing")
    payload = _json_body(body)
    assert status == 200
    assert payload["false_positive_probability"] is None


def test_no_confirmed_planet_claim_in_json_payloads() -> None:
    _, _, body = api_response(CandidateAPI([_row()]), "/summary")
    assert "confirmed planet" not in body.decode("utf-8").lower()


def test_known_object_row_is_still_review_language() -> None:
    row = _row(candidate_id="known")
    row["recommended_pathway"] = "known_object_annotation"
    row["status"] = "candidate_signal"
    api = CandidateAPI([row])
    payload = candidate_to_payload(api.candidates[0])
    assert payload["pathway"] == "known_object_annotation"
    assert "candidate signal" in payload["language_guardrail"]
