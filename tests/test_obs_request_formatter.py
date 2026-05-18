"""Tests for Skills.obs_request_formatter."""
from __future__ import annotations

import json

from Skills.obs_request_formatter import (
    ObsRequestResult,
    build_obs_request,
    format_obs_request,
)

_BASE = {
    "tic_id": 150428135,
    "ra_deg": 93.185,
    "dec_deg": -65.179,
    "tmag": 11.3,
    "period_days": 37.42,
    "epoch_bjd": 2458360.0,
    "duration_hours": 2.8,
    "depth_ppm": 1050.0,
}


class TestBuildObsRequest:
    def test_returns_result(self) -> None:
        r = build_obs_request(**_BASE)
        assert isinstance(r, ObsRequestResult)

    def test_request_fields(self) -> None:
        r = build_obs_request(**_BASE)
        assert r.request.tic_id == 150428135
        assert r.request.period_days == 37.42

    def test_priority_default_medium(self) -> None:
        r = build_obs_request(**_BASE)
        assert r.request.priority == "MEDIUM"

    def test_priority_high(self) -> None:
        r = build_obs_request(**_BASE, priority="HIGH")
        assert r.request.priority == "HIGH"

    def test_invalid_priority_defaults_medium(self) -> None:
        r = build_obs_request(**_BASE, priority="URGENT")
        assert r.request.priority == "MEDIUM"

    def test_markdown_contains_tic_id(self) -> None:
        r = build_obs_request(**_BASE)
        assert "150428135" in r.markdown

    def test_markdown_contains_period(self) -> None:
        r = build_obs_request(**_BASE)
        assert "37.42" in r.markdown

    def test_json_parseable(self) -> None:
        r = build_obs_request(**_BASE)
        data = json.loads(r.json_payload)
        assert data["tic_id"] == 150428135

    def test_json_contains_ra_hms(self) -> None:
        r = build_obs_request(**_BASE)
        data = json.loads(r.json_payload)
        assert "ra_hms" in data

    def test_notes_in_markdown(self) -> None:
        r = build_obs_request(**_BASE, notes="Check for centroid shift.")
        assert "centroid" in r.markdown.lower()


class TestFormatObsRequest:
    def test_returns_string(self) -> None:
        r = build_obs_request(**_BASE)
        assert isinstance(format_obs_request(r), str)

    def test_same_as_markdown(self) -> None:
        r = build_obs_request(**_BASE)
        assert format_obs_request(r) == r.markdown
