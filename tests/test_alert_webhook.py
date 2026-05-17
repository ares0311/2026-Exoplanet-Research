"""Tests for Skills.alert_webhook."""
from __future__ import annotations

import pytest
from Skills.alert_webhook import (
    AlertPayload,
    build_alert_payload,
    format_generic_payload,
    format_slack_payload,
    send_alert,
)


def _row(**kw: object) -> dict:
    base = {
        "tic_id": 12345,
        "candidate_id": "TIC12345-001",
        "best_fpp": 0.05,
        "best_pathway": "tfop_ready",
        "period_days": 7.42,
        "rank_score": 0.88,
    }
    base.update(kw)
    return base


class TestBuildAlertPayload:
    def test_returns_alert_payload(self) -> None:
        p = build_alert_payload(_row())
        assert isinstance(p, AlertPayload)

    def test_tic_id_stored(self) -> None:
        p = build_alert_payload(_row(tic_id=99))
        assert p.tic_id == 99

    def test_fpp_stored(self) -> None:
        p = build_alert_payload(_row(best_fpp=0.07))
        assert p.fpp == pytest.approx(0.07)

    def test_pathway_stored(self) -> None:
        p = build_alert_payload(_row())
        assert p.pathway == "tfop_ready"

    def test_period_stored(self) -> None:
        p = build_alert_payload(_row(period_days=3.5))
        assert p.period_days == pytest.approx(3.5)

    def test_custom_message_preserved(self) -> None:
        p = build_alert_payload(_row(), message="Custom note")
        assert p.message == "Custom note"

    def test_auto_message_contains_candidate_id(self) -> None:
        p = build_alert_payload(_row(candidate_id="TIC1-001"))
        assert "TIC1-001" in p.message

    def test_none_fpp_handled(self) -> None:
        p = build_alert_payload(_row(best_fpp=None))
        assert p.fpp is None


class TestFormatSlackPayload:
    def test_returns_dict(self) -> None:
        p = build_alert_payload(_row())
        body = format_slack_payload(p)
        assert isinstance(body, dict)

    def test_blocks_key_present(self) -> None:
        p = build_alert_payload(_row())
        body = format_slack_payload(p)
        assert "blocks" in body

    def test_candidate_id_in_body(self) -> None:
        p = build_alert_payload(_row(candidate_id="TIC1-001"))
        body = format_slack_payload(p)
        text = str(body)
        assert "TIC1-001" in text


class TestFormatGenericPayload:
    def test_text_key_present(self) -> None:
        p = build_alert_payload(_row())
        body = format_generic_payload(p)
        assert "text" in body

    def test_text_is_string(self) -> None:
        p = build_alert_payload(_row())
        body = format_generic_payload(p)
        assert isinstance(body["text"], str)


class TestSendAlert:
    def test_returns_true_when_http_fn_ok(self) -> None:
        p = build_alert_payload(_row())
        ok = send_alert(p, "http://example.com",
                        http_fn=lambda url, data, h: {"ok": True})
        assert ok is True

    def test_returns_false_when_http_fn_not_ok(self) -> None:
        p = build_alert_payload(_row())
        ok = send_alert(p, "http://example.com",
                        http_fn=lambda url, data, h: {"ok": False})
        assert ok is False

    def test_no_url_raises_value_error(self) -> None:
        p = build_alert_payload(_row())
        import os
        old = os.environ.pop("EXO_WEBHOOK_URL", None)
        try:
            with pytest.raises(ValueError):
                send_alert(p, None)
        finally:
            if old is not None:
                os.environ["EXO_WEBHOOK_URL"] = old

    def test_generic_format_used_when_specified(self) -> None:
        captured: list[bytes] = []
        def _fn(url: str, data: bytes, h: dict) -> dict:
            captured.append(data)
            return {"ok": True}
        p = build_alert_payload(_row())
        send_alert(p, "http://example.com", format="generic", http_fn=_fn)
        import json
        body = json.loads(captured[0])
        assert "text" in body
        assert "blocks" not in body
