"""Send candidate alert notifications via HTTP webhook (Slack, Discord, generic).

Formats a high-priority candidate row into a short notification message and
POSTs it to a webhook URL.  The webhook URL is read from the environment
variable ``EXO_WEBHOOK_URL`` or passed explicitly.  No external dependencies
beyond stdlib.

Public API
----------
AlertPayload(tic_id, candidate_id, fpp, pathway, period_days, rank_score, message)
build_alert_payload(candidate_row, *, message) -> AlertPayload
format_slack_payload(payload) -> dict        # Slack Block Kit JSON
format_generic_payload(payload) -> dict      # plain {"text": "..."}
send_alert(payload, webhook_url, *, format, timeout) -> bool
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class AlertPayload:
    tic_id: int
    candidate_id: str
    fpp: float | None
    pathway: str
    period_days: float | None
    rank_score: float | None
    message: str


def build_alert_payload(
    candidate_row: dict[str, Any],
    *,
    message: str = "",
) -> AlertPayload:
    """Build an AlertPayload from a candidate result dict.

    Args:
        candidate_row: Dict with pipeline output keys.
        message: Optional custom note to append.

    Returns:
        :class:`AlertPayload`.
    """
    tic_id = int(candidate_row.get("tic_id", 0))
    cid    = str(candidate_row.get("candidate_id", f"TIC{tic_id}"))
    fpp    = candidate_row.get("best_fpp") or candidate_row.get("fpp")
    path   = candidate_row.get("best_pathway") or candidate_row.get("pathway", "")
    period = candidate_row.get("period_days")
    rank   = candidate_row.get("rank_score")

    if not message:
        fpp_str = f"{float(fpp):.3f}" if fpp is not None else "—"
        message = (
            f"New transit candidate {cid}  |  "
            f"P={float(period):.3f} d  |  FPP={fpp_str}  |  {path}"
        ) if period is not None else f"New transit candidate {cid}  |  FPP={fpp_str}  |  {path}"

    return AlertPayload(
        tic_id=tic_id,
        candidate_id=cid,
        fpp=float(fpp) if fpp is not None else None,
        pathway=path,
        period_days=float(period) if period is not None else None,
        rank_score=float(rank) if rank is not None else None,
        message=message,
    )


def format_slack_payload(payload: AlertPayload) -> dict[str, Any]:
    """Format as Slack Block Kit message."""
    fpp_str    = f"{payload.fpp:.3f}" if payload.fpp is not None else "—"
    period_str = f"{payload.period_days:.3f} d" if payload.period_days is not None else "—"
    rank_str   = f"{payload.rank_score:.3f}" if payload.rank_score is not None else "—"

    return {
        "blocks": [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": "🔭 Exoplanet Candidate Alert"},
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Candidate:*\n{payload.candidate_id}"},
                    {"type": "mrkdwn", "text": f"*TIC ID:*\n{payload.tic_id}"},
                    {"type": "mrkdwn", "text": f"*Period:*\n{period_str}"},
                    {"type": "mrkdwn", "text": f"*FPP:*\n{fpp_str}"},
                    {"type": "mrkdwn", "text": f"*Pathway:*\n{payload.pathway}"},
                    {"type": "mrkdwn", "text": f"*Rank score:*\n{rank_str}"},
                ],
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": payload.message},
            },
        ]
    }


def format_generic_payload(payload: AlertPayload) -> dict[str, Any]:
    """Format as a generic webhook payload with a 'text' key."""
    return {"text": payload.message}


def send_alert(
    payload: AlertPayload,
    webhook_url: str | None = None,
    *,
    format: Literal["slack", "generic"] = "slack",
    timeout: int = 10,
    http_fn: Any | None = None,
) -> bool:
    """POST an alert to the webhook URL.

    Args:
        payload: Built by :func:`build_alert_payload`.
        webhook_url: Target URL.  Falls back to ``EXO_WEBHOOK_URL`` env var.
        format: ``"slack"`` (Block Kit) or ``"generic"`` (plain JSON).
        timeout: HTTP timeout in seconds.
        http_fn: Injectable ``(url, data, headers) -> response`` for testing.

    Returns:
        True if the webhook returned 2xx; False otherwise.
    """
    url = webhook_url or os.environ.get("EXO_WEBHOOK_URL", "")
    if not url:
        raise ValueError(
            "No webhook URL provided and EXO_WEBHOOK_URL is not set."
        )

    body = (
        format_slack_payload(payload)
        if format == "slack"
        else format_generic_payload(payload)
    )
    data = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"}

    if http_fn is not None:
        resp = http_fn(url, data, headers)
        return resp.get("ok", True)

    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return 200 <= r.status < 300
    except (urllib.error.URLError, OSError):
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="alert_webhook",
        description="Send a candidate alert notification via webhook.",
    )
    parser.add_argument("--candidate", required=True, metavar="JSON",
                        help="Candidate result JSON file or '-' for stdin.")
    parser.add_argument("--url", default=None,
                        help="Webhook URL (overrides EXO_WEBHOOK_URL).")
    parser.add_argument("--format", choices=["slack", "generic"], default="slack")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print payload without sending.")
    args = parser.parse_args(argv)

    import sys
    from pathlib import Path

    if args.candidate == "-":
        row = json.loads(sys.stdin.read())
    else:
        row = json.loads(Path(args.candidate).read_text())

    if isinstance(row, list):
        row = row[0]

    payload = build_alert_payload(row)

    if args.dry_run:
        body = (format_slack_payload(payload) if args.format == "slack"
                else format_generic_payload(payload))
        print(json.dumps(body, indent=2))
        return 0

    ok = send_alert(payload, args.url, format=args.format)  # type: ignore[arg-type]
    print("Alert sent successfully." if ok else "Alert delivery failed.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
