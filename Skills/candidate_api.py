"""Serve local candidate rows through a read-only standard-library API.

The API loads existing local JSON candidate rows, normalizes them through the
dashboard contract, and exposes JSON plus an optional static dashboard HTML
view. It never queries live services and never mutates source data.

Public API
----------
CandidateAPI(rows, *, title, source_label)
candidate_to_payload(candidate) -> dict[str, Any]
summary_payload(candidates) -> dict[str, Any]
api_response(api, path) -> tuple[int, str, bytes]
run_server(rows, *, host, port, title, source_label) -> None
"""
from __future__ import annotations

import json
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from Skills.candidate_dashboard_export import (
    DashboardCandidate,
    build_dashboard,
    load_dashboard_rows,
    normalize_candidate,
)


class CandidateAPI:
    """Read-only candidate API state."""

    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        title: str = "Candidate Review Dashboard",
        source_label: str = "local JSON",
    ) -> None:
        self.rows = list(rows)
        self.title = title
        self.source_label = source_label
        self.candidates = sorted(
            (normalize_candidate(row) for row in self.rows),
            key=lambda c: (
                c.false_positive_probability is None,
                (
                    c.false_positive_probability
                    if c.false_positive_probability is not None
                    else 1.0
                ),
                c.candidate_id,
            ),
        )

    def candidate_by_id(self, candidate_id: str) -> DashboardCandidate | None:
        """Return a normalized candidate by ID, or None."""
        for candidate in self.candidates:
            if candidate.candidate_id == candidate_id:
                return candidate
        return None


def candidate_to_payload(candidate: DashboardCandidate) -> dict[str, Any]:
    """Convert a dashboard candidate to a JSON-serializable payload."""
    payload = asdict(candidate)
    payload["positive_evidence"] = list(candidate.positive_evidence)
    payload["negative_evidence"] = list(candidate.negative_evidence)
    payload["blocking_issues"] = list(candidate.blocking_issues)
    payload["language_guardrail"] = (
        "candidate signal; no discovery or external validation claim"
    )
    return payload


def summary_payload(candidates: list[DashboardCandidate]) -> dict[str, Any]:
    """Build aggregate summary payload for normalized candidates."""
    fpps = [
        candidate.false_positive_probability
        for candidate in candidates
        if candidate.false_positive_probability is not None
    ]
    risk_counts: dict[str, int] = {}
    pathway_counts: dict[str, int] = {}
    for candidate in candidates:
        risk_counts[candidate.risk_band] = risk_counts.get(candidate.risk_band, 0) + 1
        pathway_counts[candidate.pathway] = pathway_counts.get(candidate.pathway, 0) + 1
    return {
        "n_candidates": len(candidates),
        "min_false_positive_probability": min(fpps) if fpps else None,
        "median_false_positive_probability": (
            sorted(fpps)[len(fpps) // 2] if fpps else None
        ),
        "blocked_count": risk_counts.get("blocked", 0),
        "risk_counts": risk_counts,
        "pathway_counts": pathway_counts,
        "read_only": True,
        "live_services": False,
        "external_submission": False,
        "language_guardrail": (
            "candidate signals and follow-up targets only; no discovery claims"
        ),
    }


def _json_response(
    status: int,
    payload: dict[str, Any] | list[dict[str, Any]],
) -> tuple[int, str, bytes]:
    body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    return status, "application/json; charset=utf-8", body


def api_response(api: CandidateAPI, path: str) -> tuple[int, str, bytes]:
    """Return ``(status, content_type, body)`` for a local API path."""
    parsed = urlparse(path)
    route = parsed.path.rstrip("/") or "/"

    if route == "/health":
        return _json_response(
            200,
            {
                "ok": True,
                "read_only": True,
                "live_services": False,
                "n_candidates": len(api.candidates),
            },
        )
    if route == "/summary":
        return _json_response(200, summary_payload(api.candidates))
    if route == "/candidates":
        return _json_response(
            200,
            [candidate_to_payload(candidate) for candidate in api.candidates],
        )
    if route.startswith("/candidates/"):
        candidate_id = unquote(route.removeprefix("/candidates/"))
        candidate = api.candidate_by_id(candidate_id)
        if candidate is None:
            return _json_response(
                404,
                {"error": "candidate not found", "candidate_id": candidate_id},
            )
        return _json_response(200, candidate_to_payload(candidate))
    if route == "/dashboard":
        body = build_dashboard(
            api.rows,
            title=api.title,
            source_label=api.source_label,
        ).encode("utf-8")
        return 200, "text/html; charset=utf-8", body
    if route == "/" or route == "/index.html":
        return _json_response(
            200,
            {
                "service": "candidate_api",
                "read_only": True,
                "endpoints": [
                    "/health",
                    "/summary",
                    "/candidates",
                    "/candidates/<candidate_id>",
                    "/dashboard",
                ],
            },
        )
    return _json_response(404, {"error": "not found", "path": route})


def make_handler(api: CandidateAPI) -> type[BaseHTTPRequestHandler]:
    """Create a request handler bound to a ``CandidateAPI`` instance."""

    class CandidateAPIHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            status, content_type, body = api_response(api, self.path)
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802
            body = json.dumps(
                {"error": "method not allowed", "read_only": True},
                sort_keys=True,
            ).encode("utf-8")
            self.send_response(405)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Allow", "GET")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    return CandidateAPIHandler


def run_server(
    rows: list[dict[str, Any]],
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    title: str = "Candidate Review Dashboard",
    source_label: str = "local JSON",
) -> None:
    """Run the local read-only HTTP server until interrupted."""
    api = CandidateAPI(rows, title=title, source_label=source_label)
    server = ThreadingHTTPServer((host, port), make_handler(api))
    print(f"Serving read-only candidate API at http://{host}:{port}")
    server.serve_forever()


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="candidate_api",
        description="Serve local candidate JSON through a read-only API.",
    )
    parser.add_argument("files", nargs="+", type=Path, metavar="JSON")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--title", default="Candidate Review Dashboard")
    parser.add_argument("--source-label", default="local JSON")
    args = parser.parse_args(argv)

    rows = load_dashboard_rows(args.files)
    run_server(
        rows,
        host=args.host,
        port=args.port,
        title=args.title,
        source_label=args.source_label,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
