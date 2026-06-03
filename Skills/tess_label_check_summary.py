"""Summarize the live TESS label-check SQLite audit log.

This utility is read-only. It does not query ExoFOP and does not create live
network traffic; it only reads rows written by ``count_tess_labels.py``.

Usage
-----
    python Skills/tess_label_check_summary.py
    python Skills/tess_label_check_summary.py --json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_LOG_DB = _PROJECT_ROOT / "logs" / "tess_label_check.sqlite3"


@dataclass(frozen=True)
class TessLabelCheckSummary:
    db_path: str
    exists: bool
    n_runs: int
    n_success: int
    n_errors: int
    latest_started_at: str | None
    latest_status: str | None
    latest_exit_code: int | None
    latest_cp: int | None
    latest_fp: int | None
    latest_eb: int | None
    latest_total: int | None
    latest_gate_open: bool | None
    latest_error_message: str | None
    last_success_started_at: str | None
    last_success_cp: int | None
    last_success_total: int | None
    last_success_gate_open: bool | None


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return dict(row)


def _bool_or_none(value: object) -> bool | None:
    if value is None:
        return None
    return bool(value)


def build_summary(db_path: Path = _DEFAULT_LOG_DB) -> TessLabelCheckSummary:
    """Build a read-only summary of the TESS label-check audit log."""
    if not db_path.exists():
        return TessLabelCheckSummary(
            db_path=str(db_path),
            exists=False,
            n_runs=0,
            n_success=0,
            n_errors=0,
            latest_started_at=None,
            latest_status=None,
            latest_exit_code=None,
            latest_cp=None,
            latest_fp=None,
            latest_eb=None,
            latest_total=None,
            latest_gate_open=None,
            latest_error_message=None,
            last_success_started_at=None,
            last_success_cp=None,
            last_success_total=None,
            last_success_gate_open=None,
        )

    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        n_runs = int(conn.execute("SELECT COUNT(*) FROM tess_label_checks").fetchone()[0])
        n_success = int(
            conn.execute(
                "SELECT COUNT(*) FROM tess_label_checks WHERE status = 'success'"
            ).fetchone()[0]
        )
        n_errors = int(
            conn.execute(
                "SELECT COUNT(*) FROM tess_label_checks WHERE status = 'error'"
            ).fetchone()[0]
        )
        latest = _row_to_dict(
            conn.execute(
                "SELECT * FROM tess_label_checks ORDER BY id DESC LIMIT 1"
            ).fetchone()
        )
        last_success = _row_to_dict(
            conn.execute(
                """
                SELECT * FROM tess_label_checks
                WHERE status = 'success'
                ORDER BY id DESC
                LIMIT 1
                """
            ).fetchone()
        )

    return TessLabelCheckSummary(
        db_path=str(db_path),
        exists=True,
        n_runs=n_runs,
        n_success=n_success,
        n_errors=n_errors,
        latest_started_at=latest["started_at"] if latest else None,
        latest_status=latest["status"] if latest else None,
        latest_exit_code=latest["exit_code"] if latest else None,
        latest_cp=latest["cp"] if latest else None,
        latest_fp=latest["fp"] if latest else None,
        latest_eb=latest["eb"] if latest else None,
        latest_total=latest["total"] if latest else None,
        latest_gate_open=_bool_or_none(latest["gate_open"] if latest else None),
        latest_error_message=latest["error_message"] if latest else None,
        last_success_started_at=last_success["started_at"] if last_success else None,
        last_success_cp=last_success["cp"] if last_success else None,
        last_success_total=last_success["total"] if last_success else None,
        last_success_gate_open=_bool_or_none(
            last_success["gate_open"] if last_success else None
        ),
    )


def summary_to_dict(summary: TessLabelCheckSummary) -> dict[str, object]:
    """Convert a summary to JSON-serializable data."""
    return asdict(summary)


def format_summary(summary: TessLabelCheckSummary) -> str:
    """Format the label-check log summary as Markdown."""
    lines = [
        "## TESS Label-Check Log Summary",
        "",
        f"- Log DB: `{summary.db_path}`",
        f"- Exists: {'yes' if summary.exists else 'no'}",
        f"- Runs: {summary.n_runs}",
        f"- Successes: {summary.n_success}",
        f"- Errors: {summary.n_errors}",
    ]
    if not summary.exists:
        lines.append("- Latest run: none")
        return "\n".join(lines)

    lines.extend(
        [
            "",
            "### Latest Run",
            "",
            f"- Started: {summary.latest_started_at or 'unknown'}",
            f"- Status: {summary.latest_status or 'unknown'}",
            f"- Exit code: {summary.latest_exit_code}",
            f"- CP/FP/EB/total: {summary.latest_cp}/{summary.latest_fp}/"
            f"{summary.latest_eb}/{summary.latest_total}",
            f"- Gate open: {summary.latest_gate_open}",
        ]
    )
    if summary.latest_error_message:
        lines.append(f"- Error: {summary.latest_error_message}")

    lines.extend(
        [
            "",
            "### Last Successful Count",
            "",
            f"- Started: {summary.last_success_started_at or 'none'}",
            f"- CP: {summary.last_success_cp}",
            f"- Total labeled: {summary.last_success_total}",
            f"- Gate open: {summary.last_success_gate_open}",
        ]
    )
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-db",
        type=Path,
        default=_DEFAULT_LOG_DB,
        help=f"SQLite audit log path (default: {_DEFAULT_LOG_DB}).",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON instead of Markdown.")
    return parser.parse_args(argv)


def _cli(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = build_summary(args.log_db)
    if args.json:
        print(json.dumps(summary_to_dict(summary), indent=2))
    else:
        print(format_summary(summary))
    return 0 if summary.exists else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
