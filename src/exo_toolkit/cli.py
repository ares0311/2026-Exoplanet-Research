"""Command line interface for exo-toolkit."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from exo_toolkit.background.config import DEFAULT_CONFIG_PATH, ConfigError, load_background_config
from exo_toolkit.background.fixtures import fixture_summary, load_known_tess_examples
from exo_toolkit.background.priority import build_priority_summary
from exo_toolkit.background.runner import background_run_once
from exo_toolkit.background.storage import BackgroundStore

DEFAULT_DB_PATH = Path("logs/background_search.sqlite3")
EXIT_SUCCESS = 0
EXIT_NEEDS_FOLLOW_UP = 20
EXIT_BLOCKED = 30
EXIT_CONFIG_ERROR = 40
EXIT_INTERNAL_ERROR = 50


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _db_path(args: argparse.Namespace) -> Path:
    return Path(args.db_path)


def _cmd_target_priority_summary(args: argparse.Namespace) -> int:
    store = BackgroundStore(_db_path(args))
    config = load_background_config(Path(args.config_path))
    targets = load_known_tess_examples()
    summary = build_priority_summary(targets, store, config)
    _print_json(summary.to_jsonable())
    return 0


def _cmd_config_summary(args: argparse.Namespace) -> int:
    config = load_background_config(Path(args.config_path))
    _print_json(
        {
            "path": str(config.path),
            "version": config.version,
            "schema_version": config.schema_version,
            "fingerprint": config.fingerprint,
            "target_pool": config.target_pool,
            "priority_weights": config.priority_weights,
            "thresholds": config.thresholds,
            "scheduler": config.scheduler,
            "reports": config.reports,
        }
    )
    return EXIT_SUCCESS


def _cmd_fixture_summary(args: argparse.Namespace) -> int:
    _print_json({"fixtures": fixture_summary()})
    return EXIT_SUCCESS


def _cmd_background_run_once(args: argparse.Namespace) -> int:
    try:
        result = background_run_once(
            db_path=_db_path(args),
            command="exo background-run-once",
            config_path=Path(args.config_path),
            target_id=args.target_id,
            dry_run=args.dry_run,
            export_reports=not args.no_report_export,
        )
    except ConfigError as error:
        _print_json({"status": "config_error", "error": str(error)})
        return EXIT_CONFIG_ERROR if args.scheduler_exit_codes else EXIT_SUCCESS
    except Exception as error:
        _print_json({"status": "internal_error", "error": str(error)})
        return EXIT_INTERNAL_ERROR if args.scheduler_exit_codes else EXIT_SUCCESS
    payload = result.to_jsonable()
    _print_json(payload)
    if not args.scheduler_exit_codes:
        return EXIT_SUCCESS
    if payload["outcome"] == "needs_follow_up":
        return EXIT_NEEDS_FOLLOW_UP
    if payload["outcome"] == "blocked":
        return EXIT_BLOCKED
    return EXIT_SUCCESS


def _cmd_background_ledger_summary(args: argparse.Namespace) -> int:
    store = BackgroundStore(_db_path(args))
    _print_json({"runs": store.list_run_ledger()})
    return 0


def _cmd_reviewed_log_summary(args: argparse.Namespace) -> int:
    store = BackgroundStore(_db_path(args))
    _print_json({"reviewed": store.list_reviewed_outcomes()})
    return 0


def _cmd_needs_follow_up_summary(args: argparse.Namespace) -> int:
    store = BackgroundStore(_db_path(args))
    _print_json({"needs_follow_up": store.list_needs_follow_up_outcomes()})
    return 0


def _cmd_follow_up_test_summary(args: argparse.Namespace) -> int:
    store = BackgroundStore(_db_path(args))
    _print_json({"follow_up_tests": store.list_follow_up_tests()})
    return 0


def _cmd_draft_report_summary(args: argparse.Namespace) -> int:
    store = BackgroundStore(_db_path(args))
    _print_json({"draft_reports": store.list_draft_reports()})
    return 0


def _cmd_submission_recommendation_summary(args: argparse.Namespace) -> int:
    store = BackgroundStore(_db_path(args))
    _print_json({"submission_recommendations": store.list_submission_recommendations()})
    return 0


def _cmd_report_export_summary(args: argparse.Namespace) -> int:
    store = BackgroundStore(_db_path(args))
    _print_json({"report_exports": store.list_report_exports()})
    return 0


def _cmd_approval_record_summary(args: argparse.Namespace) -> int:
    store = BackgroundStore(_db_path(args))
    _print_json({"approval_records": store.list_approval_records()})
    return 0


def _cmd_run_summary(args: argparse.Namespace) -> int:
    store = BackgroundStore(_db_path(args))
    _print_json(store.run_summary(args.run_id))
    return EXIT_SUCCESS


def _cmd_target_history(args: argparse.Namespace) -> int:
    store = BackgroundStore(_db_path(args))
    _print_json(store.target_history(args.target_id))
    return EXIT_SUCCESS


def _cmd_sqlite_integrity(args: argparse.Namespace) -> int:
    store = BackgroundStore(_db_path(args))
    summary = store.integrity_summary()
    _print_json(summary)
    return EXIT_SUCCESS if summary["ok"] else EXIT_BLOCKED


def _cmd_scheduler_notification_summary(args: argparse.Namespace) -> int:
    store = BackgroundStore(_db_path(args))
    _print_json(store.scheduler_notification_summary(args.run_id))
    return EXIT_SUCCESS


def _cmd_validation_summary(args: argparse.Namespace) -> int:
    store = BackgroundStore(_db_path(args))
    validation = store.validation_summary()
    _print_json(validation)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="exo")
    subparsers = parser.add_subparsers(dest="command", required=True)

    commands = {
        "config-summary": _cmd_config_summary,
        "fixture-summary": _cmd_fixture_summary,
        "target-priority-summary": _cmd_target_priority_summary,
        "background-run-once": _cmd_background_run_once,
        "background-ledger-summary": _cmd_background_ledger_summary,
        "reviewed-log-summary": _cmd_reviewed_log_summary,
        "needs-follow-up-summary": _cmd_needs_follow_up_summary,
        "follow-up-test-summary": _cmd_follow_up_test_summary,
        "draft-report-summary": _cmd_draft_report_summary,
        "submission-recommendation-summary": _cmd_submission_recommendation_summary,
        "report-export-summary": _cmd_report_export_summary,
        "approval-record-summary": _cmd_approval_record_summary,
        "run-summary": _cmd_run_summary,
        "target-history": _cmd_target_history,
        "sqlite-integrity": _cmd_sqlite_integrity,
        "scheduler-notification-summary": _cmd_scheduler_notification_summary,
        "validation-summary": _cmd_validation_summary,
    }
    for name, handler in commands.items():
        command_parser = subparsers.add_parser(name)
        command_parser.add_argument(
            "--db-path",
            default=str(DEFAULT_DB_PATH),
            help="SQLite path for background logs. Defaults to logs/background_search.sqlite3.",
        )
        command_parser.add_argument(
            "--config-path",
            default=str(DEFAULT_CONFIG_PATH),
            help="Background config path. Defaults to configs/background_search_v0.json.",
        )
        if name == "background-run-once":
            command_parser.add_argument(
                "--target-id",
                default=None,
                help="Run one specific fixture target by target_id.",
            )
            command_parser.add_argument(
                "--dry-run",
                action="store_true",
                help="Plan the run without writing SQLite rows or report files.",
            )
            command_parser.add_argument(
                "--no-report-export",
                action="store_true",
                help="Do not export Markdown/HTML draft reports for this run.",
            )
            command_parser.add_argument(
                "--scheduler-exit-codes",
                action="store_true",
                help="Return non-zero exit codes for needs-follow-up, blocked, and errors.",
            )
        if name in {"run-summary", "scheduler-notification-summary"}:
            command_parser.add_argument(
                "--run-id",
                default=None,
                help="Run id to summarize. Defaults to the latest completed run.",
            )
        if name == "target-history":
            command_parser.add_argument(
                "--target-id", required=True, help="Target id to summarize."
            )
        command_parser.set_defaults(handler=handler)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = args.handler
    return int(handler(args))


def app() -> int:
    """Console script entry point used by pyproject.toml."""
    return main()


if __name__ == "__main__":
    raise SystemExit(main())
