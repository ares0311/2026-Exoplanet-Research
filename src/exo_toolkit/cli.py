<<<<<<< HEAD
"""Command-line interface for exo-toolkit.

Entry point: ``exo <TIC-ID>``

Runs the full pipeline (fetch → clean → search → vet → score → classify)
on a single target and prints a Rich-formatted candidate report.

Usage examples
--------------
    exo "TIC 150428135"
    exo "TIC 150428135" --mission TESS --min-snr 7.0 --scorer xgboost --model-path model.json
    exo "TIC 150428135" --output results.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from exo_toolkit.clean import clean_lightcurve
from exo_toolkit.fetch import fetch_lightcurve
from exo_toolkit.pathway import classify_submission_pathway
from exo_toolkit.schemas import CandidateSignal, Mission
from exo_toolkit.scoring import score_candidate
from exo_toolkit.search import search_lightcurve
from exo_toolkit.vet import vet_signal

app = typer.Typer(
    name="exo",
    help="Exoplanet transit candidate detection and scoring toolkit.",
    add_completion=False,
)

_VALID_SCORERS = ("bayesian", "xgboost", "ensemble")


# ---------------------------------------------------------------------------
# Pipeline orchestration (separated for testability)
# ---------------------------------------------------------------------------


def run_pipeline(
    target_id: str,
    mission: Mission,
    *,
    min_snr: float = 5.0,
    max_peaks: int = 5,
    scorer: str = "bayesian",
    model_path: Path | None = None,
    fetch_fn: Any = None,
    clean_fn: Any = None,
) -> list[dict[str, Any]]:
    """Run the full pipeline on one target and return serialisable results.

    Args:
        target_id: Target identifier, e.g. ``"TIC 150428135"``.
        mission: ``"TESS"``, ``"Kepler"``, or ``"K2"``.
        min_snr: Minimum BLS SNR threshold for candidate signals.
        max_peaks: Maximum number of signals to return from the BLS search.
        scorer: Scoring model — ``"bayesian"`` (default), ``"xgboost"``, or
            ``"ensemble"`` (average of Bayesian + XGBoost).
        model_path: Path to a saved ``XGBoostScorer`` metadata JSON file.
            Required when ``scorer`` is ``"xgboost"`` or ``"ensemble"``.
        fetch_fn: Optional callable replacing ``fetch_lightcurve`` (for tests).
        clean_fn: Optional callable replacing ``clean_lightcurve`` (for tests).

    Returns:
        List of dicts, one per candidate signal, suitable for JSON output.
        When ``scorer`` is ``"xgboost"`` or ``"ensemble"``, each dict also
        contains ``"xgb_planet_probability"`` (and ``"ensemble_planet_probability"``
        for ``"ensemble"``).

    Raises:
        ValueError: If ``scorer`` is ``"xgboost"`` or ``"ensemble"`` but
            ``model_path`` is ``None``.
    """
    if scorer not in _VALID_SCORERS:
        raise ValueError(f"scorer must be one of {_VALID_SCORERS}, got {scorer!r}")

    _fetch = fetch_fn if fetch_fn is not None else fetch_lightcurve
    _clean = clean_fn if clean_fn is not None else clean_lightcurve

    # Load XGBoost scorer once before the per-signal loop.
    xgb_scorer = None
    if scorer in ("xgboost", "ensemble"):
        if model_path is None:
            raise ValueError(
                f"model_path is required when scorer='{scorer}'"
            )
        from exo_toolkit.ml.xgboost_scorer import XGBoostScorer
        xgb_scorer = XGBoostScorer.load(model_path)

    fetch_result = _fetch(target_id, mission)
    clean_result = _clean(fetch_result.light_curve)

    signals: list[CandidateSignal] = search_lightcurve(
        clean_result.light_curve,
        target_id=target_id,
        mission=mission,
        min_snr=min_snr,
        max_peaks=max_peaks,
    )

    if not signals:
        return []

    rows: list[dict[str, Any]] = []
    for signal in signals:
        vet_result = vet_signal(signal, clean_result.light_curve)
        posterior, scores = score_candidate(signal, vet_result.features)
        pathway = classify_submission_pathway(
            signal, vet_result.features, posterior, scores
        )

        row: dict[str, Any] = {
            "candidate_id": signal.candidate_id,
            "target_id": signal.target_id,
            "mission": signal.mission,
            "period_days": signal.period_days,
            "epoch_bjd": signal.epoch_bjd,
            "duration_hours": signal.duration_hours,
            "depth_ppm": signal.depth_ppm,
            "transit_count": signal.transit_count,
            "snr": signal.snr,
            "scorer": scorer,
            "posterior": {
                "planet_candidate": posterior.planet_candidate,
                "eclipsing_binary": posterior.eclipsing_binary,
                "background_eclipsing_binary": posterior.background_eclipsing_binary,
                "stellar_variability": posterior.stellar_variability,
                "instrumental_artifact": posterior.instrumental_artifact,
                "known_object": posterior.known_object,
            },
            "scores": {
                "false_positive_probability": scores.false_positive_probability,
                "detection_confidence": scores.detection_confidence,
                "novelty_score": scores.novelty_score,
            },
            "pathway": pathway,
        }

        if xgb_scorer is not None:
            xgb_prob = xgb_scorer.predict_proba(vet_result.features)
            row["xgb_planet_probability"] = xgb_prob
            if scorer == "ensemble":
                row["ensemble_planet_probability"] = (
                    0.5 * posterior.planet_candidate + 0.5 * xgb_prob
                )

        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _print_results(rows: list[dict[str, Any]], target_id: str) -> None:
    """Print pipeline results.  Key summary lines go via typer.echo so that
    CliRunner captures them; rich Tables are printed to a fresh Console."""
    if not rows:
        typer.echo(f"No transit candidates found above SNR threshold for {target_id}.")
        return

    typer.echo(f"{len(rows)} candidate signal(s) found for {target_id}")

    # Fresh Console so it binds to the current sys.stdout (important in tests).
    c = Console(highlight=False)
    for i, row in enumerate(rows, 1):
        table = Table(title=f"Signal {i}: {row['candidate_id']}", show_header=True)
        table.add_column("Property", style="dim")
        table.add_column("Value")

        table.add_row("Period", f"{row['period_days']:.4f} d")
        table.add_row("Depth", f"{row['depth_ppm']:.0f} ppm")
        table.add_row("Duration", f"{row['duration_hours']:.2f} h")
        table.add_row("Transits", str(row["transit_count"]))
        table.add_row("SNR", f"{row['snr']:.1f}")
        table.add_row("", "")
        table.add_row(
            "P(planet candidate)",
            f"{row['posterior']['planet_candidate']:.3f}",
        )
        table.add_row(
            "P(eclipsing binary)",
            f"{row['posterior']['eclipsing_binary']:.3f}",
        )
        if "xgb_planet_probability" in row:
            table.add_row(
                "XGBoost P(planet)",
                f"{row['xgb_planet_probability']:.3f}",
            )
        if "ensemble_planet_probability" in row:
            table.add_row(
                "Ensemble P(planet)",
                f"{row['ensemble_planet_probability']:.3f}",
            )
        table.add_row("FPP", f"{row['scores']['false_positive_probability']:.3f}")
        table.add_row(
            "Detection confidence",
            f"{row['scores']['detection_confidence']:.3f}",
        )
        table.add_row("Pathway", row["pathway"])
        c.print(table)
        typer.echo(f"  Pathway: {row['pathway']}")


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


@app.command()
def scan(
    target_id: str = typer.Argument(..., help='Target identifier, e.g. "TIC 150428135"'),
    mission: str = typer.Option("TESS", help="Mission: TESS, Kepler, or K2"),
    min_snr: float = typer.Option(5.0, "--min-snr", help="Minimum BLS SNR threshold"),
    max_peaks: int = typer.Option(5, "--max-peaks", help="Maximum signals to search for"),
    scorer: str = typer.Option(
        "bayesian",
        "--scorer",
        help="Scoring model: bayesian (default), xgboost, or ensemble",
    ),
    model_path: Path | None = typer.Option(
        None,
        "--model-path",
        help="Path to XGBoost model JSON (required for xgboost/ensemble scorers)",
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Write JSON results to this file"
    ),
) -> None:
    """Scan a target for exoplanet transit candidates.

    Runs the full pipeline: fetch → clean → search → vet → score → classify.
    """
    valid_missions = ("TESS", "Kepler", "K2")
    if mission not in valid_missions:
        typer.echo(f"Invalid mission '{mission}'. Choose from: {valid_missions}", err=True)
        raise typer.Exit(code=1)

    if scorer not in _VALID_SCORERS:
        typer.echo(
            f"Invalid scorer '{scorer}'. Choose from: {_VALID_SCORERS}", err=True
        )
        raise typer.Exit(code=1)

    if scorer in ("xgboost", "ensemble") and model_path is None:
        typer.echo(
            f"--model-path is required when --scorer={scorer}", err=True
        )
        raise typer.Exit(code=1)

    typer.echo(f"Scanning {target_id} ({mission}) [scorer={scorer}] ...")

    try:
        rows = run_pipeline(
            target_id,
            mission,  # type: ignore[arg-type]
            min_snr=min_snr,
            max_peaks=max_peaks,
            scorer=scorer,
            model_path=model_path,
        )
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"Pipeline error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    _print_results(rows, target_id)

    if output is not None:
        output.write_text(json.dumps(rows, indent=2))
        typer.echo(f"Results written to {output}")


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------


def main() -> None:
    app()


if __name__ == "__main__":
    main()
=======
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
>>>>>>> codex-background-automation-hardening
