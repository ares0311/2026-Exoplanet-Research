"""Polished shell entry points for the durable EXO-Hunter workflow."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

from rich.console import Console
from rich.table import Table

from exo_toolkit import __version__
from exo_toolkit.cli import run_pipeline
from exo_toolkit.hunter_models import (
    ArtifactIdentity,
    ExecutionProvenance,
    FollowUpRecommendation,
    HunterCandidate,
    SearchExecutionSummary,
    TargetExecutionResult,
)
from exo_toolkit.hunter_ranking import (
    FOLLOW_UP_CONFIDENCE_MIN,
    FOLLOW_UP_FPP_MAX,
    FOLLOW_UP_PATHWAYS,
    FOLLOW_UP_SELECTOR_VERSION,
    NEW_RANKING_WEIGHTS,
    NEW_SELECTOR_VERSION,
    OPERATOR_SELECTOR_VERSION,
    selection_contract,
)
from exo_toolkit.search_lifecycle import (
    HunterStore,
    format_eta,
)

DEFAULT_HUNTER_DB = Path("data/hunter_searches.sqlite3")
DEFAULT_MANIFEST_DIR = Path("reports/search_manifests")
DEFAULT_HISTORY_MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "data_selection"
    / "hunter_prior_search_history_v1.json"
)
DEFAULT_POOL_SIZE = 10_000
VALID_SCORERS = {"bayesian", "xgboost", "ensemble", "cnn", "full-ensemble"}
MAX_LIVE_WORKERS = 12


def _artifact_identity(path: Path, role: str) -> ArtifactIdentity:
    resolved = path.resolve()
    return ArtifactIdentity(
        role=role,
        path=str(path),
        sha256=hashlib.sha256(resolved.read_bytes()).hexdigest(),
        size_bytes=resolved.stat().st_size,
    )


def _repository_git_commit() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    if not commit:
        raise RuntimeError("Could not resolve the repository git commit")
    return commit


def _model_artifact_identities(
    *,
    scorer: str,
    model_path: Path | None,
    cnn_checkpoint: Path | None,
) -> tuple[ArtifactIdentity, ...]:
    artifacts: list[ArtifactIdentity] = []
    if model_path is not None:
        metadata = json.loads(model_path.read_text(encoding="utf-8"))
        model_file = metadata.get("model_file")
        if not isinstance(model_file, str) or not model_file:
            raise RuntimeError(f"XGBoost metadata is missing model_file: {model_path}")
        booster_path = model_path.parent / model_file
        if not booster_path.is_file():
            raise RuntimeError(f"Required XGBoost booster is missing: {booster_path}")
        artifacts.extend(
            (
                _artifact_identity(booster_path, "xgboost_model"),
                _artifact_identity(model_path, "xgboost_metadata"),
            )
        )
    if cnn_checkpoint is not None:
        artifacts.append(_artifact_identity(cnn_checkpoint, "cnn_checkpoint"))
    context_path = Path("models/candidate_context_v1.json")
    if scorer == "full-ensemble" and context_path.is_file():
        artifacts.append(_artifact_identity(context_path, "score_context"))
    return tuple(artifacts)


def _load_project_skill(module_name: str) -> ModuleType:
    """Load a repository Skill from installed console-script entry points."""
    repo_root = Path(__file__).resolve().parents[2]
    expected_path = repo_root / "Skills" / f"{module_name}.py"
    if not expected_path.is_file():
        raise RuntimeError(
            f"Required project Skill is missing: {expected_path}. "
            "Run EXO-Hunter from an editable checkout of this repository."
        )
    root_text = str(repo_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    try:
        return importlib.import_module(f"Skills.{module_name}")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Required project Skill Skills.{module_name} could not be imported: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _console(*, no_color: bool = False) -> Console:
    return Console(no_color=no_color, force_terminal=False if no_color else None)


def _parser_create() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="Create-New-Search",
        description="Rank candidates and durably create an exact pending EXO-Hunter search.",
    )
    parser.add_argument("--targets", type=int, required=True)
    parser.add_argument("--mode", choices=("new", "follow-up"), required=True)
    parser.add_argument("--db", type=Path, default=DEFAULT_HUNTER_DB)
    parser.add_argument("--candidate-file", type=Path)
    parser.add_argument("--history-manifest", type=Path, default=DEFAULT_HISTORY_MANIFEST)
    parser.add_argument("--pool-size", type=int)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--tmag-min", type=float, default=12.0)
    parser.add_argument("--tmag-max", type=float, default=14.5)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--no-color", action="store_true")
    return parser


def _parser_run() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="Run-New-Search",
        description="Execute the exact pending EXO-Hunter manifest, with safe resume.",
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_HUNTER_DB)
    parser.add_argument("--search-id")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--scorer", default="bayesian")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--cnn-checkpoint", type=Path)
    parser.add_argument("--pipeline", default="QLP")
    parser.add_argument("--exptime", default="long")
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--no-color", action="store_true")
    return parser


def _parser_follow_ups() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="Show-Follow-Ups",
        description="Show actionable EXO-Hunter follow-up recommendations.",
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_HUNTER_DB)
    parser.add_argument(
        "--status",
        choices=("open", "scheduled", "completed", "deferred", "all"),
        default="open",
    )
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--no-color", action="store_true")
    return parser


def _parser_import_follow_up() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="Import-Follow-Up",
        description="Durably import one reviewed prior result and recommendation.",
    )
    parser.add_argument("--evidence-file", type=Path, required=True)
    parser.add_argument("--db", type=Path, default=DEFAULT_HUNTER_DB)
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--no-color", action="store_true")
    return parser


def _load_reviewed_follow_up(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Reviewed evidence file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise RuntimeError("Reviewed evidence must be a schema_version=1 JSON object")
    required = {
        "source_search_id",
        "source_attempt_id",
        "completed_at",
        "source_files",
        "candidate",
        "recommendation",
        "source_result",
        "source_provenance",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise RuntimeError(f"Reviewed evidence is missing required fields: {missing}")
    source_files = payload["source_files"]
    if not isinstance(source_files, list) or not source_files:
        raise RuntimeError("Reviewed evidence source_files must be a non-empty list")
    for source in source_files:
        if not isinstance(source, dict) or set(source) != {"path", "sha256"}:
            raise RuntimeError("Each source file requires exactly path and sha256")
        source_path = Path(str(source["path"]))
        expected = str(source["sha256"])
        if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
            raise RuntimeError(f"Invalid SHA-256 for source file: {source_path}")
        if not source_path.is_file():
            raise RuntimeError(f"Reviewed evidence source is missing: {source_path}")
        actual = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(
                f"Reviewed evidence source hash mismatch: {source_path}; "
                f"expected={expected} actual={actual}"
            )
    payload["candidate"] = HunterCandidate.model_validate(payload["candidate"])
    payload["recommendation"] = FollowUpRecommendation.model_validate(
        payload["recommendation"]
    )
    payload["completed_at"] = datetime.fromisoformat(str(payload["completed_at"]))
    if payload["completed_at"].tzinfo is None:
        raise RuntimeError("Reviewed evidence completed_at must include a timezone")
    return payload


def _load_candidate_file(path: Path) -> list[HunterCandidate]:
    if not path.is_file():
        raise RuntimeError(f"Candidate file does not exist: {path}")
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            raw_rows: list[dict[str, Any]] = list(csv.DictReader(handle))
        for row in raw_rows:
            for field in ("aliases", "source_provenance", "metrics", "prior_searches"):
                if field in row and row[field]:
                    row[field] = json.loads(row[field])
            for field in ("ranking_score", "distance_pc", "estimated_download_gb"):
                if field in row and row[field] not in (None, ""):
                    row[field] = float(row[field])
            if "eligible" in row:
                row["eligible"] = str(row["eligible"]).lower() in {"1", "true", "yes"}
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("candidates") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise RuntimeError("Candidate JSON must be a list or an object with a candidates list")
        raw_rows = rows
    return [HunterCandidate.model_validate(row) for row in raw_rows]


def _storage_penalty(estimated_gb: float) -> float:
    if estimated_gb <= 5:
        return 0.0
    if estimated_gb <= 25:
        return 1.0
    if estimated_gb <= 100:
        return 2.0
    if estimated_gb <= 250:
        return 3.0
    return 5.0


def _select_live_new_candidates(
    *,
    targets: int,
    pool_size: int,
    workers: int,
    tmag_range: tuple[float, float],
    store: HunterStore,
    progress_fn: Callable[[str], None] | None = print,
) -> tuple[list[HunterCandidate], dict[str, Any]]:
    """Two-stage metadata-only TIC selection over a large reproducible pool."""
    scanner = _load_project_skill("star_scanner")
    excluded = set(store.searched_target_ids())
    excluded_tics = {
        int(value.split()[-1]) for value in excluded if value.upper().startswith("TIC ")
    }
    excluded_tics.update(scanner._load_toi_tic_ids(strict=True))
    excluded_tics.update(scanner._load_ctoi_tic_ids(strict=True))
    excluded_tics.update(scanner._load_confirmed_host_tic_ids(strict=True))
    search_log: dict[str, Any] = {}
    raw_targets = scanner.select_targets(
        pool_size,
        tmag_range=tmag_range,
        exclude_tic_ids=excluded_tics,
        full_sweep=True,
        max_workers=workers,
        search_log=search_log,
    )
    if len(raw_targets) < targets:
        raise RuntimeError(
            f"Wide TIC sweep returned {len(raw_targets)} candidates for {targets} requested targets"
        )

    stage_two_goal = min(len(raw_targets), max(targets * 3, targets))
    known_variables = scanner._load_asassn_variable_tic_ids(
        [row["tic_id"] for row in raw_targets[:stage_two_goal]], strict=True
    )

    inspected: dict[int, dict[str, Any]] = {}
    inspect_rows = [
        row for row in raw_targets[:stage_two_goal] if int(row["tic_id"]) not in known_variables
    ]
    inspection_started = time.monotonic()
    if progress_fn is not None:
        progress_fn(
            "EXO-Hunter metadata inspection: "
            f"targets={len(inspect_rows)} workers={workers} ETA=pending"
        )
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                scanner.inspect_target_products,
                row,
                mission="TESS",
                pipeline="QLP",
                exptime="long",
            ): row
            for row in inspect_rows
        }
        for completed, future in enumerate(as_completed(futures), 1):
            row = futures[future]
            try:
                inspected[int(row["tic_id"])] = future.result()
            except Exception as exc:  # noqa: BLE001
                inspected[int(row["tic_id"])] = {"inspection_error": f"{type(exc).__name__}: {exc}"}
            if progress_fn is not None:
                elapsed = time.monotonic() - inspection_started
                rate = completed / elapsed if elapsed > 0 else 0.0
                remaining = (len(inspect_rows) - completed) / rate if rate > 0 else float("inf")
                progress_fn(
                    f"  [{completed}/{len(inspect_rows)}] "
                    f"elapsed={elapsed:.0f}s ETA={format_eta(remaining)}"
                )

    candidates: list[HunterCandidate] = []
    for index, row in enumerate(raw_targets):
        tic_id = int(row["tic_id"])
        product = inspected.get(tic_id)
        is_known_variable = tic_id in known_variables
        product_count = len(product.get("products", ())) if product else 0
        estimated_gb = (
            float(product.get("total_bytes", 0)) / 1_000_000_000.0
            if product and product_count
            else None
        )
        advanced = index < stage_two_goal
        eligible = advanced and not is_known_variable and product_count > 0
        if is_known_variable:
            eligibility_reason = "excluded_known_asassn_variable"
        elif not advanced:
            eligibility_reason = "not_advanced_after_first_stage_rank"
        elif product and product.get("inspection_error"):
            eligibility_reason = f"product_metadata_failed:{product['inspection_error']}"
        elif not product_count:
            eligibility_reason = "no_qlp_light_curve_products"
        else:
            eligibility_reason = "eligible_with_qlp_products"

        base_priority = float(product.get("priority", row["priority"])) if product else float(
            row["priority"]
        )
        availability_score = min(product_count / 5.0, 1.0) if product_count else 0.0
        expected_information_gain = base_priority * availability_score
        cost_penalty = _storage_penalty(estimated_gb) if estimated_gb is not None else 5.0
        ranking_score = (
            NEW_RANKING_WEIGHTS["tic_priority"] * base_priority
            + NEW_RANKING_WEIGHTS["availability"] * availability_score
            + NEW_RANKING_WEIGHTS["expected_information_gain"]
            * expected_information_gain
            + NEW_RANKING_WEIGHTS["storage_cost_penalty"] * cost_penalty
        )
        candidates.append(
            HunterCandidate(
                target_id=f"TIC {tic_id}",
                canonical_id=f"TIC {tic_id}",
                aliases=(str(tic_id),),
                source="MAST TIC and QLP product metadata",
                source_provenance={
                    "search_category": "new",
                    "selector_version": NEW_SELECTOR_VERSION,
                    "pool_ordinal": index + 1,
                    "product_metadata": product or {},
                },
                eligible=eligible,
                eligibility_reason=eligibility_reason,
                estimated_download_gb=estimated_gb,
                ranking_score=ranking_score,
                selection_reason=(
                    "High deterministic TIC priority with verified QLP data availability, "
                    "low storage cost, novelty exclusions, and no prior EXO-Hunter search"
                ),
                metrics={
                    "tic_priority": base_priority,
                    "tmag": row.get("tmag"),
                    "teff_k": row.get("teff"),
                    "radius_rsun": row.get("radius_rsun"),
                    "contamination_ratio": row.get("contratio"),
                    "qlp_product_count": product_count,
                    "availability_score": availability_score,
                    "expected_information_gain": expected_information_gain,
                    "scientific_suitability": base_priority,
                    "storage_cost_penalty": cost_penalty,
                },
            )
        )
    search_log.update(
        {
            "candidate_universe_requested": pool_size,
            "candidate_universe_returned": len(raw_targets),
            "stage_two_metadata_count": stage_two_goal,
            "known_variables_excluded": len(known_variables),
            "stage_two_eligible_count": sum(row.eligible for row in candidates),
        }
    )
    return candidates, search_log


def _manifest_table(search: Mapping[str, Any]) -> Table:
    table = Table(title=f"Pending {search['mode']} search — {search['search_id']}")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Target", style="cyan")
    table.add_column("Class")
    table.add_column("Est. GB", justify="right")
    table.add_column("Score", justify="right", style="green")
    table.add_column("Reason")
    for row in search["targets"]:
        candidate = HunterCandidate.model_validate(row["candidate"])
        table.add_row(
            str(row["ordinal"]),
            candidate.canonical_id,
            candidate.object_classification,
            (
                "—"
                if candidate.estimated_download_gb is None
                else f"{candidate.estimated_download_gb:.4f}"
            ),
            f"{candidate.ranking_score:.3f}",
            candidate.selection_reason,
        )
    return table


def create_new_search(
    argv: Sequence[str] | None = None,
    *,
    live_selector: Callable[
        ..., tuple[list[HunterCandidate], dict[str, Any]]
    ] = _select_live_new_candidates,
) -> int:
    args = _parser_create().parse_args(argv)
    try:
        if args.targets < 1:
            raise ValueError("targets must be at least 1")
        if not 1 <= args.workers <= MAX_LIVE_WORKERS:
            raise ValueError(f"workers must be between 1 and {MAX_LIVE_WORKERS}")
        store = HunterStore(args.db)
        pool_size = max(args.pool_size or 0, DEFAULT_POOL_SIZE, args.targets * 100)
        if args.candidate_file:
            candidates = _load_candidate_file(args.candidate_file)
            selector_log = {
                "candidate_file": str(args.candidate_file),
                "candidate_universe_returned": len(candidates),
            }
            selector_version = OPERATOR_SELECTOR_VERSION
        elif args.mode == "follow-up":
            import_summary = store.import_history_manifest(args.history_manifest)
            candidates = store.follow_up_universe()
            selector_log = {
                "source": "durable_prior_search_history_and_follow_up_registry",
                "history_manifest": str(args.history_manifest),
                "history_import": import_summary,
                "candidate_universe_returned": len(candidates),
                "eligible_candidate_count": sum(row.eligible for row in candidates),
            }
            selector_version = FOLLOW_UP_SELECTOR_VERSION
        else:
            candidates, selector_log = live_selector(
                targets=args.targets,
                pool_size=pool_size,
                workers=args.workers,
                tmag_range=(args.tmag_min, args.tmag_max),
                store=store,
                progress_fn=lambda line: print(line, file=sys.stderr, flush=True),
            )
            selector_version = NEW_SELECTOR_VERSION
        contract = selection_contract(
            args.mode,
            operator_supplied=bool(args.candidate_file),
        )
        config = {
            "code_version": __version__,
            "workers": args.workers,
            "tmag_range": [args.tmag_min, args.tmag_max],
            "pool_size_requested": pool_size,
            "selector_log": selector_log,
            "selection_contract": contract,
        }
        search = store.create_search(
            candidates,
            requested_target_count=args.targets,
            mode=args.mode,
            selector_version=selector_version,
            config=config,
        )
        integrity = store.validity_summary(
            history_manifest=(
                args.history_manifest
                if args.mode == "follow-up" and not args.candidate_file
                else None
            )
        )
        if not integrity["ok"]:
            raise RuntimeError(f"Hunter database integrity failed: {integrity}")

        manifest_path: Path | None = None
        if len(search["targets"]) > 100:
            manifest_path = args.manifest_dir / f"{search['search_id']}.csv"
            store.export_manifest_csv(search["search_id"], manifest_path)
        shortfall = search["requested_target_count"] - search["selected_target_count"]
        payload = {
            "search_id": search["search_id"],
            "state": search["state"],
            "mode": search["mode"],
            "requested_targets": search["requested_target_count"],
            "selected_targets": search["selected_target_count"],
            "shortfall": shortfall,
            "candidate_pool_count": search["candidate_pool_count"],
            "manifest_sha256": search["manifest_sha256"],
            "manifest_csv": str(manifest_path) if manifest_path else None,
        }
        if args.json_output:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            console = _console(no_color=args.no_color)
            console.print(
                f"[green]Created exact pending search[/green] {search['search_id']} "
                f"from {search['candidate_pool_count']} frozen candidates."
            )
            if shortfall > 0:
                console.print(
                    f"[yellow]Returned the best available {search['selected_target_count']} "
                    f"of {search['requested_target_count']} requested targets[/yellow]: "
                    "fewer valid candidates exist after exploring the full evaluated pool."
                )
            if manifest_path:
                console.print(f"Manifest CSV: [cyan]{manifest_path}[/cyan]")
            else:
                console.print(_manifest_table(search))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"Create-New-Search failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


def _required_score(row: Mapping[str, Any], name: str) -> float:
    """Read one finite scorer value from the production or legacy row shape."""
    value = row.get(name)
    if value is None:
        scores = row.get("scores")
        if isinstance(scores, Mapping):
            value = scores.get(name)
    if value is None:
        raise RuntimeError(f"Pipeline scorer row is missing required score {name!r}")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Pipeline scorer row has invalid score {name!r}: {value!r}") from exc
    if not math.isfinite(numeric):
        raise RuntimeError(f"Pipeline scorer row has non-finite score {name!r}: {value!r}")
    return numeric


def _follow_up_from_row(row: Mapping[str, Any]) -> FollowUpRecommendation:
    """Build a follow-up recommendation for every detected candidate signal.

    Every row here already passed the full search/vet/score pipeline for a real
    detected transit-like signal; FPP/confidence/pathway are continuous evidence
    about that signal, not a gate on whether it exists. Rank (priority, driven by
    these same values) and absolute quality (meets_strict_production_bar) are
    reported separately so a request for N follow-up targets can still return the
    best available N even when none clear the strict production bar.
    """
    fpp = _required_score(row, "false_positive_probability")
    confidence = _required_score(row, "detection_confidence")
    pathway = row.get("pathway")
    meets_production_bar = bool(
        fpp < FOLLOW_UP_FPP_MAX
        and confidence > FOLLOW_UP_CONFIDENCE_MIN
        and pathway in FOLLOW_UP_PATHWAYS
    )
    priority = 100.0 * (1.0 - fpp) + 10.0 * confidence
    return FollowUpRecommendation(
        candidate_id=str(row.get("candidate_id", "unknown-candidate")),
        priority=priority,
        reason=(
            f"candidate signal: FPP={fpp:.4f}, confidence={confidence:.4f}, "
            f"pathway={pathway}; meets strict production bar={meets_production_bar}"
        ),
        evidence={**dict(row), "meets_strict_production_bar": meets_production_bar},
        recommended_action=(
            "Review phase-fold, centroid/contamination, odd-even, and secondary-eclipse "
            "evidence; obtain event-covering photometry if key diagnostics remain unavailable"
        ),
    )


def _pipeline_runner(
    *,
    scorer: str,
    model_path: Path | None,
    cnn_checkpoint: Path | None,
    pipeline: str,
    exptime: str,
    model_artifacts: tuple[ArtifactIdentity, ...] = (),
) -> Callable[[HunterCandidate], TargetExecutionResult]:
    def run(candidate: HunterCandidate) -> TargetExecutionResult:
        context: dict[str, Any] = {}

        def provenance(*, failure_stage: str | None = None) -> ExecutionProvenance:
            return ExecutionProvenance(
                candidate_snapshot=candidate,
                pipeline_context=context,
                code_version=__version__,
                git_commit=str(context.get("git_commit") or _repository_git_commit()),
                scorer=scorer,
                model_artifacts=model_artifacts,
                runner="exo_toolkit.cli.run_pipeline",
                failure_stage=failure_stage,
            )

        try:
            rows = run_pipeline(
                candidate.target_id,
                candidate.mission,
                pipeline=pipeline,
                exptime=exptime,
                scorer=scorer,
                model_path=model_path,
                cnn_checkpoint_path=cnn_checkpoint,
                run_context=context,
            )
        except Exception as exc:  # noqa: BLE001
            message = str(exc)
            no_data_prefixes = (
                "No TESS light curves",
                "No Kepler light curves",
                "No K2 light curves",
                "No JWST data",
                "No downloadable",
            )
            if not isinstance(exc, ValueError) or not message.startswith(no_data_prefixes):
                return TargetExecutionResult(
                    status="failed",
                    result={"target_id": candidate.target_id},
                    provenance=provenance(failure_stage="acquisition_or_pipeline"),
                    error_message=f"{type(exc).__name__}: {exc}",
                )
            return TargetExecutionResult(
                status="no_data",
                result={
                    "target_id": candidate.target_id,
                    "individual_scores": [],
                    "composite_interpretation": "No requested archive data were available",
                    "no_data_reason": message,
                },
                provenance=provenance(),
            )
        if not rows:
            return TargetExecutionResult(
                status="no_signal",
                result={
                    "target_id": candidate.target_id,
                    "individual_scores": [],
                    "composite_interpretation": "No signal passed the configured BLS threshold",
                },
                provenance=provenance(),
            )
        strongest = min(
            rows,
            key=lambda row: _required_score(row, "false_positive_probability"),
        )
        follow_ups = tuple(_follow_up_from_row(row) for row in rows)
        return TargetExecutionResult(
            status="candidate_found",
            result={
                "target_id": candidate.target_id,
                "individual_scores": rows,
                "composite_result": strongest,
                "composite_interpretation": (
                    "Candidate signal requires conservative false-positive review"
                ),
            },
            provenance=provenance(),
            follow_ups=follow_ups,
        )

    return run


def _write_run_report(
    summary: SearchExecutionSummary,
    db_path: Path = DEFAULT_HUNTER_DB,
) -> None:
    run_report = _load_project_skill("run_report")
    elapsed = (summary.completed_at - summary.started_at).total_seconds()
    report = run_report.RunReport(
        script="hunter_search",
        status="success" if summary.status == "completed" else summary.status,
        started_at=summary.started_at.isoformat(),
        completed_at=summary.completed_at.isoformat(),
        elapsed_seconds=elapsed,
        items_processed=summary.targets_processed,
        items_written=summary.targets_succeeded,
        items_failed=summary.targets_failed,
        output_paths=(str(db_path),),
        notes=f"search_id={summary.search_id}; attempt_id={summary.attempt_id}",
    )
    path = run_report.report_path_for("hunter_search")
    _commit_run_report(run_report, report, path)


def _commit_run_report(run_report: Any, report: Any, path: Path) -> bool:
    committed = bool(run_report.run_and_commit_report(report, path))
    if not committed:
        print(
            f"WARNING: Run Report data was preserved at {path}, but its git commit/push "
            "failed; push that exact report file manually.",
            file=sys.stderr,
            flush=True,
        )
    return committed


def _write_follow_up_import_report(
    result: Mapping[str, Any],
    db_path: Path,
    started_at: datetime,
    completed_at: datetime,
) -> None:
    run_report = _load_project_skill("run_report")
    report = run_report.RunReport(
        script="hunter_followup_import",
        status="success",
        started_at=started_at.isoformat(),
        completed_at=completed_at.isoformat(),
        elapsed_seconds=(completed_at - started_at).total_seconds(),
        items_processed=1,
        items_written=int(bool(result["created"])),
        items_failed=0,
        output_paths=(str(db_path),),
        notes=(
            f"search_id={result['search_id']}; follow_up_id={result['follow_up_id']}; "
            f"created={str(bool(result['created'])).lower()}"
        ),
    )
    _commit_run_report(
        run_report,
        report,
        run_report.report_path_for("hunter_followup_import"),
    )


def run_new_search(
    argv: Sequence[str] | None = None,
    *,
    runner_factory: Callable[
        ..., Callable[[HunterCandidate], TargetExecutionResult]
    ] = _pipeline_runner,
    report_fn: Callable[[SearchExecutionSummary], None] | None = None,
) -> int:
    args = _parser_run().parse_args(argv)
    try:
        if not 1 <= args.workers <= MAX_LIVE_WORKERS:
            raise ValueError(f"workers must be between 1 and {MAX_LIVE_WORKERS}")
        if args.scorer not in VALID_SCORERS:
            raise ValueError(
                f"scorer must be one of {sorted(VALID_SCORERS)}, got {args.scorer!r}"
            )
        model_path = args.model_path
        if args.scorer in {"xgboost", "ensemble", "full-ensemble"} and model_path is None:
            model_path = Path("models/xgboost_toi.json")
        cnn_checkpoint = args.cnn_checkpoint
        if args.scorer in {"cnn", "full-ensemble"} and cnn_checkpoint is None:
            cnn_checkpoint = Path("models/cnn/benchmark_cnn_v1/best.pt")
        if model_path is not None and not model_path.is_file():
            raise RuntimeError(f"Required scorer model is missing: {model_path}")
        if cnn_checkpoint is not None and not cnn_checkpoint.is_file():
            raise RuntimeError(f"Required CNN checkpoint is missing: {cnn_checkpoint}")
        git_commit = _repository_git_commit()
        model_artifacts = _model_artifact_identities(
            scorer=args.scorer,
            model_path=model_path,
            cnn_checkpoint=cnn_checkpoint,
        )
        runner = runner_factory(
            scorer=args.scorer,
            model_path=model_path,
            cnn_checkpoint=cnn_checkpoint,
            pipeline=args.pipeline,
            exptime=args.exptime,
            model_artifacts=model_artifacts,
        )
        store = HunterStore(args.db)
        if not args.json_output:
            print(
                f"Run-New-Search: workers={args.workers} scorer={args.scorer} "
                f"pipeline={args.pipeline} resume=true",
                flush=True,
            )
        summary = store.execute_search(
            runner,
            search_id=args.search_id,
            workers=args.workers,
            run_config={
                "code_version": __version__,
                "git_commit": git_commit,
                "scorer": args.scorer,
                "pipeline": args.pipeline,
                "exptime": args.exptime,
                "workers": args.workers,
                "model_artifacts": [
                    artifact.model_dump(mode="json") for artifact in model_artifacts
                ],
            },
            progress_fn=None if args.json_output else lambda line: print(line, flush=True),
        )
        if not store.integrity_summary()["ok"]:
            raise RuntimeError("Hunter database integrity failed after execution")
        if report_fn is None:
            _write_run_report(summary, args.db)
        else:
            report_fn(summary)
        payload = summary.model_dump(mode="json")
        if args.json_output:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            console = _console(no_color=args.no_color)
            style = "green" if summary.status == "completed" else "yellow"
            console.print(
                f"[{style}]Search {summary.status}[/{style}]: "
                f"{summary.targets_succeeded} succeeded, {summary.targets_failed} failed, "
                f"{summary.follow_ups_registered} follow-ups registered."
            )
        return 0 if summary.status == "completed" else 2
    except Exception as exc:  # noqa: BLE001
        print(f"Run-New-Search failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


def show_follow_ups(argv: Sequence[str] | None = None) -> int:
    args = _parser_follow_ups().parse_args(argv)
    try:
        store = HunterStore(args.db)
        rows = store.list_follow_ups(status=args.status)
        if args.json_output:
            print(json.dumps({"follow_ups": rows}, indent=2, sort_keys=True))
            return 0
        console = _console(no_color=args.no_color)
        if not rows:
            console.print("[dim]No follow-ups match the requested status.[/dim]")
            return 0
        table = Table(title=f"EXO-Hunter follow-ups — {args.status}")
        table.add_column("Priority", justify="right", style="green")
        table.add_column("Target", style="cyan")
        table.add_column("Status")
        table.add_column("Search ready?", justify="center")
        table.add_column("Evidence / reason", overflow="fold")
        table.add_column("Prior search", overflow="fold")
        table.add_column("Recommended next action", overflow="fold")
        provenance_details: list[str] = []
        for row in rows:
            prior_searches = row["prior_search_provenance"]
            if prior_searches:
                prior = prior_searches[-1]
                prior_label = (
                    f"{prior['source_project']} — {prior['searched_by']} — "
                    f"{prior['searched_at']}"
                )
                for prior_index, prior_entry in enumerate(prior_searches, 1):
                    provenance_details.append(
                        f"{row['target_id']} [{prior_index}/{len(prior_searches)}]: "
                        f"{prior_entry['source_project']} — {prior_entry['searched_by']} — "
                        f"{prior_entry['searched_at']}; "
                        f"method/data={prior_entry['method_or_data']}; "
                        f"result={prior_entry['result']}; "
                        f"provenance={prior_entry['provenance_uri']}"
                    )
            else:
                prior_label = f"EXO-Hunter search {row['search_id']}"
                provenance_details.append(f"{row['target_id']}: {prior_label}")
            table.add_row(
                f"{float(row['priority']):.2f}",
                str(row["target_id"]),
                str(row["status"]),
                "yes" if bool(row["search_eligible"]) else "no",
                str(row["reason"]),
                prior_label,
                str(row["recommended_action"]),
            )
        console.print(table)
        console.print("[bold]Prior-search provenance[/bold]")
        for detail in provenance_details:
            console.print(detail)
        deferred = [
            row
            for row in rows
            if not bool(row["search_eligible"]) and row.get("revisit_reason")
        ]
        if deferred:
            console.print("[bold]Revisit gates[/bold]")
            for row in deferred:
                console.print(f"{row['target_id']}: {row['revisit_reason']}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"Show-Follow-Ups failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


def import_follow_up(
    argv: Sequence[str] | None = None,
    *,
    report_fn: Callable[[Mapping[str, Any], Path, datetime, datetime], None] | None = None,
) -> int:
    args = _parser_import_follow_up().parse_args(argv)
    started_at = datetime.now(UTC)
    try:
        evidence = _load_reviewed_follow_up(args.evidence_file)
        progress_stream = sys.stderr if args.json_output else sys.stdout
        print(
            "Import-Follow-Up: validating 1 reviewed result",
            file=progress_stream,
            flush=True,
        )
        store = HunterStore(args.db)
        result = store.import_reviewed_follow_up(
            candidate=evidence["candidate"],
            recommendation=evidence["recommendation"],
            source_search_id=str(evidence["source_search_id"]),
            source_attempt_id=str(evidence["source_attempt_id"]),
            source_result=evidence["source_result"],
            source_provenance=evidence["source_provenance"],
            completed_at=evidence["completed_at"],
        )
        if not store.integrity_summary()["ok"]:
            raise RuntimeError("Hunter database integrity failed after follow-up import")
        elapsed = (datetime.now(UTC) - started_at).total_seconds()
        print(
            f"[1/1] status=success elapsed={elapsed:.0f}s ETA=0s",
            file=progress_stream,
            flush=True,
        )
        completed_at = datetime.now(UTC)
        if report_fn is None:
            _write_follow_up_import_report(result, args.db, started_at, completed_at)
        else:
            report_fn(result, args.db, started_at, completed_at)
        if args.json_output:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            console = _console(no_color=args.no_color)
            verb = "Imported" if result["created"] else "Already imported"
            console.print(
                f"[green]{verb} reviewed follow-up[/green] {result['follow_up_id']} "
                f"under {result['search_id']}."
            )
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"Import-Follow-Up failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


def create_new_search_entry() -> None:
    raise SystemExit(create_new_search())


def run_new_search_entry() -> None:
    raise SystemExit(run_new_search())


def show_follow_ups_entry() -> None:
    raise SystemExit(show_follow_ups())


def import_follow_up_entry() -> None:
    raise SystemExit(import_follow_up())
