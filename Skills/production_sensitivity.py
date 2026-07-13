"""Measure production-pipeline transit sensitivity on cached real backgrounds.

The runner is cache-only: it reads existing Kepler FITS products, injects a
versioned transit grid, executes the full production pipeline, writes recovery
curves, and records a structured Run Report.  It never downloads data.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from exo_toolkit import __version__  # noqa: E402
from exo_toolkit.cli import run_pipeline  # noqa: E402
from exo_toolkit.fetch import FetchProvenance, FetchResult  # noqa: E402
from Skills.run_report import (  # noqa: E402
    RunReport,
    report_path_for,
    run_and_commit_report,
)


@dataclass(frozen=True)
class Trial:
    """One deterministic injection on one real background."""

    index: int
    target_id: str
    background_label: str
    period_days: float
    depth_ppm: float
    duration_hours: float
    epoch_phase: float
    scenario_id: str = "periodic_grid"
    injection_type: str = "periodic"
    ttv_amplitude_hours: float = 0.0
    ttv_cycle_transits: int = 3
    gap_transit_index: int | None = None
    gap_duration_hours: float = 0.0
    variability_amplitude_ppm: float = 0.0
    variability_period_days: float = 1.0


def build_trials(
    config: dict[str, Any], *, shard_index: int = 0, shard_count: int = 1
) -> list[Trial]:
    """Expand the configured grid and return this process shard."""
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("require shard_count >= 1 and 0 <= shard_index < shard_count")
    trials: list[Trial] = []
    index = 0
    scenarios = config.get("injection_scenarios")
    if scenarios is not None:
        for background in config["backgrounds"]:
            for scenario in scenarios:
                trial = Trial(
                    index=index,
                    target_id=str(background["target_id"]),
                    background_label=str(background["background_label"]),
                    period_days=float(scenario["period_days"]),
                    depth_ppm=float(scenario["depth_ppm"]),
                    duration_hours=float(scenario["duration_hours"]),
                    epoch_phase=float(scenario.get("epoch_phase", 0.37)),
                    scenario_id=str(scenario["scenario_id"]),
                    injection_type=str(scenario["injection_type"]),
                    ttv_amplitude_hours=float(scenario.get("ttv_amplitude_hours", 0.0)),
                    ttv_cycle_transits=int(scenario.get("ttv_cycle_transits", 3)),
                    gap_transit_index=(
                        int(scenario["gap_transit_index"])
                        if scenario.get("gap_transit_index") is not None
                        else None
                    ),
                    gap_duration_hours=float(scenario.get("gap_duration_hours", 0.0)),
                    variability_amplitude_ppm=float(
                        scenario.get("variability_amplitude_ppm", 0.0)
                    ),
                    variability_period_days=float(
                        scenario.get("variability_period_days", 1.0)
                    ),
                )
                if index % shard_count == shard_index:
                    trials.append(trial)
                index += 1
        return trials

    grid = config["injection_grid"]
    for background in config["backgrounds"]:
        for period in grid["period_days"]:
            for depth in grid["depth_ppm"]:
                for duration in grid["duration_hours"]:
                    trial = Trial(
                        index=index,
                        target_id=str(background["target_id"]),
                        background_label=str(background["background_label"]),
                        period_days=float(period),
                        depth_ppm=float(depth),
                        duration_hours=float(duration),
                        epoch_phase=float(grid["epoch_phase"]),
                    )
                    if index % shard_count == shard_index:
                        trials.append(trial)
                    index += 1
    return trials


def inject_flux(
    time_bjd: np.ndarray,
    flux: np.ndarray,
    *,
    period_days: float,
    epoch_bjd: float,
    duration_hours: float,
    depth_ppm: float,
) -> np.ndarray:
    """Return a copied flux array with a periodic box transit injected."""
    half_duration_days = duration_hours / 48.0
    phase_days = ((time_bjd - epoch_bjd + period_days / 2.0) % period_days) - (
        period_days / 2.0
    )
    result = np.asarray(flux, dtype=float).copy()
    result[np.abs(phase_days) <= half_duration_days] *= 1.0 - depth_ppm / 1_000_000.0
    return result


def _inject_at_centers(
    time_bjd: np.ndarray,
    flux: np.ndarray,
    centers_bjd: list[float],
    *,
    duration_hours: float,
    depth_ppm: float,
) -> np.ndarray:
    result = np.asarray(flux, dtype=float).copy()
    half_duration_days = duration_hours / 48.0
    for center in centers_bjd:
        in_transit = np.abs(time_bjd - center) <= half_duration_days
        result[in_transit] *= 1.0 - depth_ppm / 1_000_000.0
    return result


def inject_trial_arrays(
    time_bjd: np.ndarray, flux: np.ndarray, trial: Trial
) -> tuple[np.ndarray, np.ndarray, float]:
    """Inject one configured scenario and return flux, cadence mask, and epoch."""
    finite_time = time_bjd[np.isfinite(time_bjd)]
    if len(finite_time) == 0:
        raise ValueError(f"background {trial.target_id} has no finite timestamps")
    epoch_bjd = float(finite_time[0] + trial.epoch_phase * trial.period_days)
    result = np.asarray(flux, dtype=float).copy()

    if trial.variability_amplitude_ppm:
        if trial.variability_period_days <= 0:
            raise ValueError("variability_period_days must be positive")
        phase = 2.0 * np.pi * (time_bjd - finite_time[0]) / trial.variability_period_days
        result *= 1.0 + trial.variability_amplitude_ppm / 1_000_000.0 * np.sin(phase)

    if trial.injection_type == "single_transit":
        centers = [epoch_bjd]
    elif trial.injection_type == "ttv":
        if trial.ttv_cycle_transits < 2:
            raise ValueError("ttv_cycle_transits must be at least 2")
        n_min = math.floor((float(finite_time[0]) - epoch_bjd) / trial.period_days) - 1
        n_max = math.ceil((float(finite_time[-1]) - epoch_bjd) / trial.period_days) + 1
        centers = [
            epoch_bjd
            + n * trial.period_days
            + trial.ttv_amplitude_hours
            / 24.0
            * math.sin(2.0 * math.pi * n / trial.ttv_cycle_transits)
            for n in range(n_min, n_max + 1)
        ]
    elif trial.injection_type in {"periodic", "data_gap", "stellar_variability"}:
        centers = []
    else:
        raise ValueError(f"unsupported injection_type {trial.injection_type!r}")

    if centers:
        result = _inject_at_centers(
            time_bjd,
            result,
            centers,
            duration_hours=trial.duration_hours,
            depth_ppm=trial.depth_ppm,
        )
    else:
        result = inject_flux(
            time_bjd,
            result,
            period_days=trial.period_days,
            epoch_bjd=epoch_bjd,
            duration_hours=trial.duration_hours,
            depth_ppm=trial.depth_ppm,
        )

    keep = np.ones(len(time_bjd), dtype=bool)
    if trial.gap_transit_index is not None and trial.gap_duration_hours > 0:
        gap_center = epoch_bjd + trial.gap_transit_index * trial.period_days
        keep &= np.abs(time_bjd - gap_center) > trial.gap_duration_hours / 48.0
    return result, keep, epoch_bjd


def inject_lightcurve(light_curve: Any, trial: Trial) -> tuple[Any, float]:
    """Copy a Lightkurve object and inject one configured transit signal."""
    time_bjd = np.asarray(light_curve.time.jd, dtype=float)
    flux = np.asarray(light_curve.flux.value, dtype=float)
    injected, keep, epoch_bjd = inject_trial_arrays(time_bjd, flux, trial)
    result = light_curve.copy()
    unit = getattr(light_curve.flux, "unit", None)
    result.flux = injected if unit is None else injected * unit
    if not np.all(keep):
        result = result[keep]
    return result, epoch_bjd


def _cache_glob(target_id: str) -> str:
    digits = int(target_id.removeprefix("KIC ").strip())
    prefix = f"kplr{digits:09d}"
    return f"{prefix}_lc_*/{prefix}-*_llc.fits"


def select_quarter_products(
    paths: list[Path], curves: list[Any], requested_quarters: set[int]
) -> tuple[list[Path], list[Any]]:
    """Filter paths and curves together so provenance matches actual inputs."""
    if len(paths) != len(curves):
        raise ValueError("cached product paths and loaded curves must align")
    if not requested_quarters:
        return paths, curves
    selected = [
        (path, curve)
        for path, curve in zip(paths, curves, strict=True)
        if int(curve.meta["QUARTER"]) in requested_quarters
    ]
    return [path for path, _ in selected], [curve for _, curve in selected]


def load_cached_background(
    background: dict[str, Any], cache_root: Path
) -> tuple[Any, FetchProvenance]:
    """Load and stitch one Kepler target strictly from local FITS cache."""
    import lightkurve as lk

    target_id = str(background["target_id"])
    paths = sorted(cache_root.glob(_cache_glob(target_id)))
    if not paths:
        raise FileNotFoundError(
            f"no cached Kepler FITS products for {target_id} under {cache_root}"
        )
    requested_quarters = {int(value) for value in background.get("quarters", [])}
    curves = [lk.read(path) for path in paths]
    paths, curves = select_quarter_products(paths, curves, requested_quarters)
    if not curves:
        raise FileNotFoundError(
            f"cached products for {target_id} do not include quarters "
            f"{sorted(requested_quarters)}"
        )
    stitched = lk.LightCurveCollection(curves).stitch(corrector_func=None)
    max_baseline_days = background.get("max_baseline_days")
    if max_baseline_days is not None:
        all_times = np.asarray(stitched.time.jd, dtype=float)
        finite_times = all_times[np.isfinite(all_times)]
        if len(finite_times) == 0:
            raise ValueError(f"background {target_id} has no finite timestamps")
        keep = all_times <= float(finite_times[0] + float(max_baseline_days))
        stitched = stitched[keep]
    time_bjd = np.asarray(stitched.time.jd, dtype=float)
    finite = time_bjd[np.isfinite(time_bjd)]
    cadence_seconds = float(np.nanmedian(np.diff(finite)) * 86_400.0)
    quarters = tuple(sorted({int(curve.meta["QUARTER"]) for curve in curves}))
    raw_uris = tuple(f"cache://{path.relative_to(cache_root)}" for path in paths)
    provenance = FetchProvenance(
        target_id=target_id,
        mission="Kepler",
        sectors_or_quarters=quarters,
        cadence_seconds=cadence_seconds,
        pipeline="Kepler-cache-only",
        flux_column=str(stitched.meta.get("FLUX_ORIGIN", "pdcsap_flux")),
        n_cadences=len(stitched),
        time_baseline_days=float(finite[-1] - finite[0]),
        fetched_at=datetime.now(UTC).isoformat(),
        raw_uris=raw_uris,
    )
    return stitched, provenance


def _period_matches(observed: float, injected: float, tolerance: float) -> bool:
    return any(
        abs(observed - expected) / expected <= tolerance
        for expected in (injected, injected / 2.0, injected * 2.0)
    )


def match_recovery(
    rows: list[dict[str, Any]], injected_period: float, *, tolerance: float = 0.02
) -> dict[str, Any] | None:
    """Return the closest exact/half/double-period recovery, if present."""
    matching = [
        row
        for row in rows
        if isinstance(row.get("period_days"), (int, float))
        and _period_matches(float(row["period_days"]), injected_period, tolerance)
    ]
    if not matching:
        return None
    return min(
        matching,
        key=lambda row: min(
            abs(float(row["period_days"]) - expected) / expected
            for expected in (injected_period, injected_period / 2.0, injected_period * 2.0)
        ),
    )


def match_single_event_recovery(
    rows: list[dict[str, Any]], injected_epoch_bjd: float, duration_hours: float
) -> dict[str, Any] | None:
    """Match a single event by overlap with any recovered periodic ephemeris."""
    matches: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        period = row.get("period_days")
        epoch = row.get("epoch_bjd")
        recovered_duration = row.get("duration_hours")
        if (
            not all(isinstance(value, (int, float)) for value in (period, epoch))
            or float(period) <= 0
        ):
            continue
        phase_distance = abs(
            ((injected_epoch_bjd - float(epoch) + float(period) / 2.0) % float(period))
            - float(period) / 2.0
        )
        tolerance_days = max(
            duration_hours,
            float(recovered_duration) if isinstance(recovered_duration, (int, float)) else 0.0,
        ) / 24.0
        if phase_distance <= tolerance_days:
            matches.append((phase_distance, row))
    return min(matches, key=lambda item: item[0])[1] if matches else None


PipelineFn = Callable[..., list[dict[str, Any]]]


def execute_trial(
    trial: Trial,
    *,
    background_lc: Any,
    provenance: FetchProvenance,
    config: dict[str, Any],
    pipeline_fn: PipelineFn = run_pipeline,
) -> dict[str, Any]:
    """Inject one signal, execute the full pipeline, and capture recovery metadata."""
    injected_lc, epoch_bjd = inject_lightcurve(background_lc, trial)

    def fetch_fn(_target_id: str, _mission: str, **_kwargs: Any) -> FetchResult:
        return FetchResult(light_curve=injected_lc, provenance=provenance)

    pipeline = config["pipeline"]
    rows = pipeline_fn(
        trial.target_id,
        "Kepler",
        period_min=float(pipeline["period_min"]),
        period_max=float(pipeline["period_max"]),
        duration_min_hours=float(pipeline["duration_min_hours"]),
        duration_max_hours=float(pipeline["duration_max_hours"]),
        n_durations=int(pipeline["n_durations"]),
        min_snr=float(pipeline["min_snr"]),
        max_peaks=int(pipeline["max_peaks"]),
        max_period_grid_points=int(pipeline["max_period_grid_points"]),
        scorer="full-ensemble",
        model_path=REPO_ROOT / str(pipeline["xgboost_model_path"]),
        cnn_checkpoint_path=REPO_ROOT / str(pipeline["cnn_checkpoint_path"]),
        allow_cross_mission_cnn=True,
        fetch_fn=fetch_fn,
    )
    if trial.injection_type == "single_transit":
        recovered = match_single_event_recovery(rows, epoch_bjd, trial.duration_hours)
        recovery_basis = "event_time_overlap"
    else:
        recovered = match_recovery(
            rows,
            trial.period_days,
            tolerance=float(config["recovery_period_relative_tolerance"]),
        )
        recovery_basis = "period_exact_half_double"
    return {
        "trial_index": trial.index,
        "target_id": trial.target_id,
        "background_label": trial.background_label,
        "scenario_id": trial.scenario_id,
        "injection_type": trial.injection_type,
        "period_days": trial.period_days,
        "depth_ppm": trial.depth_ppm,
        "duration_hours": trial.duration_hours,
        "epoch_bjd": epoch_bjd,
        "ttv_amplitude_hours": trial.ttv_amplitude_hours,
        "ttv_cycle_transits": trial.ttv_cycle_transits,
        "gap_transit_index": trial.gap_transit_index,
        "gap_duration_hours": trial.gap_duration_hours,
        "source_cadences": provenance.n_cadences,
        "injected_cadences": len(injected_lc),
        "cadences_removed": provenance.n_cadences - len(injected_lc),
        "variability_amplitude_ppm": trial.variability_amplitude_ppm,
        "variability_period_days": trial.variability_period_days,
        "recovery_basis": recovery_basis,
        "recovered": recovered is not None,
        "candidate_count": len(rows),
        "recovered_period_days": recovered.get("period_days") if recovered else None,
        "recovered_depth_ppm": recovered.get("depth_ppm") if recovered else None,
        "recovered_snr": recovered.get("snr") if recovered else None,
        "cnn_planet_probability": (
            recovered.get("cnn_planet_probability") if recovered else None
        ),
        "full_ensemble_planet_probability": (
            recovered.get("full_ensemble_planet_probability") if recovered else None
        ),
        "false_positive_probability": (
            recovered.get("scores", {}).get("false_positive_probability")
            if recovered
            else None
        ),
        "pathway": recovered.get("pathway") if recovered else None,
    }


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def summarize_curves(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate sample results into period/depth/duration recovery curves."""
    curves: dict[str, list[dict[str, Any]]] = {}
    for field in (
        "scenario_id",
        "injection_type",
        "period_days",
        "depth_ppm",
        "duration_hours",
        "background_label",
    ):
        points: list[dict[str, Any]] = []
        for value in sorted({result[field] for result in results}):
            group = [result for result in results if result[field] == value]
            recovered = [result for result in group if result["recovered"]]
            points.append(
                {
                    field: value,
                    "trials": len(group),
                    "recovered": len(recovered),
                    "recovery_rate": len(recovered) / len(group),
                    "mean_cnn_planet_probability": _mean(
                        [
                            float(item["cnn_planet_probability"])
                            for item in recovered
                            if isinstance(item["cnn_planet_probability"], (int, float))
                        ]
                    ),
                    "mean_full_ensemble_planet_probability": _mean(
                        [
                            float(item["full_ensemble_planet_probability"])
                            for item in recovered
                            if isinstance(
                                item["full_ensemble_planet_probability"], (int, float)
                            )
                        ]
                    ),
                }
            )
        curves[f"by_{field}"] = points
    recovered_count = sum(bool(result["recovered"]) for result in results)
    return {
        "overall": {
            "trials": len(results),
            "recovered": recovered_count,
            "recovery_rate": recovered_count / len(results) if results else 0.0,
        },
        **curves,
    }


def scoped_output_path(path: Path, *, shard_index: int, shard_count: int) -> Path:
    """Return a collision-free output path for process shards."""
    if shard_count <= 1:
        return path
    return path.with_name(f"{path.stem}.shard{shard_index}of{shard_count}{path.suffix}")


BackgroundLoader = Callable[[dict[str, Any], Path], tuple[Any, FetchProvenance]]
ReportFn = Callable[..., bool]


def run_sensitivity(
    config: dict[str, Any],
    *,
    output_path: Path,
    cache_root: Path,
    workers: int,
    shard_index: int = 0,
    shard_count: int = 1,
    background_loader: BackgroundLoader = load_cached_background,
    pipeline_fn: PipelineFn = run_pipeline,
    report_fn: ReportFn = run_and_commit_report,
) -> dict[str, Any]:
    """Execute one bounded sensitivity shard and write its report artifact."""
    started_at = datetime.now(UTC)
    started = time.monotonic()
    trials = build_trials(config, shard_index=shard_index, shard_count=shard_count)
    if not trials:
        raise ValueError("selected shard contains no trials")
    print(
        "Production sensitivity startup: "
        f"trials={len(trials)} workers={workers} shard={shard_index}/{shard_count} "
        f"backgrounds={len(config['backgrounds'])} scorer=full-ensemble",
        flush=True,
    )

    background_by_target: dict[str, tuple[Any, FetchProvenance]] = {}
    for background in config["backgrounds"]:
        target_id = str(background["target_id"])
        if any(trial.target_id == target_id for trial in trials):
            background_by_target[target_id] = background_loader(background, cache_root)

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    heartbeat_seconds = float(config.get("heartbeat_seconds", 30.0))
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        future_to_trial: dict[Future[dict[str, Any]], Trial] = {
            executor.submit(
                execute_trial,
                trial,
                background_lc=background_by_target[trial.target_id][0],
                provenance=background_by_target[trial.target_id][1],
                config=config,
                pipeline_fn=pipeline_fn,
            ): trial
            for trial in trials
        }
        pending = set(future_to_trial)
        completed = 0
        while pending:
            done, pending = wait(
                pending, timeout=heartbeat_seconds, return_when=FIRST_COMPLETED
            )
            elapsed = time.monotonic() - started
            if not done:
                rate = completed / elapsed if completed else 0.0
                eta = (len(trials) - completed) / rate if rate else math.inf
                print(
                    f"  heartbeat completed={completed} active="
                    f"{min(workers, len(pending))} pending={len(pending)} "
                    f"elapsed={elapsed:.1f}s ETA={eta:.1f}s",
                    flush=True,
                )
                continue
            for future in done:
                completed += 1
                trial = future_to_trial[future]
                try:
                    results.append(future.result())
                    status = "recovered" if results[-1]["recovered"] else "missed"
                except Exception as exc:  # noqa: BLE001
                    failures.append(
                        {
                            "trial_index": trial.index,
                            "target_id": trial.target_id,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    status = "failed"
                elapsed = time.monotonic() - started
                rate = completed / elapsed if elapsed else 0.0
                eta = (len(trials) - completed) / rate if rate else math.inf
                print(
                    f"  [{completed}/{len(trials)}] trial={trial.index} {status} "
                    f"elapsed={elapsed:.1f}s ETA={eta:.1f}s",
                    flush=True,
                )

    results.sort(key=lambda item: int(item["trial_index"]))
    output_path = scoped_output_path(
        output_path, shard_index=shard_index, shard_count=shard_count
    )
    completed_at = datetime.now(UTC)
    elapsed = time.monotonic() - started
    payload = {
        "schema_version": 1,
        "suite_id": config["suite_id"],
        "toolkit_version": __version__,
        "source_dataset_id": config["source_dataset_id"],
        "model_ids": config["model_ids"],
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "elapsed_seconds": elapsed,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "cache_only": True,
        "sensitivity_scope": config.get("sensitivity_scope", {}),
        "backgrounds": [
            provenance.model_dump(mode="json")
            for _, provenance in background_by_target.values()
        ],
        "curves": summarize_curves(results),
        "results": results,
        "failures": failures,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    report = RunReport(
        script="production_sensitivity",
        status="success" if not failures else "partial",
        started_at=started_at.isoformat(),
        completed_at=completed_at.isoformat(),
        elapsed_seconds=elapsed,
        items_processed=len(results) + len(failures),
        items_written=len(results),
        items_failed=len(failures),
        output_paths=(str(output_path),),
        shard_index=shard_index,
        shard_count=shard_count,
        notes="cache-only real-background full-pipeline injection recovery",
    )
    report_path = report_path_for(
        "production_sensitivity", shard_index=shard_index, shard_count=shard_count
    )
    if not report_fn(report, report_path):
        print(f"WARNING: Run Report push failed for {report_path}", flush=True)
    print(
        f"Production sensitivity {'COMPLETE' if not failures else 'PARTIAL'}: "
        f"recovered={sum(bool(item['recovered']) for item in results)}/{len(results)} "
        f"failed={len(failures)} elapsed={elapsed:.1f}s output={output_path}",
        flush=True,
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path.home() / ".lightkurve" / "cache" / "mastDownload" / "Kepler",
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    payload = run_sensitivity(
        config,
        output_path=args.output,
        cache_root=args.cache_root,
        workers=args.workers,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    return 0 if not payload["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
