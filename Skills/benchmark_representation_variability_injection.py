"""Compare blind BLS recovery with frozen embedding shifts on variable stars.

The benchmark is cache-only, owns one TIC-modulo shard, persists no arrays, and
keeps every ASAS-SN row training-disabled. Aggregate mode reconciles all six
shards and reports descriptive paired metrics without promoting a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import numpy as np  # noqa: E402
from astropy.io import fits  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from exo_toolkit.search import search_lightcurve  # noqa: E402
from Skills.injection_recovery import _MockLC  # noqa: E402
from Skills.production_sensitivity import inject_flux, match_recovery  # noqa: E402
from Skills.run_report import RunReport, report_path_for, run_and_commit_report  # noqa: E402

DEFAULT_CONTRACT = REPO_ROOT / "metadata/representation_variability_injection_contract_v1.json"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts/manifests/representation_variability_injection_v1.jsonl"
DEFAULT_SUMMARY = (
    REPO_ROOT / "artifacts/manifests/representation_variability_injection_summary_v1.json"
)
DEFAULT_AGGREGATE = (
    REPO_ROOT / "artifacts/manifests/representation_variability_injection_aggregate_v1.json"
)
DEFAULT_CACHE_ROOT = Path.home() / ".lightkurve/cache/mastDownload/TESS"
DEFAULT_MODEL_CACHE = REPO_ROOT / ".cache/representation_models"

ReportFn = Callable[..., bool]
SearchFn = Callable[..., Sequence[Any]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    resolved = (path if path.is_absolute() else REPO_ROOT / path).resolve()
    resolved.relative_to(REPO_ROOT.resolve())
    return resolved


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_contract(path: Path) -> dict[str, Any]:
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1:
        raise ValueError("representation variability/injection schema_version must be 1")
    for section in (
        "inputs",
        "target_selection",
        "injection_scenarios",
        "bls_comparator",
        "embedding_policy",
        "parallel_shape",
        "execution_gate",
    ):
        if section not in raw:
            raise ValueError(f"contract requires {section}")
    if len(raw["injection_scenarios"]) != 4:
        raise ValueError("contract requires exactly four injection scenarios")
    if raw.get("training_authorized") is not False:
        raise ValueError("benchmark must keep training_authorized=false")
    if raw.get("production_change_authorized") is not False:
        raise ValueError("benchmark must keep production_change_authorized=false")
    return raw


def validate_inputs(contract: Mapping[str, Any]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name, spec in contract["inputs"].items():
        path = _repo_path(spec["path"])
        if not path.is_file():
            raise ValueError(f"missing contracted input {name}: {path}")
        actual = _sha256(path)
        if actual != spec["sha256"]:
            raise ValueError(f"contracted input hash changed for {name}: {actual}")
        paths[name] = path
    overlap = json.loads(paths["asassn_overlap"].read_text(encoding="utf-8"))
    if (
        overlap.get("status") != "pass"
        or overlap.get("matched_unique_tics") != 48
        or overlap.get("training_authorized") is not False
    ):
        raise ValueError("ASAS-SN overlap evidence does not authorize benchmark design")
    smoke = json.loads(paths["inference_smoke_evidence"].read_text(encoding="utf-8"))
    if smoke.get("status") != "success" or smoke.get("training_authorized") is not False:
        raise ValueError("representation inference smoke is not a training-disabled PASS")
    return paths


def load_matched_labels(overlap: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    """Load matched rows only after verifying every aggregate-owned shard."""
    labels: dict[int, dict[str, Any]] = {}
    summaries = overlap.get("shard_summaries", [])
    if len(summaries) != 6:
        raise ValueError("ASAS-SN aggregate must own exactly six shard outputs")
    shard_indexes: set[int] = set()
    for summary in summaries:
        shard_index = int(summary["shard_index"])
        if shard_index in shard_indexes:
            raise ValueError(f"duplicate ASAS-SN shard summary: {shard_index}")
        shard_indexes.add(shard_index)
        path = _repo_path(summary["output_path"])
        if _sha256(path) != summary["output_sha256"]:
            raise ValueError(f"ASAS-SN shard output hash changed: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["status"] == "matched":
                tic_id = int(row["tic_id"])
                if tic_id in labels:
                    raise ValueError(f"duplicate matched TIC across ASAS-SN shards: {tic_id}")
                if row.get("training_authorized") is not False:
                    raise ValueError(
                        f"ASAS-SN matched TIC unexpectedly authorizes training: {tic_id}"
                    )
                labels[tic_id] = row
    if shard_indexes != set(range(6)):
        raise ValueError(f"ASAS-SN shard indexes are incomplete: {sorted(shard_indexes)}")
    if len(labels) != 48:
        raise ValueError(f"expected 48 matched labels, found {len(labels)}")
    return labels


def load_inventory(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def select_products(
    inventory: Sequence[Mapping[str, Any]], labels: Mapping[int, Mapping[str, Any]]
) -> dict[int, dict[str, Any]]:
    """Select the largest product per matched TIC with deterministic ties."""
    selected: dict[int, tuple[tuple[int, int, str], dict[str, Any]]] = {}
    for raw in inventory:
        tic_id = int(str(raw["target_id"]).split()[1])
        if tic_id not in labels:
            continue
        row = dict(raw)
        key = (
            int(row["size_bytes"]),
            -int(row["sector"]),
            str(row["cache_relative_path"]),
        )
        if tic_id not in selected or key > selected[tic_id][0]:
            selected[tic_id] = (key, row)
    result = {tic_id: value[1] for tic_id, value in selected.items()}
    if set(result) != set(labels):
        raise ValueError("not every matched TIC has a contracted cache product")
    return result


def select_shard(tic_ids: Sequence[int], shard_index: int, shard_count: int) -> tuple[int, ...]:
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid shard index/count")
    return tuple(sorted(tic_id for tic_id in tic_ids if tic_id % shard_count == shard_index))


def _shard_path(path: Path, shard_index: int, shard_count: int) -> Path:
    if shard_count == 1:
        return path
    return path.with_name(f"{path.stem}.shard{shard_index + 1}of{shard_count}{path.suffix}")


def thin_relative_magnitude(
    times: np.ndarray,
    flux: np.ndarray,
    *,
    reference_median: float,
    max_observations: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Evenly thin one series over its full baseline and use a shared median."""
    if times.size != flux.size or times.size < 200:
        raise ValueError("embedding input requires at least 200 aligned cadences")
    if max_observations < 200:
        raise ValueError("max_observations must be at least 200")
    if times.size > max_observations:
        indices = np.linspace(0, times.size - 1, max_observations, dtype=np.int64)
        times = times[indices]
        flux = flux[indices]
    magnitude = -2.5 * np.log10(flux / reference_median)
    if not np.all(np.isfinite(magnitude)):
        raise ValueError("relative magnitude contains non-finite values")
    return times.astype(np.float64), magnitude.astype(np.float32)


def cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("embedding norms must be finite and positive")
    value = 1.0 - float(np.dot(left, right) / denominator)
    return max(0.0, value)


def _load_target(
    tic_id: int,
    product: Mapping[str, Any],
    label: Mapping[str, Any],
    contract: Mapping[str, Any],
    cache_root: Path,
    search_fn: SearchFn,
) -> dict[str, Any]:
    root = cache_root.expanduser().resolve(strict=True)
    path = (root / str(product["cache_relative_path"])).resolve(strict=True)
    path.relative_to(root)
    if path.stat().st_size != int(product["size_bytes"]):
        raise ValueError(f"cached product size changed for TIC {tic_id}")
    with fits.open(path, mode="readonly", memmap=True) as hdul:
        table = hdul[1].data
        names = set(table.names or ())
        required = {"TIME", "PDCSAP_FLUX", "PDCSAP_FLUX_ERR", "QUALITY"}
        if not required.issubset(names):
            raise ValueError(f"TIC {tic_id} missing FITS columns: {sorted(required - names)}")
        times = np.asarray(table["TIME"], dtype=np.float64)
        flux = np.asarray(table["PDCSAP_FLUX"], dtype=np.float64)
        errors = np.asarray(table["PDCSAP_FLUX_ERR"], dtype=np.float64)
        quality = np.asarray(table["QUALITY"])
    keep = (
        np.isfinite(times)
        & np.isfinite(flux)
        & np.isfinite(errors)
        & (flux > 0)
        & (errors > 0)
        & (quality == 0)
    )
    times, flux, errors = times[keep], flux[keep], errors[keep]
    order = np.argsort(times, kind="stable")
    times, flux, errors = times[order], flux[order], errors[order]
    times, unique = np.unique(times, return_index=True)
    flux, errors = flux[unique], errors[unique]
    if times.size < 200 or times[-1] - times[0] < 20.0:
        raise ValueError(f"TIC {tic_id} has insufficient clean baseline")
    median = float(np.median(flux))
    normalized_flux = flux / median
    normalized_error = errors / median
    embedding_policy = contract["embedding_policy"]
    original_time, original_mag = thin_relative_magnitude(
        times,
        normalized_flux,
        reference_median=1.0,
        max_observations=int(embedding_policy["max_observations"]),
    )
    trials: list[dict[str, Any]] = []
    comparator = contract["bls_comparator"]
    epoch_phase = float(contract["injection_policy"]["epoch_phase"])
    for scenario in contract["injection_scenarios"]:
        period = float(scenario["period_days"])
        epoch = float(times[0] + epoch_phase * period)
        injected = inject_flux(
            times,
            normalized_flux,
            period_days=period,
            epoch_bjd=epoch,
            duration_hours=float(scenario["duration_hours"]),
            depth_ppm=float(scenario["depth_ppm"]),
        )
        signals = search_fn(
            _MockLC(times, injected, normalized_error),
            target_id=f"TIC {tic_id}",
            mission="TESS",
            period_min=float(comparator["period_min_days"]),
            period_max=float(comparator["period_max_days"]),
            duration_min_hours=float(comparator["duration_min_hours"]),
            duration_max_hours=float(comparator["duration_max_hours"]),
            n_durations=int(comparator["n_durations"]),
            min_snr=float(comparator["min_snr"]),
            max_peaks=int(comparator["max_peaks"]),
            max_period_grid_points=int(comparator["max_period_grid_points"]),
        )
        candidates = [
            {
                "period_days": float(signal.period_days),
                "depth_ppm": float(signal.depth_ppm),
                "snr": float(signal.snr),
            }
            for signal in signals
        ]
        recovered = match_recovery(
            candidates,
            period,
            tolerance=float(comparator["period_relative_tolerance"]),
        )
        injected_time, injected_mag = thin_relative_magnitude(
            times,
            injected,
            reference_median=1.0,
            max_observations=int(embedding_policy["max_observations"]),
        )
        trials.append(
            {
                "scenario": dict(scenario),
                "epoch_bjd": epoch,
                "times": injected_time,
                "magnitude": injected_mag,
                "bls_recovered": recovered is not None,
                "bls_candidate_count": len(candidates),
                "bls_recovered_period_days": recovered.get("period_days") if recovered else None,
                "bls_recovered_snr": recovered.get("snr") if recovered else None,
            }
        )
    return {
        "tic_id": tic_id,
        "label": dict(label),
        "product": dict(product),
        "input_cadences": int(times.size),
        "baseline_days": float(times[-1] - times[0]),
        "original_time": original_time,
        "original_magnitude": original_mag,
        "trials": trials,
    }


def _load_wrappers(
    source_contract: Mapping[str, Any], model_cache: Path
) -> list[tuple[dict[str, Any], Any]]:
    import onnxruntime as ort
    from light_curve.embed import Astromer2, ChronosBolt

    wrappers: list[tuple[dict[str, Any], Any]] = []
    for raw in source_contract["models"]:
        model = dict(raw)
        path = (
            model_cache
            / str(model["repo"]).replace("/", "__")
            / str(model["commit"])
            / str(model["filename"])
        ).resolve(strict=True)
        if path.stat().st_size != int(model["size_bytes"]) or _sha256(path) != model["sha256"]:
            raise ValueError(f"cached model identity changed: {model['name']}")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        if model["role"] == "general_time_series_foundation_baseline":
            wrapper = ChronosBolt(session=session, size="tiny", output="mean", reduction="end")
        elif model["role"] == "astronomy_native_foundation_comparator":
            wrapper = Astromer2(
                session=session, output="mean", reduction="non-overlapping-windows"
            )
        else:
            raise ValueError(f"unsupported model role: {model['role']}")
        wrappers.append((model, wrapper))
    return wrappers


def _embed(
    model: Mapping[str, Any], wrapper: Any, times: np.ndarray, mag: np.ndarray
) -> np.ndarray:
    if model["role"] == "general_time_series_foundation_baseline":
        value = wrapper(mag)
    else:
        value = wrapper(times, mag)
    embedding = np.asarray(value, dtype=np.float32)
    expected = (1, 1, 1, int(model["embedding_dimension"]))
    if embedding.shape != expected or not np.all(np.isfinite(embedding)):
        raise ValueError(f"invalid embedding from {model['name']}: {embedding.shape}")
    return embedding.reshape(-1)


def _format_eta(seconds: float) -> str:
    if seconds == float("inf"):
        return "unknown"
    return f"{seconds / 60:.0f}m{seconds % 60:.0f}s" if seconds > 90 else f"{seconds:.0f}s"


def run_shard(
    contract_path: Path,
    output_path: Path,
    summary_path: Path,
    *,
    cache_root: Path,
    model_cache: Path,
    workers: int,
    shard_index: int,
    shard_count: int,
    search_fn: SearchFn = search_lightcurve,
    wrappers_fn: Callable[
        [Mapping[str, Any], Path], list[tuple[dict[str, Any], Any]]
    ] = _load_wrappers,
    report_fn: ReportFn = run_and_commit_report,
) -> dict[str, Any]:
    contract = load_contract(contract_path)
    required_shards = int(contract["parallel_shape"]["process_shards"])
    required_workers = int(contract["parallel_shape"]["workers_per_shard"])
    if shard_count != required_shards or workers != required_workers:
        raise ValueError(
            f"benchmark requires {required_shards} shards x {required_workers} workers"
        )
    paths = validate_inputs(contract)
    overlap = json.loads(paths["asassn_overlap"].read_text(encoding="utf-8"))
    labels = load_matched_labels(overlap)
    products = select_products(load_inventory(paths["inventory"]), labels)
    selected = select_shard(tuple(products), shard_index, shard_count)
    if not selected:
        raise ValueError(f"shard {shard_index}/{shard_count} has no selected TICs")
    source_contract = json.loads(paths["model_source_contract"].read_text(encoding="utf-8"))
    started_at = datetime.now(UTC)
    started = time.monotonic()
    print(
        "Representation variability/injection startup: "
        f"tics={len(selected)} trials={len(selected) * 4} workers={workers} "
        f"shard={shard_index}/{shard_count} models=2 downloads=0 training=false",
        flush=True,
    )
    wrappers = wrappers_fn(source_contract, model_cache.resolve())
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _load_target,
                tic_id,
                products[tic_id],
                labels[tic_id],
                contract,
                cache_root,
                search_fn,
            ): tic_id
            for tic_id in selected
        }
        for completed, future in enumerate(as_completed(futures), 1):
            target = future.result()
            for model, wrapper in wrappers:
                original = _embed(
                    model, wrapper, target["original_time"], target["original_magnitude"]
                )
                original_hash = hashlib.sha256(original.tobytes()).hexdigest()
                for trial in target["trials"]:
                    injected = _embed(model, wrapper, trial["times"], trial["magnitude"])
                    scenario = trial["scenario"]
                    rows.append(
                        {
                            "tic_id": target["tic_id"],
                            "asassn_id": target["label"]["asassn_id"],
                            "class_code": target["label"]["class_code"],
                            "label_probability": target["label"]["probability"],
                            "discovery": target["label"]["discovery"],
                            "sector": target["product"]["sector"],
                            "cache_relative_path": target["product"]["cache_relative_path"],
                            "input_cadences": target["input_cadences"],
                            "baseline_days": target["baseline_days"],
                            "scenario_id": scenario["scenario_id"],
                            "period_days": scenario["period_days"],
                            "depth_ppm": scenario["depth_ppm"],
                            "duration_hours": scenario["duration_hours"],
                            "epoch_bjd": trial["epoch_bjd"],
                            "bls_recovered": trial["bls_recovered"],
                            "bls_candidate_count": trial["bls_candidate_count"],
                            "bls_recovered_period_days": trial["bls_recovered_period_days"],
                            "bls_recovered_snr": trial["bls_recovered_snr"],
                            "model_name": model["name"],
                            "original_embedding_sha256": original_hash,
                            "injected_embedding_sha256": hashlib.sha256(
                                injected.tobytes()
                            ).hexdigest(),
                            "cosine_distance": cosine_distance(original, injected),
                            "l2_distance": float(np.linalg.norm(original - injected)),
                            "downloaded_bytes": 0,
                            "persisted_embedding": False,
                            "training_authorized": False,
                            "production_change_authorized": False,
                        }
                    )
            elapsed = time.monotonic() - started
            rate = completed / elapsed if elapsed else 0.0
            remaining = (len(selected) - completed) / rate if rate else float("inf")
            print(
                f"  [{completed}/{len(selected)}] TICs elapsed={elapsed:.1f}s "
                f"ETA={_format_eta(remaining)} rows={len(rows)}",
                flush=True,
            )
    rows.sort(key=lambda row: (row["tic_id"], row["model_name"], row["scenario_id"]))
    output_path = _shard_path(output_path, shard_index, shard_count)
    summary_path = _shard_path(summary_path, shard_index, shard_count)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    elapsed = time.monotonic() - started
    unique_trials = {(row["tic_id"], row["scenario_id"]) for row in rows}
    summary = {
        "schema_version": 1,
        "artifact_id": "representation_variability_injection_shard_v1",
        "status": "success",
        "contract_sha256": _sha256(contract_path),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "workers": workers,
        "selected_tics": len(selected),
        "unique_trials": len(unique_trials),
        "model_rows": len(rows),
        "bls_recovered_trials": len(
            {(row["tic_id"], row["scenario_id"]) for row in rows if row["bls_recovered"]}
        ),
        "elapsed_seconds": elapsed,
        "output_path": _display_path(output_path),
        "output_sha256": _sha256(output_path),
        "downloaded_bytes": 0,
        "persisted_embeddings": 0,
        "training_authorized": False,
        "production_change_authorized": False,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = RunReport(
        script="benchmark_representation_variability_injection",
        status="success",
        started_at=started_at.isoformat(),
        completed_at=datetime.now(UTC).isoformat(),
        elapsed_seconds=elapsed,
        items_processed=len(unique_trials),
        items_written=len(rows),
        output_paths=(_display_path(output_path), _display_path(summary_path)),
        shard_index=shard_index,
        shard_count=shard_count,
        notes="cache-only paired BLS and frozen embedding shifts; no persisted arrays",
    )
    report_path = report_path_for(
        "benchmark_representation_variability_injection",
        shard_index=shard_index,
        shard_count=shard_count,
    )
    if not report_fn(report, report_path):
        print(f"WARNING: Run Report was not committed: {report_path}", flush=True)
    print(
        f"Representation variability/injection shard COMPLETE: tics={len(selected)} "
        f"trials={len(unique_trials)} model_rows={len(rows)} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return summary


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    trial_keys = {(int(row["tic_id"]), str(row["scenario_id"])) for row in rows}
    tic_classes = {int(row["tic_id"]): str(row["class_code"]) for row in rows}
    models = sorted({str(row["model_name"]) for row in rows})
    by_scenario: list[dict[str, Any]] = []
    for scenario_id in sorted({str(row["scenario_id"]) for row in rows}):
        subset = [row for row in rows if row["scenario_id"] == scenario_id]
        unique = {(int(row["tic_id"]), str(row["scenario_id"])) for row in subset}
        recovered = {
            (int(row["tic_id"]), str(row["scenario_id"]))
            for row in subset
            if row["bls_recovered"]
        }
        metrics = {}
        for model in models:
            values = [float(row["cosine_distance"]) for row in subset if row["model_name"] == model]
            metrics[model] = {
                "median_cosine_distance": float(np.median(values)),
                "mean_cosine_distance": float(np.mean(values)),
            }
        by_scenario.append(
            {
                "scenario_id": scenario_id,
                "trials": len(unique),
                "bls_recovered": len(recovered),
                "bls_recovery_rate": len(recovered) / len(unique),
                "embedding_metrics": metrics,
            }
        )
    depth_order: dict[str, dict[str, float | int]] = {}
    for model in models:
        lookup = {
            (int(row["tic_id"]), float(row["period_days"]), float(row["depth_ppm"])): float(
                row["cosine_distance"]
            )
            for row in rows
            if row["model_name"] == model
        }
        comparisons = 0
        ordered = 0
        for tic_id in {key[0] for key in lookup}:
            for period in (3.0, 10.0):
                low = lookup[(tic_id, period, 500.0)]
                high = lookup[(tic_id, period, 2000.0)]
                comparisons += 1
                ordered += int(high > low)
        depth_order[model] = {
            "comparisons": comparisons,
            "higher_depth_larger_shift": ordered,
            "fraction": ordered / comparisons,
        }
    return {
        "unique_tics": len({int(row["tic_id"]) for row in rows}),
        "unique_trials": len(trial_keys),
        "model_rows": len(rows),
        "models": models,
        "class_counts": dict(sorted(Counter(tic_classes.values()).items())),
        "by_scenario": by_scenario,
        "depth_order": depth_order,
    }


def aggregate_shards(
    contract_path: Path,
    output_path: Path,
    summary_path: Path,
    aggregate_path: Path,
    *,
    shard_count: int,
    report_fn: ReportFn = run_and_commit_report,
) -> dict[str, Any]:
    contract = load_contract(contract_path)
    if shard_count != int(contract["parallel_shape"]["process_shards"]):
        raise ValueError("aggregate requires the contracted shard count")
    started_at = datetime.now(UTC)
    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for shard_index in range(shard_count):
        output = _shard_path(output_path, shard_index, shard_count)
        summary_path_i = _shard_path(summary_path, shard_index, shard_count)
        summary = json.loads(summary_path_i.read_text(encoding="utf-8"))
        if summary["output_sha256"] != _sha256(output):
            raise ValueError(f"shard output hash mismatch: {output}")
        summaries.append(summary)
        rows.extend(json.loads(line) for line in output.read_text().splitlines() if line)
    metrics = summarize_rows(rows)
    gate = contract["execution_gate"]
    duplicate_tic_scenarios = len(rows) - len(
        {(row["tic_id"], row["scenario_id"], row["model_name"]) for row in rows}
    )
    checks = {
        "required_tics": metrics["unique_tics"] == int(gate["required_tics"]),
        "required_trials": metrics["unique_trials"] == int(gate["required_trials"]),
        "required_models": len(metrics["models"]) == int(gate["required_models"]),
        "required_model_rows": metrics["model_rows"]
        == int(gate["required_trials"]) * int(gate["required_models"]),
        "zero_duplicate_model_trials": duplicate_tic_scenarios == 0,
        "zero_downloaded_bytes": sum(int(row["downloaded_bytes"]) for row in rows) == 0,
        "zero_persisted_embeddings": not any(row["persisted_embedding"] for row in rows),
    }
    status = "pass" if all(checks.values()) else "fail"
    artifact = {
        "schema_version": 1,
        "artifact_id": "representation_variability_injection_aggregate_v1",
        "status": status,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "contract_path": _display_path(contract_path),
        "contract_sha256": _sha256(contract_path),
        "shard_count": shard_count,
        "workers_per_shard": int(contract["parallel_shape"]["workers_per_shard"]),
        "checks": checks,
        "duplicate_model_trials": duplicate_tic_scenarios,
        "downloaded_bytes": 0,
        "persisted_embeddings": 0,
        **metrics,
        "shard_summaries": summaries,
        "scientific_metrics_are_descriptive": True,
        "training_authorized": False,
        "production_change_authorized": False,
        "limitations": contract["limitations"],
    }
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    elapsed = time.monotonic() - started
    report = RunReport(
        script="benchmark_representation_variability_injection_aggregate",
        status="success" if status == "pass" else "failed",
        started_at=started_at.isoformat(),
        completed_at=datetime.now(UTC).isoformat(),
        elapsed_seconds=elapsed,
        items_processed=int(metrics["unique_trials"]),
        items_written=1,
        items_failed=0 if status == "pass" else sum(not value for value in checks.values()),
        output_paths=(_display_path(aggregate_path),),
    )
    report_path = report_path_for("benchmark_representation_variability_injection_aggregate")
    if not report_fn(report, report_path):
        print(f"WARNING: Run Report was not committed: {report_path}", flush=True)
    print(
        f"Representation variability/injection aggregate COMPLETE: status={status.upper()} "
        f"tics={metrics['unique_tics']} trials={metrics['unique_trials']} training=false",
        flush=True,
    )
    return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--aggregate-output", type=Path, default=DEFAULT_AGGREGATE)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--model-cache", type=Path, default=DEFAULT_MODEL_CACHE)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.aggregate_only:
        artifact = aggregate_shards(
            _repo_path(args.contract),
            _repo_path(args.output),
            _repo_path(args.summary),
            _repo_path(args.aggregate_output),
            shard_count=args.shard_count,
        )
        return 0 if artifact["status"] == "pass" else 1
    try:
        run_shard(
            _repo_path(args.contract),
            _repo_path(args.output),
            _repo_path(args.summary),
            cache_root=args.cache_root,
            model_cache=_repo_path(args.model_cache),
            workers=args.workers,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
        )
    except (ImportError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(
            f"ERROR: representation variability/injection benchmark failed: {exc}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
