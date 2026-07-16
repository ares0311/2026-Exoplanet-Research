"""Benchmark frozen external representations on grouped real Kepler labels.

The six shard processes read only exact cache-local Kepler products, prepare
phase-folded physical-magnitude inputs, and write temporary compressed feature
files below ``logs/``. Aggregate mode fits identical deterministic linear
probes on training rows, selects thresholds on validation rows, opens the
frozen test subset once, writes descriptive evidence, and removes every
temporary embedding array. No download or production change is authorized.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import re
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
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

from exo_toolkit.ml.cnn_scorer import CnnScorer  # noqa: E402
from Skills.run_report import RunReport, report_path_for, run_and_commit_report  # noqa: E402
from Skills.train_cnn import _compute_auc  # noqa: E402

DEFAULT_CONTRACT = REPO_ROOT / "metadata/grouped_external_representation_contract_v2.json"
DEFAULT_CACHE_ROOT = Path.home() / ".lightkurve/cache/mastDownload/Kepler"
DEFAULT_MODEL_CACHE = REPO_ROOT / ".cache/representation_models"
DEFAULT_TEMP_ROOT = REPO_ROOT / "logs/grouped_external_representation_benchmark_v1"
DEFAULT_SUMMARY = REPO_ROOT / "artifacts/manifests/grouped_external_representation_summary_v1.json"
DEFAULT_AGGREGATE = (
    REPO_ROOT / "artifacts/manifests/grouped_external_representation_aggregate_v1.json"
)
_KIC_PATTERN = re.compile(r"kplr(?P<id>\d{9})")

ReportFn = Callable[..., bool]


@dataclass(frozen=True)
class PreparedRow:
    """One selected signal prepared for frozen-model inference."""

    source_row_id: str
    group_key: str
    split: str
    label: int
    target_id: int
    skipped_cache_files: int
    chronos_magnitude: np.ndarray
    astromer_phase: np.ndarray
    astromer_magnitude: np.ndarray
    classical_features: np.ndarray
    cnn_flux: tuple[float, ...]


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


def _canonical_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = "".join(
        json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n" for row in rows
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    """Load the immutable grouped-benchmark contract and fail closed."""
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1:
        raise ValueError("grouped external representation schema_version must be 1")
    for section in (
        "inputs",
        "population",
        "preprocessing",
        "probe",
        "parallel_shape",
        "execution_gate",
    ):
        if section not in raw:
            raise ValueError(f"contract requires {section}")
    if raw.get("production_change_authorized") is not False:
        raise ValueError("contract must keep production_change_authorized=false")
    if raw.get("broad_extraction_authorized") is not False:
        raise ValueError("contract must keep broad_extraction_authorized=false")
    if raw.get("training_authorized") is not False:
        raise ValueError("contract must keep training_authorized=false")
    return raw


def validate_inputs(contract: Mapping[str, Any]) -> dict[str, Path]:
    """Validate every committed input path and SHA-256."""
    paths: dict[str, Path] = {}
    for name, spec in contract["inputs"].items():
        path = _repo_path(spec["path"])
        if not path.is_file():
            raise ValueError(f"missing contracted input {name}: {path}")
        actual = _sha256(path)
        if actual != spec["sha256"]:
            raise ValueError(f"contracted input hash changed for {name}: {actual}")
        paths[name] = path
    return paths


def discover_cache(cache_root: Path) -> dict[int, tuple[Path, ...]]:
    """Index cache-local Kepler FITS paths by KIC without opening payloads."""
    root = cache_root.expanduser().resolve(strict=True)
    paths: dict[int, list[Path]] = collections.defaultdict(list)
    for path in root.rglob("*.fits"):
        match = _KIC_PATTERN.search(path.name) or _KIC_PATTERN.search(path.parent.name)
        if match is not None:
            paths[int(match.group("id"))].append(path.resolve())
    if not paths:
        raise ValueError("Kepler cache contains no indexed FITS products")
    return {target_id: tuple(sorted(values)) for target_id, values in paths.items()}


def _selection_record(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_row_id": row["source_row_id"],
        "group_key": row["group_key"],
        "split": row["split"],
        "label": row["label"],
        "target_id": row["target_id"],
    }


def select_population(
    corpus_path: Path,
    cached_target_ids: set[int],
    contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Select one signal per KIC, balanced within each predefined split."""
    rows = [
        json.loads(line)
        for line in corpus_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_group: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        if int(row["target_id"]) in cached_target_ids:
            by_group[str(row["group_key"])].append(row)
    representatives = [
        sorted(group, key=lambda row: (str(row["source_row_id"]), str(row["target_name"])))[0]
        for group in by_group.values()
    ]
    seed = str(contract["population"]["selection_seed"])
    selected: list[dict[str, Any]] = []
    for split, per_label_raw in contract["population"]["rows_per_label"].items():
        per_label = int(per_label_raw)
        for label in (0, 1):
            candidates = [
                row
                for row in representatives
                if row["split"] == split and int(row["label"]) == label
            ]
            candidates.sort(
                key=lambda row: hashlib.sha256(
                    f"{seed}|{row['group_key']}".encode()
                ).hexdigest()
            )
            if len(candidates) < per_label:
                raise ValueError(f"insufficient {split} label={label} cache coverage")
            selected.extend(candidates[:per_label])
    selected.sort(key=lambda row: (str(row["split"]), int(row["label"]), str(row["group_key"])))
    groups = [str(row["group_key"]) for row in selected]
    if len(groups) != len(set(groups)):
        raise ValueError("selected population contains duplicate KIC groups")
    group_splits: dict[str, set[str]] = collections.defaultdict(set)
    for row in rows:
        group_splits[str(row["group_key"])].add(str(row["split"]))
    leaked = [group for group in groups if len(group_splits[group]) != 1]
    if leaked:
        raise ValueError(f"predefined corpus leaks groups across splits: {leaked[:3]}")
    fingerprint = _canonical_sha256([_selection_record(row) for row in selected])
    if fingerprint != contract["population"]["selection_sha256"]:
        raise ValueError(f"selected population fingerprint changed: {fingerprint}")
    return selected


def validate_selected_inventory(
    selected: Sequence[Mapping[str, Any]],
    cache: Mapping[int, Sequence[Path]],
    cache_root: Path,
    contract: Mapping[str, Any],
) -> set[Path]:
    """Validate the exact selected cache path/size inventory fingerprint."""
    root = cache_root.expanduser().resolve(strict=True)
    records: list[dict[str, Any]] = []
    for row in sorted(selected, key=lambda item: int(item["target_id"])):
        target_id = int(row["target_id"])
        for path in cache[target_id]:
            path.relative_to(root)
            records.append(
                {
                    "target_id": target_id,
                    "relative_path": str(path.relative_to(root)),
                    "size_bytes": path.stat().st_size,
                }
            )
    inventory = contract["population"]["cache_inventory"]
    if len(records) != int(inventory["files"]):
        raise ValueError(f"selected cache file count changed: {len(records)}")
    total_bytes = sum(int(record["size_bytes"]) for record in records)
    if total_bytes != int(inventory["bytes"]):
        raise ValueError(f"selected cache byte count changed: {total_bytes}")
    fingerprint = _canonical_sha256(records)
    if fingerprint != inventory["sha256"]:
        raise ValueError(f"selected cache inventory fingerprint changed: {fingerprint}")
    unreadable = inventory["known_unreadable_products"]
    exact_size = int(unreadable["exact_size_bytes"])
    unreadable_paths = {
        path
        for row in selected
        for path in cache[int(row["target_id"])]
        if path.stat().st_size == exact_size
    }
    unreadable_records = [
        {
            "relative_path": str(path.relative_to(root)),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(unreadable_paths, key=lambda value: str(value.relative_to(root)))
    ]
    if len(unreadable_records) != int(unreadable["files"]):
        raise ValueError(f"known unreadable cache file count changed: {len(unreadable_records)}")
    unreadable_bytes = sum(int(record["size_bytes"]) for record in unreadable_records)
    if unreadable_bytes != int(unreadable["bytes"]):
        raise ValueError(f"known unreadable cache byte count changed: {unreadable_bytes}")
    unreadable_fingerprint = _canonical_sha256(unreadable_records)
    if unreadable_fingerprint != unreadable["sha256"]:
        raise ValueError(
            f"known unreadable cache inventory changed: {unreadable_fingerprint}"
        )
    selected_targets = {int(row["target_id"]) for row in selected}
    unusable_targets = [
        target_id
        for target_id in selected_targets
        if all(path in unreadable_paths for path in cache[target_id])
    ]
    if unusable_targets:
        raise ValueError(f"selected targets have no readable cache product: {unusable_targets[:3]}")
    return unreadable_paths


def select_shard(
    rows: Sequence[Mapping[str, Any]], shard_index: int, shard_count: int
) -> list[dict[str, Any]]:
    """Partition selected rows by KIC modulo shard count."""
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid shard index/count")
    return [dict(row) for row in rows if int(row["target_id"]) % shard_count == shard_index]


def _shard_path(path: Path, shard_index: int, shard_count: int) -> Path:
    return path.with_name(f"{path.stem}.shard{shard_index + 1}of{shard_count}{path.suffix}")


def _read_phase_flux(
    paths: Sequence[Path],
    row: Mapping[str, Any],
    known_unreadable_paths: set[Path],
) -> tuple[np.ndarray, np.ndarray, int]:
    times_parts: list[np.ndarray] = []
    flux_parts: list[np.ndarray] = []
    skipped = 0
    for path in paths:
        if path in known_unreadable_paths:
            skipped += 1
            continue
        with fits.open(path, mode="readonly", memmap=True) as hdul:
            table = hdul[1].data
            names = set(table.names or ())
            required = {"TIME", "PDCSAP_FLUX", "SAP_QUALITY"}
            if not required.issubset(names):
                raise ValueError(f"{path.name} missing FITS columns: {sorted(required - names)}")
            header = hdul[1].header
            primary = hdul[0].header
            ref_i = header.get("BJDREFI", primary.get("BJDREFI"))
            ref_f = header.get("BJDREFF", primary.get("BJDREFF", 0.0))
            if ref_i is None:
                raise ValueError(f"{path.name} has no BJD reference")
            time_values = np.asarray(table["TIME"], dtype=np.float64) + float(ref_i) + float(ref_f)
            flux_values = np.asarray(table["PDCSAP_FLUX"], dtype=np.float64)
            quality = np.asarray(table["SAP_QUALITY"])
        keep = (
            np.isfinite(time_values)
            & np.isfinite(flux_values)
            & (flux_values > 0.0)
            & (quality == 0)
        )
        if np.any(keep):
            times_parts.append(time_values[keep])
            flux_parts.append(flux_values[keep])
    if not times_parts:
        raise ValueError(f"KIC {row['target_id']} has no usable cache cadences")
    times = np.concatenate(times_parts)
    flux = np.concatenate(flux_parts)
    period = float(row["period_days"])
    epoch = float(row["epoch_bjd"])
    if not math.isfinite(period) or period <= 0 or not math.isfinite(epoch):
        raise ValueError(f"KIC {row['target_id']} has invalid ephemeris")
    phase = ((times - epoch + 0.5 * period) % period) / period - 0.5
    order = np.argsort(phase, kind="stable")
    return phase[order], flux[order], skipped


def phase_bin_relative_magnitude(
    phase: np.ndarray,
    flux: np.ndarray,
    *,
    n_bins: int,
    minimum_filled_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Median-bin physical flux by phase and convert to relative magnitude."""
    if phase.size != flux.size or phase.size < n_bins:
        raise ValueError("phase and flux must be aligned with at least n_bins cadences")
    edges = np.linspace(-0.5, 0.5, n_bins + 1)
    indices = np.clip(np.searchsorted(edges, phase, side="right") - 1, 0, n_bins - 1)
    binned = np.full(n_bins, np.nan, dtype=np.float64)
    for index in np.unique(indices):
        binned[index] = float(np.median(flux[indices == index]))
    valid = np.flatnonzero(np.isfinite(binned) & (binned > 0))
    if valid.size / n_bins < minimum_filled_fraction:
        raise ValueError(f"phase coverage is too sparse: {valid.size}/{n_bins} bins")
    if valid.size != n_bins:
        binned = np.interp(np.arange(n_bins), valid, binned[valid], period=n_bins)
    reference = float(np.median(flux))
    magnitude = -2.5 * np.log10(binned / reference)
    if not np.all(np.isfinite(magnitude)):
        raise ValueError("relative magnitude contains non-finite values")
    centers = (edges[:-1] + edges[1:]) / 2.0
    return centers.astype(np.float64), magnitude.astype(np.float32)


def _prepare_row(
    row: Mapping[str, Any],
    paths: Sequence[Path],
    contract: Mapping[str, Any],
    known_unreadable_paths: set[Path],
) -> PreparedRow:
    phase, flux, skipped = _read_phase_flux(paths, row, known_unreadable_paths)
    preprocessing = contract["preprocessing"]
    minimum = float(preprocessing["minimum_filled_fraction"])
    chronos_phase, chronos_mag = phase_bin_relative_magnitude(
        phase,
        flux,
        n_bins=int(preprocessing["chronos_phase_bins"]),
        minimum_filled_fraction=minimum,
    )
    astromer_phase, astromer_mag = phase_bin_relative_magnitude(
        phase,
        flux,
        n_bins=int(preprocessing["astromer_phase_bins"]),
        minimum_filled_fraction=minimum,
    )
    del chronos_phase
    ordered = np.sort(astromer_mag.astype(np.float64))
    classical = np.asarray(
        [
            math.log1p(float(row["period_days"])),
            math.log1p(float(row["duration_hours"])),
            float(np.mean(astromer_mag)),
            float(np.std(astromer_mag)),
            float(np.quantile(ordered, 0.01)),
            float(np.quantile(ordered, 0.05)),
            float(np.min(ordered)),
        ],
        dtype=np.float32,
    )
    cnn_flux = tuple(float(value) for value in row["flux"])
    if len(cnn_flux) != int(preprocessing["cnn_bins"]):
        raise ValueError(f"KIC {row['target_id']} CNN snippet length changed")
    return PreparedRow(
        source_row_id=str(row["source_row_id"]),
        group_key=str(row["group_key"]),
        split=str(row["split"]),
        label=int(row["label"]),
        target_id=int(row["target_id"]),
        skipped_cache_files=skipped,
        chronos_magnitude=chronos_mag,
        astromer_phase=astromer_phase,
        astromer_magnitude=astromer_mag,
        classical_features=classical,
        cnn_flux=cnn_flux,
    )


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
            wrapper = Astromer2(session=session, output="mean", reduction="non-overlapping-windows")
        else:
            raise ValueError(f"unsupported model role: {model['role']}")
        wrappers.append((model, wrapper))
    return wrappers


def _embed(model: Mapping[str, Any], wrapper: Any, row: PreparedRow) -> np.ndarray:
    if model["role"] == "general_time_series_foundation_baseline":
        value = wrapper(row.chronos_magnitude)
    else:
        value = wrapper(row.astromer_phase, row.astromer_magnitude)
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
    cache_root: Path,
    model_cache: Path,
    temp_root: Path,
    summary_path: Path,
    *,
    workers: int,
    shard_index: int,
    shard_count: int,
    wrappers_fn: Callable[
        [Mapping[str, Any], Path], list[tuple[dict[str, Any], Any]]
    ] = _load_wrappers,
    cnn_scorer: CnnScorer | None = None,
    report_fn: ReportFn = run_and_commit_report,
) -> dict[str, Any]:
    """Prepare and score one deterministic KIC-modulo shard."""
    contract = load_contract(contract_path)
    if shard_count != int(contract["parallel_shape"]["process_shards"]):
        raise ValueError("benchmark requires contracted process shard count")
    if workers != int(contract["parallel_shape"]["fits_workers_per_shard"]):
        raise ValueError("benchmark requires contracted FITS worker count")
    paths = validate_inputs(contract)
    cache = discover_cache(cache_root)
    selected = select_population(paths["corpus"], set(cache), contract)
    known_unreadable_paths = validate_selected_inventory(
        selected, cache, cache_root, contract
    )
    shard_rows = select_shard(selected, shard_index, shard_count)
    if not shard_rows:
        raise ValueError(f"shard {shard_index}/{shard_count} has no selected rows")
    source_contract = json.loads(paths["model_source_contract"].read_text(encoding="utf-8"))
    wrappers = wrappers_fn(source_contract, model_cache.resolve())
    cnn = cnn_scorer or CnnScorer.from_checkpoint(
        paths["cnn_checkpoint"], calibration_path=paths["cnn_calibration"]
    )
    if not cnn.is_available:
        raise RuntimeError("promoted benchmark CNN is unavailable")
    started_at = datetime.now(UTC)
    started = time.monotonic()
    print(
        "Grouped external representation startup: "
        f"rows={len(shard_rows)} workers={workers} shard={shard_index}/{shard_count} "
        "models=2 downloads=0 temporary_embeddings=true production=false",
        flush=True,
    )
    prepared: list[PreparedRow] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _prepare_row,
                row,
                cache[int(row["target_id"])],
                contract,
                known_unreadable_paths,
            ): row
            for row in shard_rows
        }
        for completed, future in enumerate(as_completed(futures), 1):
            prepared.append(future.result())
            elapsed = time.monotonic() - started
            rate = completed / elapsed if elapsed else 0.0
            eta = (len(shard_rows) - completed) / rate if rate else float("inf")
            print(
                f"  prepare [{completed}/{len(shard_rows)}] elapsed={elapsed:.1f}s "
                f"ETA={_format_eta(eta)}",
                flush=True,
            )
    prepared.sort(key=lambda row: (row.split, row.label, row.group_key))
    model_names = {str(model["name"]) for model, _ in wrappers}
    expected_model_names = {"Chronos-Bolt tiny", "Astromer2"}
    if model_names != expected_model_names:
        raise ValueError(f"contracted model set changed: {sorted(model_names)}")
    model_embeddings: dict[str, list[np.ndarray]] = {
        str(model["name"]): [] for model, _ in wrappers
    }
    for completed, row in enumerate(prepared, 1):
        for model, wrapper in wrappers:
            model_embeddings[str(model["name"])].append(_embed(model, wrapper, row))
        elapsed = time.monotonic() - started
        rate = completed / elapsed if elapsed else 0.0
        eta = (len(prepared) - completed) / rate if rate else float("inf")
        print(
            f"  embed [{completed}/{len(prepared)}] elapsed={elapsed:.1f}s "
            f"ETA={_format_eta(eta)}",
            flush=True,
        )
    cnn_probabilities = cnn.predict_proba_batch([list(row.cnn_flux) for row in prepared])
    if len(cnn_probabilities) != len(prepared) or len(set(cnn_probabilities)) <= 1:
        raise RuntimeError("benchmark CNN probabilities are missing or constant")
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_path = _shard_path(temp_root / "features_v1.npz", shard_index, shard_count)
    np.savez_compressed(
        temp_path,
        source_row_id=np.asarray([row.source_row_id for row in prepared]),
        group_key=np.asarray([row.group_key for row in prepared]),
        split=np.asarray([row.split for row in prepared]),
        label=np.asarray([row.label for row in prepared], dtype=np.int8),
        target_id=np.asarray([row.target_id for row in prepared], dtype=np.int64),
        chronos=np.stack(model_embeddings["Chronos-Bolt tiny"]).astype(np.float32),
        astromer=np.stack(model_embeddings["Astromer2"]).astype(np.float32),
        classical=np.stack([row.classical_features for row in prepared]).astype(np.float32),
        cnn_probability=np.asarray(cnn_probabilities, dtype=np.float32),
    )
    elapsed = time.monotonic() - started
    summary_path = _shard_path(summary_path, shard_index, shard_count)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1,
        "artifact_id": "grouped_external_representation_shard_v1",
        "status": "success",
        "contract_sha256": _sha256(contract_path),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "workers": workers,
        "rows": len(prepared),
        "split_counts": dict(sorted(collections.Counter(row.split for row in prepared).items())),
        "label_counts": dict(
            sorted(collections.Counter(str(row.label) for row in prepared).items())
        ),
        "temp_path": _display_path(temp_path),
        "temp_sha256": _sha256(temp_path),
        "temp_bytes": temp_path.stat().st_size,
        "elapsed_seconds": elapsed,
        "downloaded_bytes": 0,
        "known_unreadable_cache_files_skipped": sum(
            row.skipped_cache_files for row in prepared
        ),
        "production_change_authorized": False,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = RunReport(
        script="benchmark_grouped_external_representations",
        status="success",
        started_at=started_at.isoformat(),
        completed_at=datetime.now(UTC).isoformat(),
        elapsed_seconds=elapsed,
        items_processed=len(prepared),
        items_written=len(prepared),
        output_paths=(_display_path(temp_path), _display_path(summary_path)),
        shard_index=shard_index,
        shard_count=shard_count,
        notes="cache-only features; temporary embeddings require aggregate cleanup",
    )
    report_path = report_path_for(
        "benchmark_grouped_external_representations",
        shard_index=shard_index,
        shard_count=shard_count,
    )
    if not report_fn(report, report_path):
        print(f"WARNING: Run Report was not committed: {report_path}", flush=True)
    print(
        f"Grouped external representation shard COMPLETE: rows={len(prepared)} "
        f"elapsed={elapsed:.1f}s temporary={temp_path}",
        flush=True,
    )
    return summary


def _average_precision(labels: np.ndarray, probabilities: np.ndarray) -> float:
    order = np.argsort(-probabilities, kind="stable")
    ranked = labels[order]
    positives = int(np.sum(ranked == 1))
    if positives == 0:
        return 0.0
    cumulative = np.cumsum(ranked == 1)
    precision = cumulative / np.arange(1, ranked.size + 1)
    return float(np.sum(precision[ranked == 1]) / positives)


def _binary_cross_entropy(labels: np.ndarray, probabilities: np.ndarray) -> float:
    clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
    return float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)))


def _standardize(
    train: np.ndarray, validation: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(train, axis=0, keepdims=True)
    std = np.maximum(np.std(train, axis=0, keepdims=True), 1e-6)
    return (train - mean) / std, (validation - mean) / std, (test - mean) / std


def fit_linear_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    config: Mapping[str, Any],
    *,
    progress: bool = True,
) -> tuple[np.ndarray, float, int, float]:
    """Fit a deterministic L2 logistic probe and select by validation AUC."""
    epochs = int(config["epochs"])
    learning_rate = float(config["learning_rate"])
    weight_decay = float(config["weight_decay"])
    patience = int(config["patience"])
    weights = np.zeros(train_x.shape[1], dtype=np.float64)
    bias = 0.0
    best_auc = -1.0
    best_epoch = 0
    best_weights = weights.copy()
    best_bias = bias
    stale = 0
    for epoch in range(1, epochs + 1):
        train_logits = train_x @ weights + bias
        train_probabilities = 1.0 / (1.0 + np.exp(-np.clip(train_logits, -30, 30)))
        error = train_probabilities - train_y
        gradient = train_x.T @ error / train_y.size + weight_decay * weights
        bias_gradient = float(np.mean(error))
        weights -= learning_rate * gradient
        bias -= learning_rate * bias_gradient
        validation_probabilities = 1.0 / (
            1.0 + np.exp(-np.clip(validation_x @ weights + bias, -30, 30))
        )
        val_auc = _compute_auc(validation_y.tolist(), validation_probabilities.tolist())
        penalty = 0.5 * weight_decay * float(np.dot(weights, weights))
        train_loss = _binary_cross_entropy(train_y, train_probabilities) + penalty
        val_loss = _binary_cross_entropy(validation_y, validation_probabilities)
        improved = val_auc > best_auc + 1e-8
        if improved:
            best_auc = val_auc
            best_epoch = epoch
            best_weights = weights.copy()
            best_bias = bias
            stale = 0
        else:
            stale += 1
        if progress:
            marker = "best" if improved else f"patience {stale}/{patience}"
            print(
                f"Epoch {epoch:3d}/{epochs} train={train_loss:.6f} val={val_loss:.6f} "
                f"auc={val_auc:.6f} lr={learning_rate:.2e} {marker}",
                flush=True,
            )
        if stale >= patience:
            break
    return best_weights, best_bias, best_epoch, best_auc


def _probabilities(features: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(features @ weights + bias, -30, 30)))


def choose_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Select the validation F1 optimum, breaking ties conservatively upward."""
    best: tuple[float, float] = (-1.0, 1.0)
    for threshold in np.unique(np.r_[0.0, probabilities, 1.0]):
        predictions = probabilities >= threshold
        tp = int(np.sum(predictions & (labels == 1)))
        fp = int(np.sum(predictions & (labels == 0)))
        fn = int(np.sum(~predictions & (labels == 1)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        best = max(best, (f1, float(threshold)))
    return best[1]


def evaluate_scores(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
    top_k: int,
) -> dict[str, Any]:
    predictions = probabilities >= threshold
    tp = int(np.sum(predictions & (labels == 1)))
    fp = int(np.sum(predictions & (labels == 0)))
    tn = int(np.sum(~predictions & (labels == 0)))
    fn = int(np.sum(~predictions & (labels == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    k = min(top_k, labels.size)
    order = np.argsort(-probabilities, kind="stable")[:k]
    top_positives = int(np.sum(labels[order] == 1))
    return {
        "roc_auc": _compute_auc(labels.tolist(), probabilities.tolist()),
        "average_precision": _average_precision(labels, probabilities),
        "threshold_from_validation": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_discovery_rate": fp / (tp + fp) if tp + fp else 0.0,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "top_k": k,
        "top_k_positives": top_positives,
        "top_k_positive_fraction": top_positives / k,
    }


def aggregate_shards(
    contract_path: Path,
    temp_root: Path,
    summary_path: Path,
    aggregate_path: Path,
    *,
    shard_count: int,
    report_fn: ReportFn = run_and_commit_report,
) -> dict[str, Any]:
    """Fit probes, evaluate the frozen subset once, and remove embeddings."""
    contract = load_contract(contract_path)
    if shard_count != int(contract["parallel_shape"]["process_shards"]):
        raise ValueError("aggregate requires contracted shard count")
    started_at = datetime.now(UTC)
    started = time.monotonic()
    arrays: dict[str, list[np.ndarray]] = collections.defaultdict(list)
    summaries: list[dict[str, Any]] = []
    temp_paths: list[Path] = []
    for shard_index in range(shard_count):
        temp_path = _shard_path(temp_root / "features_v1.npz", shard_index, shard_count)
        summary_i = _shard_path(summary_path, shard_index, shard_count)
        summary = json.loads(summary_i.read_text(encoding="utf-8"))
        if (
            summary.get("status") != "success"
            or int(summary.get("shard_index", -1)) != shard_index
            or int(summary.get("shard_count", -1)) != shard_count
            or summary.get("contract_sha256") != _sha256(contract_path)
            or int(summary.get("downloaded_bytes", -1)) != 0
            or int(summary.get("known_unreadable_cache_files_skipped", -1)) < 0
            or summary.get("production_change_authorized") is not False
        ):
            raise ValueError(f"invalid shard summary: {summary_i}")
        if summary["temp_sha256"] != _sha256(temp_path):
            raise ValueError(f"temporary shard hash mismatch: {temp_path}")
        with np.load(temp_path, allow_pickle=False) as payload:
            for key in payload.files:
                arrays[key].append(np.asarray(payload[key]))
        summaries.append(summary)
        temp_paths.append(temp_path)
    merged = {key: np.concatenate(values, axis=0) for key, values in arrays.items()}
    expected = int(contract["execution_gate"]["required_rows"])
    if merged["label"].size != expected:
        raise ValueError(f"aggregate row count changed: {merged['label'].size}")
    groups = merged["group_key"].astype(str)
    if len(set(groups.tolist())) != expected:
        raise ValueError("aggregate contains duplicate KIC groups")
    splits = merged["split"].astype(str)
    labels = merged["label"].astype(np.int8)
    masks = {name: splits == name for name in ("train", "val", "test")}
    expected_split = contract["execution_gate"]["required_split_rows"]
    for name, mask in masks.items():
        if int(np.sum(mask)) != int(expected_split[name]):
            raise ValueError(f"aggregate {name} count changed")
        label_counts = collections.Counter(labels[mask].tolist())
        expected_labels = contract["execution_gate"]["required_split_label_rows"][name]
        for label in (0, 1):
            if label_counts[label] != int(expected_labels[str(label)]):
                raise ValueError(f"aggregate {name} label={label} count changed")
    skipped_cache_files = sum(
        int(summary["known_unreadable_cache_files_skipped"]) for summary in summaries
    )
    expected_skipped = int(
        contract["population"]["cache_inventory"]["known_unreadable_products"]["files"]
    )
    if skipped_cache_files != expected_skipped:
        raise ValueError(
            f"known unreadable cache skip count changed: {skipped_cache_files}"
        )
    top_k = int(contract["probe"]["top_k"])
    results: dict[str, Any] = {}
    for name, key in (
        ("Chronos-Bolt tiny", "chronos"),
        ("Astromer2", "astromer"),
        ("statistical_ephemeris_baseline", "classical"),
    ):
        print(f"Probe startup: model={name} train={int(np.sum(masks['train']))}", flush=True)
        train_x, val_x, test_x = _standardize(
            merged[key][masks["train"]].astype(np.float64),
            merged[key][masks["val"]].astype(np.float64),
            merged[key][masks["test"]].astype(np.float64),
        )
        weights, bias, best_epoch, validation_auc = fit_linear_probe(
            train_x,
            labels[masks["train"]].astype(np.float64),
            val_x,
            labels[masks["val"]].astype(np.float64),
            contract["probe"],
        )
        validation_probabilities = _probabilities(val_x, weights, bias)
        threshold = choose_threshold(labels[masks["val"]], validation_probabilities)
        test_probabilities = _probabilities(test_x, weights, bias)
        results[name] = {
            "probe_best_epoch": best_epoch,
            "validation_roc_auc": validation_auc,
            "test": evaluate_scores(
                labels[masks["test"]],
                test_probabilities,
                threshold=threshold,
                top_k=top_k,
            ),
        }
    cnn_validation = merged["cnn_probability"][masks["val"]].astype(np.float64)
    cnn_test = merged["cnn_probability"][masks["test"]].astype(np.float64)
    cnn_threshold = choose_threshold(labels[masks["val"]], cnn_validation)
    results["benchmark_cnn_v1"] = {
        "probe_best_epoch": None,
        "validation_roc_auc": _compute_auc(labels[masks["val"]].tolist(), cnn_validation.tolist()),
        "test": evaluate_scores(
            labels[masks["test"]], cnn_test, threshold=cnn_threshold, top_k=top_k
        ),
    }
    cnn_comparator = results["benchmark_cnn_v1"]["test"]
    statistical_comparator = results["statistical_ephemeris_baseline"]["test"]
    external_names = ("Chronos-Bolt tiny", "Astromer2")
    comparison_metrics = ("roc_auc", "average_precision", "top_k_positive_fraction")
    minimum_improvement = float(contract["probe"]["minimum_absolute_improvement"])
    improvements = {
        name: {
            metric: results[name]["test"][metric]
            - max(cnn_comparator[metric], statistical_comparator[metric])
            for metric in comparison_metrics
        }
        for name in external_names
    }
    adds_value = any(
        improvements[name][metric] >= minimum_improvement
        for name in external_names
        for metric in comparison_metrics
    )
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "schema_version": 1,
        "artifact_id": "grouped_external_representation_aggregate_v1",
        "status": "pass",
        "scientific_outcome": "external_adds_value" if adds_value else "no_external_added_value",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "contract_path": _display_path(contract_path),
        "contract_sha256": _sha256(contract_path),
        "rows": expected,
        "split_rows": {name: int(np.sum(mask)) for name, mask in masks.items()},
        "results": results,
        "external_improvement_over_best_required_comparator": improvements,
        "minimum_absolute_improvement": minimum_improvement,
        "shard_summaries": summaries,
        "downloaded_bytes": 0,
        "known_unreadable_cache_files_skipped": skipped_cache_files,
        "temporary_embedding_files_removed": len(temp_paths),
        "persisted_embeddings": 0,
        "test_opened_once": True,
        "training_authorized": False,
        "production_change_authorized": False,
        "limitations": contract["limitations"],
    }
    aggregate_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for temp_path in temp_paths:
        temp_path.unlink()
    if any(path.exists() for path in temp_paths):
        raise RuntimeError("temporary embedding cleanup failed")
    elapsed = time.monotonic() - started
    report = RunReport(
        script="benchmark_grouped_external_representations_aggregate",
        status="success",
        started_at=started_at.isoformat(),
        completed_at=datetime.now(UTC).isoformat(),
        elapsed_seconds=elapsed,
        items_processed=expected,
        items_written=1,
        output_paths=(_display_path(aggregate_path),),
        notes="validation-selected probes; frozen test opened once; temporary embeddings removed",
    )
    report_path = report_path_for("benchmark_grouped_external_representations_aggregate")
    if not report_fn(report, report_path):
        print(f"WARNING: Run Report was not committed: {report_path}", flush=True)
    print(
        "Grouped external representation aggregate COMPLETE: "
        f"status=PASS outcome={artifact['scientific_outcome']} rows={expected} "
        "downloads=0 persisted_embeddings=0 production=false",
        flush=True,
    )
    return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--model-cache", type=Path, default=DEFAULT_MODEL_CACHE)
    parser.add_argument("--temp-root", type=Path, default=DEFAULT_TEMP_ROOT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--aggregate-output", type=Path, default=DEFAULT_AGGREGATE)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.aggregate_only:
            aggregate_shards(
                _repo_path(args.contract),
                _repo_path(args.temp_root),
                _repo_path(args.summary),
                _repo_path(args.aggregate_output),
                shard_count=args.shard_count,
            )
        else:
            run_shard(
                _repo_path(args.contract),
                args.cache_root,
                _repo_path(args.model_cache),
                _repo_path(args.temp_root),
                _repo_path(args.summary),
                workers=args.workers,
                shard_index=args.shard_index,
                shard_count=args.shard_count,
            )
    except (ImportError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: grouped external representation benchmark failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
