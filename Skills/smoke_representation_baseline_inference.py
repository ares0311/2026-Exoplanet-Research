"""Run bounded pinned-ONNX inference on one cached TESS light curve.

This Phase 3 smoke verifies that the two source-approved external embedding
models load and emit finite 256-dimensional vectors on the validated Python
3.14 runtime. It downloads only the two exact-revision ONNX files into an
ignored in-repo cache, opens one existing cached SPOC product, and runs each
model in an isolated child process with one ONNX intra/inter-op thread.

It is not training, evaluation, promotion, or permission to extract the full
inventory. Scientific comparison remains separately gated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from Skills.benchmark_representation_preprocessing import (  # noqa: E402
    load_inventory_contract,
    select_target_balanced_sample,
)
from Skills.run_report import RunReport, report_path_for, run_and_commit_report  # noqa: E402

SCRIPT_NAME = "smoke_representation_baseline_inference"
SMOKE_ID = "representation_baseline_inference_smoke_v1"
DEFAULT_SOURCE_CONTRACT = Path("metadata/representation_baseline_source_contract_v1.json")
DEFAULT_SOURCE_EVIDENCE = Path(
    "artifacts/manifests/representation_baseline_source_verification_v1.json"
)
DEFAULT_DATASET_MANIFEST = Path(
    "metadata/dataset_manifests/tess_cached_unlabeled_representation_v1.json"
)
DEFAULT_INVENTORY_SUMMARY = Path("artifacts/manifests/representation_cache_inventory_v1.json")
DEFAULT_CACHE_ROOT = Path.home() / ".lightkurve/cache/mastDownload/TESS"
DEFAULT_MODEL_CACHE = Path(".cache/representation_models")
DEFAULT_OUTPUT = Path("artifacts/manifests/representation_baseline_inference_smoke_v1.json")
MAX_OBSERVATIONS = 2048

ReportFn = Callable[..., bool]
DownloadFn = Callable[..., str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _repo_path(path: Path) -> Path:
    candidate = path if path.is_absolute() else REPO_ROOT / path
    resolved = candidate.resolve()
    resolved.relative_to(REPO_ROOT.resolve())
    return resolved


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_source_gate(
    contract_path: Path,
    evidence_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fail closed unless the committed metadata-only source gate passed."""
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if contract.get("schema_version") != 1 or evidence.get("schema_version") != 1:
        raise ValueError("source contract/evidence schema_version must be 1")
    if evidence.get("status") != "success":
        raise ValueError("source verification evidence is not successful")
    if evidence.get("payload_bytes_downloaded") != 0:
        raise ValueError("source verification evidence must remain metadata-only")
    if evidence.get("training_authorized") is not False:
        raise ValueError("source evidence must not authorize training")
    actual_contract_sha = _sha256(contract_path)
    if evidence.get("contract_sha256") != actual_contract_sha:
        raise ValueError(
            "source contract SHA-256 mismatch: "
            f"expected={evidence.get('contract_sha256')} actual={actual_contract_sha}"
        )
    expected_bytes = sum(
        int(item["wheel"]["size_bytes"]) for item in contract.get("packages", [])
    ) + sum(int(item["size_bytes"]) for item in contract.get("models", []))
    if expected_bytes != int(contract.get("aggregate_download_bytes", -1)):
        raise ValueError("source contract aggregate bytes are inconsistent")
    if evidence.get("aggregate_download_bytes_if_installed") != expected_bytes:
        raise ValueError("source evidence aggregate bytes differ from contract")
    if len(contract.get("models", [])) != 2 or len(evidence.get("models", [])) != 2:
        raise ValueError("bounded inference smoke requires exactly two verified models")
    return contract, evidence


def prepare_relative_magnitude(
    time_values: Any,
    flux_values: Any,
    quality_values: Any,
    *,
    max_observations: int = MAX_OBSERVATIONS,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter a SPOC series and convert positive relative flux to magnitudes."""
    if max_observations < 200:
        raise ValueError("max_observations must be at least 200")
    times = np.asarray(time_values, dtype=np.float64)
    flux = np.asarray(flux_values, dtype=np.float64)
    quality = np.asarray(quality_values)
    if not (times.ndim == flux.ndim == quality.ndim == 1):
        raise ValueError("time, flux, and quality must be one-dimensional")
    if not (times.size == flux.size == quality.size):
        raise ValueError("time, flux, and quality lengths differ")
    keep = np.isfinite(times) & np.isfinite(flux) & (flux > 0.0) & (quality == 0)
    times = times[keep]
    flux = flux[keep]
    if times.size < 200:
        raise ValueError(f"only {times.size} clean positive-flux cadences remain")
    order = np.argsort(times, kind="stable")
    times = times[order]
    flux = flux[order]
    times, unique_indices = np.unique(times, return_index=True)
    flux = flux[unique_indices]
    if times.size < 200:
        raise ValueError(f"only {times.size} unique cadences remain")
    if times.size > max_observations:
        times = times[-max_observations:]
        flux = flux[-max_observations:]
    median_flux = float(np.median(flux))
    if not math.isfinite(median_flux) or median_flux <= 0.0:
        raise ValueError("median clean flux must be finite and positive")
    magnitude = -2.5 * np.log10(flux / median_flux)
    if not np.all(np.isfinite(magnitude)):
        raise ValueError("relative magnitudes contain non-finite values")
    return times.astype(np.float64), magnitude.astype(np.float32)


def load_smoke_series(
    row: Mapping[str, Any],
    cache_root: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Open one inventory-owned cached FITS product with containment checks."""
    resolved_root = cache_root.expanduser().resolve(strict=True)
    relative_path = str(row["cache_relative_path"])
    product_path = (resolved_root / relative_path).resolve(strict=True)
    product_path.relative_to(resolved_root)
    expected_size = int(row["size_bytes"])
    actual_size = product_path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"cached product size changed: expected={expected_size} actual={actual_size}"
        )
    with fits.open(product_path, mode="readonly", memmap=True) as hdul:
        table = hdul[1].data
        names = set(table.names or ())
        required = {"TIME", "PDCSAP_FLUX", "QUALITY"}
        if not required.issubset(names):
            raise ValueError(f"missing FITS columns: {sorted(required - names)}")
        input_cadences = len(table)
        times, magnitude = prepare_relative_magnitude(
            table["TIME"],
            table["PDCSAP_FLUX"],
            table["QUALITY"],
        )
    return times, magnitude, {
        "target_id": str(row["target_id"]),
        "group_key": str(row["group_key"]),
        "sector": int(row["sector"]),
        "cache_relative_path": relative_path,
        "input_bytes": actual_size,
        "input_cadences": input_cadences,
        "retained_cadences": int(times.size),
        "relative_magnitude_sha256": hashlib.sha256(magnitude.tobytes()).hexdigest(),
    }


def _default_downloader(**kwargs: Any) -> str:
    from huggingface_hub import hf_hub_download

    return str(hf_hub_download(**kwargs))


def cache_pinned_model(
    model: Mapping[str, Any],
    model_cache: Path,
    *,
    downloader: DownloadFn = _default_downloader,
) -> tuple[Path, bool]:
    """Download one exact-revision model into the ignored in-repo cache."""
    repo = str(model["repo"])
    commit = str(model["commit"])
    filename = str(model["filename"])
    destination = (model_cache / repo.replace("/", "__") / commit).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    expected_path = destination / filename
    valid_before = (
        expected_path.is_file()
        and expected_path.stat().st_size == int(model["size_bytes"])
        and _sha256(expected_path) == model["sha256"]
    )
    returned = Path(
        downloader(
            repo_id=repo,
            filename=filename,
            revision=commit,
            local_dir=str(destination),
            force_download=not valid_before,
        )
    ).resolve(strict=True)
    returned.relative_to(destination)
    if returned.name != filename:
        raise ValueError(f"download returned unexpected filename: {returned.name}")
    actual_size = returned.stat().st_size
    actual_sha = _sha256(returned)
    if actual_size != int(model["size_bytes"]) or actual_sha != model["sha256"]:
        raise ValueError(
            f"downloaded model mismatch for {repo}: expected size/hash "
            f"{model['size_bytes']}/{model['sha256']} actual {actual_size}/{actual_sha}"
        )
    return returned, not valid_before


def run_model_inference(
    model: Mapping[str, Any],
    model_path: Path,
    times: np.ndarray,
    magnitude: np.ndarray,
    *,
    ort_module: Any | None = None,
    chronos_class: Any | None = None,
    astromer_class: Any | None = None,
) -> dict[str, Any]:
    """Load one bounded CPU session and validate a finite mean embedding."""
    if ort_module is None:
        import onnxruntime as ort_module
    if chronos_class is None or astromer_class is None:
        from light_curve.embed import Astromer2, ChronosBolt

        chronos_class = ChronosBolt if chronos_class is None else chronos_class
        astromer_class = Astromer2 if astromer_class is None else astromer_class
    options = ort_module.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session_started = time.monotonic()
    session = ort_module.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    if hasattr(session, "disable_fallback"):
        session.disable_fallback()
    session_seconds = time.monotonic() - session_started
    role = str(model["role"])
    if role == "general_time_series_foundation_baseline":
        wrapper = chronos_class(session=session, size="tiny", output="mean", reduction="end")
        inference_args = (magnitude,)
    elif role == "astronomy_native_foundation_comparator":
        wrapper = astromer_class(
            session=session,
            output="mean",
            reduction="non-overlapping-windows",
        )
        inference_args = (times, magnitude)
    else:
        raise ValueError(f"unsupported model role: {role}")
    inference_started = time.monotonic()
    embedding = np.asarray(wrapper(*inference_args), dtype=np.float32)
    inference_seconds = time.monotonic() - inference_started
    expected_shape = (1, 1, 1, int(model["embedding_dimension"]))
    if embedding.shape != expected_shape:
        raise ValueError(
            f"unexpected embedding shape: expected={expected_shape} actual={embedding.shape}"
        )
    if not np.all(np.isfinite(embedding)):
        raise ValueError("embedding contains non-finite values")
    embedding_norm = float(np.linalg.norm(embedding))
    if not math.isfinite(embedding_norm) or embedding_norm <= 0.0:
        raise ValueError("embedding norm must be finite and positive")
    return {
        "name": str(model["name"]),
        "repo": str(model["repo"]),
        "commit": str(model["commit"]),
        "filename": str(model["filename"]),
        "model_sha256": _sha256(model_path),
        "model_bytes": model_path.stat().st_size,
        "provider": "CPUExecutionProvider",
        "intra_op_num_threads": 1,
        "inter_op_num_threads": 1,
        "session_seconds": session_seconds,
        "inference_seconds": inference_seconds,
        "embedding_shape": list(embedding.shape),
        "embedding_norm": embedding_norm,
        "embedding_sha256": hashlib.sha256(embedding.tobytes()).hexdigest(),
        "finite": True,
        "peak_rss_bytes": _peak_rss_bytes(),
        "status": "success",
    }


def _run_child(
    model_spec_path: Path,
    row_path: Path,
    model_path: Path,
    cache_root: Path,
    output_path: Path,
) -> int:
    model = json.loads(model_spec_path.read_text(encoding="utf-8"))
    row = json.loads(row_path.read_text(encoding="utf-8"))
    times, magnitude, input_metadata = load_smoke_series(row, cache_root)
    result = run_model_inference(model, model_path, times, magnitude)
    result["input"] = input_metadata
    _write_json_atomic(output_path, result)
    print(
        f"  child model={result['name']} status=success "
        f"session={result['session_seconds']:.3f}s inference={result['inference_seconds']:.3f}s "
        f"peak_rss={result['peak_rss_bytes']}",
        flush=True,
    )
    return 0


def _run_model_child(
    model: Mapping[str, Any],
    row: Mapping[str, Any],
    model_path: Path,
    cache_root: Path,
    temporary_root: Path,
    index: int,
) -> dict[str, Any]:
    model_spec_path = temporary_root / f"model_{index}.json"
    row_path = temporary_root / "row.json"
    output_path = temporary_root / f"result_{index}.json"
    model_spec_path.write_text(json.dumps(model), encoding="utf-8")
    if not row_path.exists():
        row_path.write_text(json.dumps(row), encoding="utf-8")
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--internal-model-spec",
        str(model_spec_path),
        "--internal-row",
        str(row_path),
        "--internal-model-path",
        str(model_path),
        "--internal-output",
        str(output_path),
        "--cache-root",
        str(cache_root),
    ]
    child_env = os.environ.copy()
    child_env.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    completed = subprocess.run(command, cwd=REPO_ROOT, env=child_env, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"inference child for {model['name']} exited {completed.returncode}")
    return json.loads(output_path.read_text(encoding="utf-8"))


def run_smoke(
    source_contract_path: Path,
    source_evidence_path: Path,
    dataset_manifest_path: Path,
    inventory_summary_path: Path,
    cache_root: Path,
    model_cache: Path,
    output_path: Path,
    *,
    downloader: DownloadFn = _default_downloader,
    report_fn: ReportFn = run_and_commit_report,
) -> dict[str, Any]:
    """Run the two-model, one-product, exact-revision inference smoke."""
    started_at = datetime.now(UTC)
    started = time.monotonic()
    contract, source_evidence = load_source_gate(source_contract_path, source_evidence_path)
    _dataset, _summary, inventory_rows, inventory_sha = load_inventory_contract(
        dataset_manifest_path,
        inventory_summary_path,
    )
    selected_row = select_target_balanced_sample(inventory_rows, 1)[0]
    models = list(contract["models"])
    print(
        "Representation inference smoke startup: models=2 products=1 "
        f"max_observations={MAX_OBSERVATIONS} provider=CPUExecutionProvider "
        "threads=1+1 execution=isolated-sequential training=false",
        flush=True,
    )
    model_cache = _repo_path(model_cache)
    downloaded_bytes = 0
    model_paths: list[Path] = []
    for index, model in enumerate(models, 1):
        model_path, downloaded = cache_pinned_model(model, model_cache, downloader=downloader)
        model_paths.append(model_path)
        if downloaded:
            downloaded_bytes += int(model["size_bytes"])
        elapsed = time.monotonic() - started
        print(
            f"  download [{index}/2] model={model['name']} cached={not downloaded} "
            f"bytes={model['size_bytes']} elapsed={elapsed:.1f}s",
            flush=True,
        )
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="exo-representation-inference-") as temporary:
        temporary_root = Path(temporary)
        for index, (model, model_path) in enumerate(zip(models, model_paths, strict=True), 1):
            child_started = time.monotonic()
            result = _run_model_child(
                model,
                selected_row,
                model_path,
                cache_root,
                temporary_root,
                index,
            )
            results.append(result)
            elapsed = time.monotonic() - started
            rate = index / max(time.monotonic() - child_started, 1e-9)
            eta = (len(models) - index) / rate if rate else 0.0
            print(
                f"  inference [{index}/2] model={model['name']} status=success "
                f"elapsed={elapsed:.1f}s ETA={eta:.1f}s",
                flush=True,
            )
    input_metadata = results[0]["input"]
    if any(result["input"] != input_metadata for result in results[1:]):
        raise ValueError("model children did not use identical input provenance")
    completed_at = datetime.now(UTC)
    elapsed = time.monotonic() - started
    artifact: dict[str, Any] = {
        "schema_version": 1,
        "artifact_id": SMOKE_ID,
        "status": "success",
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": completed_at.isoformat(),
        "elapsed_seconds": elapsed,
        "runtime_python": sys.version.split()[0],
        "source_contract_sha256": _sha256(source_contract_path),
        "source_evidence_sha256": _sha256(source_evidence_path),
        "inventory_sha256": inventory_sha,
        "source_gate_status": source_evidence["status"],
        "selected_input": input_metadata,
        "model_cache": str(model_cache.relative_to(REPO_ROOT.resolve())),
        "model_cache_bytes": sum(int(model["size_bytes"]) for model in models),
        "downloaded_this_run_bytes": downloaded_bytes,
        "results": results,
        "training_authorized": False,
        "full_inventory_extraction_authorized": False,
        "limitations": [
            "One-product smoke verifies runtime integration, finite shape, timing, and "
            "memory only.",
            "Embedding hashes are device/runtime evidence, not scientific performance metrics.",
            "No grouped holdout, top-k, variability, or injection-recovery comparison is "
            "performed.",
        ],
    }
    output_path = _repo_path(output_path)
    _write_json_atomic(output_path, artifact)
    report = RunReport(
        script=SCRIPT_NAME,
        status="success",
        started_at=started_at.isoformat(),
        completed_at=completed_at.isoformat(),
        elapsed_seconds=elapsed,
        items_processed=len(results),
        items_written=len(results),
        output_paths=(str(output_path),),
        notes=(
            "one cached TESS product; exact-revision ONNX weights; isolated CPU inference; "
            "no training or full-inventory extraction"
        ),
    )
    report_path = report_path_for(SCRIPT_NAME)
    if not report_fn(report, report_path):
        print(f"WARNING: Run Report push failed for {report_path}", flush=True)
    print(
        f"Representation inference smoke COMPLETE: models={len(results)} "
        f"downloaded_bytes={downloaded_bytes} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return artifact


def format_status(artifact: Mapping[str, Any]) -> str:
    """Format a durable inference-smoke artifact for operator inspection."""
    return (
        "Representation inference smoke status: "
        f"status={artifact.get('status')} models={len(artifact.get('results', []))} "
        f"model_cache_bytes={artifact.get('model_cache_bytes')} "
        f"downloaded_this_run_bytes={artifact.get('downloaded_this_run_bytes')} "
        f"training_authorized={artifact.get('training_authorized')}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-contract", type=Path, default=DEFAULT_SOURCE_CONTRACT)
    parser.add_argument("--source-evidence", type=Path, default=DEFAULT_SOURCE_EVIDENCE)
    parser.add_argument("--dataset-manifest", type=Path, default=DEFAULT_DATASET_MANIFEST)
    parser.add_argument("--inventory-summary", type=Path, default=DEFAULT_INVENTORY_SUMMARY)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--model-cache", type=Path, default=DEFAULT_MODEL_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--internal-model-spec", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--internal-row", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--internal-model-path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--internal-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.internal_model_spec is not None:
        required = (args.internal_row, args.internal_model_path, args.internal_output)
        if any(value is None for value in required):
            parser.error("internal child mode requires row, model path, and output")
        return _run_child(
            args.internal_model_spec,
            args.internal_row,
            args.internal_model_path,
            args.cache_root,
            args.internal_output,
        )
    output_path = _repo_path(args.output)
    if args.status_only:
        if not output_path.is_file():
            print(f"ERROR: inference smoke artifact is missing: {output_path}", file=sys.stderr)
            return 1
        artifact = json.loads(output_path.read_text(encoding="utf-8"))
        print(format_status(artifact), flush=True)
        return 0
    try:
        artifact = run_smoke(
            _repo_path(args.source_contract),
            _repo_path(args.source_evidence),
            _repo_path(args.dataset_manifest),
            _repo_path(args.inventory_summary),
            args.cache_root,
            args.model_cache,
            args.output,
        )
    except (ImportError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: representation inference smoke failed: {exc}", file=sys.stderr)
        return 1
    print(format_status(artifact), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
