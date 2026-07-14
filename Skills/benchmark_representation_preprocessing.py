"""Benchmark bounded, in-memory preprocessing of the cached TESS inventory.

This Phase 3 gate opens existing cached SPOC light-curve FITS products only. It
never queries MAST, downloads data, or persists derived light-curve arrays.
The default execution uses six logical shards with six workers per shard and
writes a small aggregate benchmark manifest plus the standard Run Report.
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
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from Skills.run_report import RunReport, report_path_for, run_and_commit_report  # noqa: E402

SCRIPT_NAME = "benchmark_representation_preprocessing"
BENCHMARK_ID = "representation_preprocessing_benchmark_v1"
DEFAULT_DATASET_MANIFEST = Path(
    "metadata/dataset_manifests/tess_cached_unlabeled_representation_v1.json"
)
DEFAULT_INVENTORY_SUMMARY = Path("artifacts/manifests/representation_cache_inventory_v1.json")
DEFAULT_OUTPUT = Path("artifacts/manifests/representation_preprocessing_benchmark_v1.json")
DEFAULT_CACHE_ROOT = Path.home() / ".lightkurve/cache/mastDownload/TESS"
DEFAULT_SAMPLE_PRODUCTS = 36
DEFAULT_SHARD_COUNT = 6
DEFAULT_WORKERS_PER_SHARD = 6
DEFAULT_OUTPUT_BINS = 2048


@dataclass(frozen=True)
class ProductResult:
    """Metadata-only result for one in-memory product transformation."""

    target_id: str
    sector: int
    cache_relative_path: str
    input_bytes: int
    input_cadences: int
    retained_cadences: int
    output_bins: int
    output_bytes: int
    output_sha256: str | None
    worker_peak_rss_bytes: int
    elapsed_seconds: float
    status: str
    error: str | None = None


ReportFn = Callable[..., bool]
ProcessorFn = Callable[[dict[str, Any], Path, int], ProductResult]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_inventory_contract(
    dataset_manifest_path: Path,
    inventory_summary_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], str]:
    """Load and validate the committed training-only inventory contract."""
    dataset = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(inventory_summary_path.read_text(encoding="utf-8"))
    dataset_id = dataset.get("dataset_id")
    if dataset_id != "tess_cached_unlabeled_representation_v1":
        raise ValueError(f"unexpected dataset_id: {dataset_id!r}")
    if dataset.get("role") != "training" or summary.get("role") != "training":
        raise ValueError("representation cache inventory must remain training-only")
    if summary.get("inventory_id") != dataset_id:
        raise ValueError("dataset manifest and inventory summary IDs differ")

    inventory_path = REPO_ROOT / str(dataset["local_path"])
    inventory_path = inventory_path.resolve(strict=True)
    inventory_path.relative_to(REPO_ROOT.resolve())
    actual_sha256 = _sha256(inventory_path)
    expected_hashes = {dataset.get("sha256"), summary.get("rows_sha256")}
    if expected_hashes != {actual_sha256}:
        raise ValueError(
            "inventory SHA-256 mismatch: "
            f"actual={actual_sha256} expected={sorted(str(value) for value in expected_hashes)}"
        )

    rows: list[dict[str, Any]] = []
    with inventory_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("dataset_id") != dataset_id or row.get("role") != "training":
                raise ValueError(f"{inventory_path}:{line_number}: invalid dataset role/ID")
            rows.append(row)
    expected_counts = {int(dataset["row_count"]), int(summary["eligible_product_count"])}
    if expected_counts != {len(rows)}:
        raise ValueError(
            f"inventory row-count mismatch: actual={len(rows)} expected={expected_counts}"
        )
    if len({str(row["group_key"]) for row in rows}) != int(dataset["group_count"]):
        raise ValueError("inventory group count does not match the dataset manifest")
    return dataset, summary, rows, actual_sha256


def select_target_balanced_sample(
    rows: Sequence[dict[str, Any]], sample_products: int
) -> list[dict[str, Any]]:
    """Select deterministic cross-sector products with no repeated TIC group."""
    if sample_products <= 0:
        raise ValueError("sample_products must be positive")
    group_count = len({str(row["group_key"]) for row in rows})
    if sample_products > group_count:
        raise ValueError(
            f"sample_products={sample_products} exceeds unique target groups={group_count}"
        )
    ordered = sorted(
        rows,
        key=lambda row: (
            int(row["sector"]),
            str(row["group_key"]),
            str(row["cache_relative_path"]),
        ),
    )
    if sample_products == 1:
        desired_indices = [len(ordered) // 2]
    else:
        desired_indices = [
            round(index * (len(ordered) - 1) / (sample_products - 1))
            for index in range(sample_products)
        ]
    selected: list[dict[str, Any]] = []
    used_groups: set[str] = set()
    for desired in desired_indices:
        chosen: dict[str, Any] | None = None
        for distance in range(len(ordered)):
            candidates = (desired - distance, desired + distance)
            for candidate_index in candidates:
                if not 0 <= candidate_index < len(ordered):
                    continue
                candidate = ordered[candidate_index]
                group_key = str(candidate["group_key"])
                if group_key not in used_groups:
                    chosen = candidate
                    break
            if chosen is not None:
                break
        if chosen is None:
            raise RuntimeError("could not construct a target-balanced sample")
        selected.append(chosen)
        used_groups.add(str(chosen["group_key"]))
    return selected


def preprocess_product(row: dict[str, Any], cache_root: Path, output_bins: int) -> ProductResult:
    """Normalize and resample one cached FITS product without retaining its array."""
    started = time.monotonic()
    relative_path = str(row["cache_relative_path"])
    target_id = str(row["target_id"])
    sector = int(row["sector"])
    expected_bytes = int(row["size_bytes"])
    input_cadences = 0
    retained_cadences = 0
    try:
        if output_bins < 8:
            raise ValueError("output_bins must be at least 8")
        resolved_root = cache_root.expanduser().resolve(strict=True)
        product_path = (resolved_root / relative_path).resolve(strict=True)
        product_path.relative_to(resolved_root)
        actual_bytes = product_path.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"cached file size changed: expected={expected_bytes} actual={actual_bytes}"
            )
        with fits.open(product_path, mode="readonly", memmap=True) as hdul:
            table = hdul[1].data
            names = set(table.names or ())
            required = {"TIME", "PDCSAP_FLUX", "QUALITY"}
            if not required.issubset(names):
                raise ValueError(f"missing FITS columns: {sorted(required - names)}")
            times = np.asarray(table["TIME"], dtype=np.float64)
            flux = np.asarray(table["PDCSAP_FLUX"], dtype=np.float64)
            quality = np.asarray(table["QUALITY"])
        input_cadences = int(times.size)
        keep = np.isfinite(times) & np.isfinite(flux) & (quality == 0)
        times = times[keep]
        flux = flux[keep]
        retained_cadences = int(times.size)
        if retained_cadences < 32:
            raise ValueError(f"only {retained_cadences} clean cadences remain")
        order = np.argsort(times, kind="stable")
        times = times[order]
        flux = flux[order]
        unique_times, unique_indices = np.unique(times, return_index=True)
        flux = flux[unique_indices]
        if unique_times.size < 32 or unique_times[-1] <= unique_times[0]:
            raise ValueError("insufficient unique time coverage")

        median = float(np.median(flux))
        scale = float(1.4826 * np.median(np.abs(flux - median)))
        if not math.isfinite(scale) or scale <= np.finfo(np.float64).eps:
            scale = float(np.std(flux))
        if not math.isfinite(scale) or scale <= np.finfo(np.float64).eps:
            raise ValueError("flux has zero or non-finite robust scale")
        normalized = (flux - median) / scale
        grid = np.linspace(unique_times[0], unique_times[-1], output_bins, dtype=np.float64)
        derived = np.interp(grid, unique_times, normalized).astype(np.float32)
        if not np.all(np.isfinite(derived)):
            raise ValueError("resampled flux contains non-finite values")
        output_bytes = int(derived.nbytes)
        output_sha256 = hashlib.sha256(derived.tobytes()).hexdigest()
        del derived, normalized, grid, flux, times, unique_times
        return ProductResult(
            target_id=target_id,
            sector=sector,
            cache_relative_path=relative_path,
            input_bytes=expected_bytes,
            input_cadences=input_cadences,
            retained_cadences=retained_cadences,
            output_bins=output_bins,
            output_bytes=output_bytes,
            output_sha256=output_sha256,
            worker_peak_rss_bytes=_peak_rss_bytes(),
            elapsed_seconds=time.monotonic() - started,
            status="success",
        )
    except Exception as exc:
        return ProductResult(
            target_id=target_id,
            sector=sector,
            cache_relative_path=relative_path,
            input_bytes=expected_bytes,
            input_cadences=input_cadences,
            retained_cadences=retained_cadences,
            output_bins=output_bins,
            output_bytes=0,
            output_sha256=None,
            worker_peak_rss_bytes=_peak_rss_bytes(),
            elapsed_seconds=time.monotonic() - started,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )


def _format_eta(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "unknown"
    return f"{seconds / 60:.0f}m{seconds % 60:.0f}s" if seconds > 90 else f"{seconds:.0f}s"


def _process_shard(
    shard_index: int,
    shard_count: int,
    rows: Sequence[dict[str, Any]],
    cache_root: Path,
    output_bins: int,
    workers_per_shard: int,
    processor: ProcessorFn,
) -> list[ProductResult]:
    """Run one logical shard inside a process, using threads for cached FITS I/O."""
    completed = 0
    started = time.monotonic()
    results: list[ProductResult] = []
    with ThreadPoolExecutor(
        max_workers=min(workers_per_shard, len(rows)),
        thread_name_prefix=f"representation-s{shard_index + 1}",
    ) as pool:
        futures = {pool.submit(processor, row, cache_root, output_bins): row for row in rows}
        for future in as_completed(futures):
            row = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = ProductResult(
                    target_id=str(row["target_id"]),
                    sector=int(row["sector"]),
                    cache_relative_path=str(row["cache_relative_path"]),
                    input_bytes=int(row["size_bytes"]),
                    input_cadences=0,
                    retained_cadences=0,
                    output_bins=output_bins,
                    output_bytes=0,
                    output_sha256=None,
                    worker_peak_rss_bytes=_peak_rss_bytes(),
                    elapsed_seconds=0.0,
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
            results.append(result)
            completed += 1
            elapsed = time.monotonic() - started
            rate = completed / elapsed if elapsed else 0.0
            eta = (len(rows) - completed) / rate if rate else float("inf")
            print(
                f"  shard={shard_index + 1}/{shard_count} [{completed}/{len(rows)}] "
                f"target={result.target_id} sector={result.sector} "
                f"status={result.status} elapsed={elapsed:.1f}s ETA={_format_eta(eta)}",
                flush=True,
            )
    return results


def process_sample(
    rows: Sequence[dict[str, Any]],
    cache_root: Path,
    output_bins: int,
    *,
    shard_count: int = DEFAULT_SHARD_COUNT,
    workers_per_shard: int = DEFAULT_WORKERS_PER_SHARD,
    execution_backend: str = "process",
    processor: ProcessorFn = preprocess_product,
) -> list[ProductResult]:
    """Process rows with six shard processes and six I/O workers per shard."""
    if shard_count <= 0 or workers_per_shard <= 0:
        raise ValueError("shard_count and workers_per_shard must be positive")
    if len(rows) < shard_count:
        raise ValueError("sample must contain at least one product per logical shard")
    if execution_backend not in {"process", "thread"}:
        raise ValueError("execution_backend must be 'process' or 'thread'")
    logical_shards = [list(rows[index::shard_count]) for index in range(shard_count)]

    if execution_backend == "process":
        if processor is not preprocess_product:
            raise ValueError("custom processors require execution_backend='thread'")
        return _process_sample_subprocesses(
            logical_shards,
            cache_root,
            output_bins,
            workers_per_shard,
        )

    combined: list[ProductResult] = []
    overall_started = time.monotonic()
    with ThreadPoolExecutor(
        max_workers=shard_count, thread_name_prefix="representation-shard"
    ) as outer_pool:
        futures = [
            outer_pool.submit(
                _process_shard,
                shard_index,
                shard_count,
                shard_rows,
                cache_root,
                output_bins,
                workers_per_shard,
                processor,
            )
            for shard_index, shard_rows in enumerate(logical_shards)
        ]
        for future in as_completed(futures):
            shard_results = future.result()
            combined.extend(shard_results)
            elapsed = time.monotonic() - overall_started
            print(
                f"  parent progress: products={len(combined)}/{len(rows)} elapsed={elapsed:.1f}s",
                flush=True,
            )
    return sorted(combined, key=lambda result: result.cache_relative_path)


def _process_sample_subprocesses(
    logical_shards: Sequence[Sequence[dict[str, Any]]],
    cache_root: Path,
    output_bins: int,
    workers_per_shard: int,
) -> list[ProductResult]:
    """Supervise one ordinary Python subprocess per logical shard."""
    shard_count = len(logical_shards)
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
    combined: list[ProductResult] = []
    with tempfile.TemporaryDirectory(prefix="exo-representation-benchmark-") as temporary:
        temporary_root = Path(temporary)
        processes: list[tuple[int, subprocess.Popen[str], Path]] = []

        def stop_children() -> None:
            for _shard_index, process, _output_path in processes:
                if process.poll() is None:
                    process.terminate()
            for _shard_index, process, _output_path in processes:
                if process.poll() is None:
                    try:
                        process.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()

        try:
            for shard_index, shard_rows in enumerate(logical_shards):
                input_path = temporary_root / f"shard_{shard_index + 1}_input.json"
                output_path = temporary_root / f"shard_{shard_index + 1}_result.json"
                input_path.write_text(json.dumps(list(shard_rows)), encoding="utf-8")
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--internal-shard-input",
                    str(input_path),
                    "--internal-shard-output",
                    str(output_path),
                    "--internal-shard-index",
                    str(shard_index),
                    "--internal-shard-count",
                    str(shard_count),
                    "--cache-root",
                    str(cache_root),
                    "--output-bins",
                    str(output_bins),
                    "--workers",
                    str(workers_per_shard),
                ]
                process = subprocess.Popen(
                    command,
                    cwd=REPO_ROOT,
                    env=child_env,
                    text=True,
                )
                processes.append((shard_index, process, output_path))
                print(
                    f"  shard={shard_index + 1}/{shard_count} started pid={process.pid} "
                    f"products={len(shard_rows)}",
                    flush=True,
                )
            for shard_index, process, output_path in processes:
                returncode = process.wait()
                if returncode != 0:
                    raise RuntimeError(
                        f"representation shard {shard_index + 1}/{shard_count} "
                        f"exited with status {returncode}"
                    )
                raw_results = json.loads(output_path.read_text(encoding="utf-8"))
                combined.extend(ProductResult(**result) for result in raw_results)
                print(
                    f"  shard={shard_index + 1}/{shard_count} complete "
                    f"products={len(raw_results)} parent_total={len(combined)}",
                    flush=True,
                )
        except BaseException:
            stop_children()
            raise
    return sorted(combined, key=lambda result: result.cache_relative_path)


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _selection_sha256(rows: Sequence[dict[str, Any]]) -> str:
    payload = "\n".join(str(row["cache_relative_path"]) for row in rows).encode()
    return hashlib.sha256(payload).hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_benchmark(
    dataset_manifest_path: Path,
    inventory_summary_path: Path,
    cache_root: Path,
    output_path: Path,
    *,
    sample_products: int = DEFAULT_SAMPLE_PRODUCTS,
    output_bins: int = DEFAULT_OUTPUT_BINS,
    shard_count: int = DEFAULT_SHARD_COUNT,
    workers_per_shard: int = DEFAULT_WORKERS_PER_SHARD,
    execution_backend: str = "process",
    max_failures: int = 0,
    processor: ProcessorFn = preprocess_product,
    report_fn: ReportFn = run_and_commit_report,
) -> dict[str, Any]:
    """Run the bounded benchmark and write one small aggregate manifest."""
    if max_failures < 0:
        raise ValueError("max_failures cannot be negative")
    started_at = datetime.now(UTC)
    started = time.monotonic()
    dataset, source_summary, all_rows, inventory_sha256 = load_inventory_contract(
        dataset_manifest_path, inventory_summary_path
    )
    selected = select_target_balanced_sample(all_rows, sample_products)
    print(
        "Representation preprocessing benchmark startup: "
        f"products={len(selected)} shards={shard_count} "
        f"workers_per_shard={workers_per_shard} total_workers={shard_count * workers_per_shard} "
        f"output_bins={output_bins} mode=cache-only/in-memory zero-download",
        flush=True,
    )
    results = process_sample(
        selected,
        cache_root,
        output_bins,
        shard_count=shard_count,
        workers_per_shard=workers_per_shard,
        execution_backend=execution_backend,
        processor=processor,
    )
    elapsed = time.monotonic() - started
    successes = [result for result in results if result.status == "success"]
    failures = [result for result in results if result.status != "success"]
    status = "success" if not failures else "partial"
    if len(failures) > max_failures:
        status = "failed"
    product_seconds = [result.elapsed_seconds for result in successes]
    total_output_bytes = sum(result.output_bytes for result in successes)
    bytes_per_product = total_output_bytes / len(successes) if successes else 0.0
    projected_output_bytes = round(bytes_per_product * len(all_rows))
    payload: dict[str, Any] = {
        "schema_version": 1,
        "benchmark_id": BENCHMARK_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": status,
        "purpose": "bound preprocessing throughput and derived-array size before Phase 3 training",
        "mode": "existing-cache-only_in-memory_no-derived-array-retention",
        "downloads_performed": 0,
        "derived_arrays_persisted": 0,
        "source": {
            "dataset_id": dataset["dataset_id"],
            "role": dataset["role"],
            "inventory_sha256": inventory_sha256,
            "inventory_products": len(all_rows),
            "inventory_groups": int(dataset["group_count"]),
            "inventory_bytes": int(source_summary["eligible_bytes"]),
        },
        "configuration": {
            "sample_products": len(selected),
            "selection": "deterministic cross-sector, one product per TIC group",
            "selection_sha256": _selection_sha256(selected),
            "logical_shards": shard_count,
            "workers_per_shard": workers_per_shard,
            "maximum_concurrent_workers": shard_count * workers_per_shard,
            "execution_backend": execution_backend,
            "quality_filter": "finite TIME and PDCSAP_FLUX with QUALITY == 0",
            "normalization": "per-product median and 1.4826*MAD; standard-deviation fallback",
            "resampling": f"linear interpolation to {output_bins} float32 flux bins",
        },
        "aggregate": {
            "products_processed": len(results),
            "products_succeeded": len(successes),
            "products_failed": len(failures),
            "elapsed_seconds": elapsed,
            "products_per_second": len(results) / elapsed if elapsed else 0.0,
            "input_bytes_processed": sum(result.input_bytes for result in results),
            "input_cadences": sum(result.input_cadences for result in results),
            "retained_cadences": sum(result.retained_cadences for result in results),
            "derived_bytes_per_product": bytes_per_product,
            "sample_derived_bytes_not_persisted": total_output_bytes,
            "projected_full_inventory_derived_bytes": projected_output_bytes,
            "projected_full_inventory_derived_gb_decimal": projected_output_bytes / 1e9,
            "projected_full_inventory_seconds_at_observed_aggregate_rate": (
                len(all_rows) * elapsed / len(results) if results else None
            ),
            "product_elapsed_seconds_min": min(product_seconds) if product_seconds else None,
            "product_elapsed_seconds_median": (
                float(np.median(product_seconds)) if product_seconds else None
            ),
            "product_elapsed_seconds_max": max(product_seconds) if product_seconds else None,
            "parent_peak_rss_bytes": _peak_rss_bytes(),
            "maximum_shard_process_peak_rss_bytes": (
                max(result.worker_peak_rss_bytes for result in results) if results else None
            ),
        },
        "product_results": [asdict(result) for result in results],
        "limitations": [
            "This is a bounded preprocessing benchmark, not a model-quality result.",
            "The projected footprint covers normalized flux arrays only, not optimizer "
            "states, checkpoints, or training caches.",
            "The inventory remains training-only and cannot support validation, "
            "calibration, frozen evaluation, or discovery claims.",
            "Peak RSS is process-level high-water memory and may include pre-existing "
            "interpreter allocations.",
        ],
    }
    _write_json_atomic(output_path, payload)
    if status != "failed":
        report = RunReport(
            script=SCRIPT_NAME,
            status=status,
            started_at=started_at.isoformat(),
            completed_at=datetime.now(UTC).isoformat(),
            elapsed_seconds=elapsed,
            items_processed=len(results),
            items_written=len(successes),
            items_failed=len(failures),
            output_paths=(str(output_path),),
            notes="zero downloads; derived float32 arrays measured in memory and discarded",
        )
        report_path = report_path_for(SCRIPT_NAME)
        if not report_fn(report, report_path):
            print(f"WARNING: Run Report push failed for {report_path}", flush=True)
    print(
        f"Representation preprocessing benchmark {status.upper()}: "
        f"processed={len(results)} succeeded={len(successes)} failed={len(failures)} "
        f"rate={payload['aggregate']['products_per_second']:.2f}/s "
        f"projected_derived={projected_output_bytes / 1e6:.2f}MB elapsed={elapsed:.1f}s",
        flush=True,
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, default=DEFAULT_DATASET_MANIFEST)
    parser.add_argument("--inventory-summary", type=Path, default=DEFAULT_INVENTORY_SUMMARY)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-products", type=int, default=DEFAULT_SAMPLE_PRODUCTS)
    parser.add_argument("--output-bins", type=int, default=DEFAULT_OUTPUT_BINS)
    parser.add_argument("--shards", type=int, default=DEFAULT_SHARD_COUNT)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS_PER_SHARD)
    parser.add_argument("--backend", choices=("process", "thread"), default="process")
    parser.add_argument("--max-failures", type=int, default=0)
    parser.add_argument("--internal-shard-input", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--internal-shard-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--internal-shard-index", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--internal-shard-count", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the contract and selection without opening FITS files or writing output.",
    )
    args = parser.parse_args(argv)
    internal_values = (
        args.internal_shard_input,
        args.internal_shard_output,
        args.internal_shard_index,
        args.internal_shard_count,
    )
    if any(value is not None for value in internal_values):
        if not all(value is not None for value in internal_values):
            parser.error("all internal shard arguments are required together")
        shard_rows = json.loads(args.internal_shard_input.read_text(encoding="utf-8"))
        shard_results = _process_shard(
            args.internal_shard_index,
            args.internal_shard_count,
            shard_rows,
            args.cache_root,
            args.output_bins,
            args.workers,
            preprocess_product,
        )
        args.internal_shard_output.write_text(
            json.dumps([asdict(result) for result in shard_results]),
            encoding="utf-8",
        )
        return 0
    if args.dry_run:
        dataset, _summary, rows, inventory_sha256 = load_inventory_contract(
            args.dataset_manifest, args.inventory_summary
        )
        selected = select_target_balanced_sample(rows, args.sample_products)
        print(
            f"DRY RUN: dataset={dataset['dataset_id']} inventory_sha256={inventory_sha256} "
            f"selected={len(selected)} sectors={min(int(row['sector']) for row in selected)}-"
            f"{max(int(row['sector']) for row in selected)} shards={args.shards} "
            f"workers={args.workers} backend={args.backend} downloads=0 writes=0",
            flush=True,
        )
        return 0
    payload = run_benchmark(
        args.dataset_manifest,
        args.inventory_summary,
        args.cache_root,
        args.output,
        sample_products=args.sample_products,
        output_bins=args.output_bins,
        shard_count=args.shards,
        workers_per_shard=args.workers,
        execution_backend=args.backend,
        max_failures=args.max_failures,
    )
    return 0 if payload["status"] != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
