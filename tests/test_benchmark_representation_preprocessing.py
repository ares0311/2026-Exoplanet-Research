from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import Skills.benchmark_representation_preprocessing as benchmark
from astropy.io import fits
from Skills.benchmark_representation_preprocessing import (
    ProductResult,
    load_inventory_contract,
    preprocess_product,
    process_sample,
    run_benchmark,
    select_target_balanced_sample,
)


def _row(index: int, *, sector: int | None = None, size_bytes: int = 100) -> dict[str, Any]:
    return {
        "dataset_id": "tess_cached_unlabeled_representation_v1",
        "target_id": f"TIC {1000 + index}",
        "group_key": f"tess:tic:{1000 + index}",
        "sector": sector if sector is not None else index + 1,
        "cache_relative_path": f"product_{index}/product_{index}_lc.fits",
        "size_bytes": size_bytes,
        "role": "training",
    }


def _write_contract(root: Path, rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    inventory = root / "metadata/inventory.jsonl"
    inventory.parent.mkdir(parents=True)
    inventory.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    digest = hashlib.sha256(inventory.read_bytes()).hexdigest()
    dataset = root / "metadata/dataset.json"
    dataset.write_text(
        json.dumps(
            {
                "dataset_id": "tess_cached_unlabeled_representation_v1",
                "role": "training",
                "local_path": "metadata/inventory.jsonl",
                "sha256": digest,
                "row_count": len(rows),
                "group_count": len({row["group_key"] for row in rows}),
            }
        ),
        encoding="utf-8",
    )
    summary = root / "metadata/summary.json"
    summary.write_text(
        json.dumps(
            {
                "inventory_id": "tess_cached_unlabeled_representation_v1",
                "role": "training",
                "rows_sha256": digest,
                "eligible_product_count": len(rows),
                "eligible_bytes": sum(int(row["size_bytes"]) for row in rows),
            }
        ),
        encoding="utf-8",
    )
    return dataset, summary


def _write_fits(path: Path, *, cadences: int = 64) -> int:
    path.parent.mkdir(parents=True)
    times = np.linspace(100.0, 102.0, cadences)
    flux = 1000.0 + 7.0 * np.sin(np.linspace(0.0, 5.0, cadences))
    flux[3] = np.nan
    quality = np.zeros(cadences, dtype=np.int32)
    quality[5] = 1
    table = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="TIME", format="D", array=times),
            fits.Column(name="PDCSAP_FLUX", format="D", array=flux),
            fits.Column(name="QUALITY", format="J", array=quality),
        ]
    )
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(path)
    return path.stat().st_size


def test_load_inventory_contract_rejects_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [_row(0)]
    dataset, summary = _write_contract(tmp_path, rows)
    monkeypatch.setattr(benchmark, "REPO_ROOT", tmp_path)
    inventory = tmp_path / "metadata/inventory.jsonl"
    inventory.write_text(inventory.read_text() + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_inventory_contract(dataset, summary)


def test_select_target_balanced_sample_is_deterministic_and_cross_sector() -> None:
    rows = [_row(index, sector=(index % 12) + 1) for index in range(72)]
    rows.extend(
        {**rows[index], "cache_relative_path": f"alternate_{index}.fits"} for index in range(8)
    )

    first = select_target_balanced_sample(rows, 12)
    second = select_target_balanced_sample(list(reversed(rows)), 12)

    assert [row["cache_relative_path"] for row in first] == [
        row["cache_relative_path"] for row in second
    ]
    assert len({row["group_key"] for row in first}) == 12
    assert min(int(row["sector"]) for row in first) == 1
    assert max(int(row["sector"]) for row in first) == 12


def test_preprocess_product_filters_quality_and_discards_derived_array(tmp_path: Path) -> None:
    relative_path = Path("product/product_lc.fits")
    size = _write_fits(tmp_path / relative_path)
    row = _row(0, size_bytes=size)
    row["cache_relative_path"] = str(relative_path)

    result = preprocess_product(row, tmp_path, output_bins=32)

    assert result.status == "success"
    assert result.input_cadences == 64
    assert result.retained_cadences == 62
    assert result.output_bins == 32
    assert result.output_bytes == 32 * np.dtype(np.float32).itemsize
    assert result.output_sha256 is not None
    assert not hasattr(result, "array")


def test_process_sample_uses_six_by_six_and_processes_every_row_once() -> None:
    rows = [_row(index) for index in range(36)]
    calls: list[str] = []

    def fake_processor(row: dict[str, Any], _cache_root: Path, output_bins: int) -> ProductResult:
        calls.append(str(row["cache_relative_path"]))
        time.sleep(0.001)
        return ProductResult(
            target_id=str(row["target_id"]),
            sector=int(row["sector"]),
            cache_relative_path=str(row["cache_relative_path"]),
            input_bytes=int(row["size_bytes"]),
            input_cadences=100,
            retained_cadences=90,
            output_bins=output_bins,
            output_bytes=output_bins * 4,
            output_sha256="a" * 64,
            worker_peak_rss_bytes=1_000_000,
            elapsed_seconds=0.001,
            status="success",
        )

    results = process_sample(
        rows,
        Path("unused"),
        64,
        shard_count=6,
        workers_per_shard=6,
        execution_backend="thread",
        processor=fake_processor,
    )

    assert len(results) == len(calls) == 36
    assert len(set(calls)) == 36
    assert all(result.status == "success" for result in results)


def test_process_sample_supervises_six_subprocess_shards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [_row(index) for index in range(36)]
    commands: list[list[str]] = []

    class FakeProcess:
        pid = 1234

        def __init__(self, command: list[str], **_kwargs: object) -> None:
            commands.append(command)
            input_path = Path(command[command.index("--internal-shard-input") + 1])
            output_path = Path(command[command.index("--internal-shard-output") + 1])
            output_bins = int(command[command.index("--output-bins") + 1])
            shard_rows = json.loads(input_path.read_text(encoding="utf-8"))
            results = [
                ProductResult(
                    target_id=str(row["target_id"]),
                    sector=int(row["sector"]),
                    cache_relative_path=str(row["cache_relative_path"]),
                    input_bytes=int(row["size_bytes"]),
                    input_cadences=100,
                    retained_cadences=90,
                    output_bins=output_bins,
                    output_bytes=output_bins * 4,
                    output_sha256="c" * 64,
                    worker_peak_rss_bytes=3_000_000,
                    elapsed_seconds=0.01,
                    status="success",
                )
                for row in shard_rows
            ]
            output_path.write_text(
                json.dumps([result.__dict__ for result in results]),
                encoding="utf-8",
            )

        def wait(self) -> int:
            return 0

        def poll(self) -> int:
            return 0

        def terminate(self) -> None:
            return None

    monkeypatch.setattr(benchmark.subprocess, "Popen", FakeProcess)

    results = process_sample(rows, Path("cache"), 64)

    assert len(commands) == 6
    assert len(results) == 36
    assert all(command[command.index("--workers") + 1] == "6" for command in commands)
    assert all(command[0] == benchmark.sys.executable for command in commands)


def test_process_sample_terminates_siblings_when_a_shard_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processes: list[Any] = []

    class FailingProcess:
        pid = 4321

        def __init__(self, _command: list[str], **_kwargs: object) -> None:
            self.returncode: int | None = 4 if not processes else None
            self.terminated = False
            processes.append(self)

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return self.returncode if self.returncode is not None else 0

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            self.terminated = True
            self.returncode = -15

        def kill(self) -> None:
            self.returncode = -9

    monkeypatch.setattr(benchmark.subprocess, "Popen", FailingProcess)

    with pytest.raises(RuntimeError, match="shard 1/6 exited with status 4"):
        process_sample([_row(index) for index in range(6)], Path("cache"), 64)

    assert len(processes) == 6
    assert all(process.terminated for process in processes[1:])


def test_run_benchmark_writes_small_projection_and_injects_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [_row(index, size_bytes=200) for index in range(12)]
    dataset, summary = _write_contract(tmp_path, rows)
    output = tmp_path / "benchmark.json"
    reports: list[object] = []
    monkeypatch.setattr(benchmark, "REPO_ROOT", tmp_path)

    def fake_processor(row: dict[str, Any], _cache_root: Path, output_bins: int) -> ProductResult:
        return ProductResult(
            target_id=str(row["target_id"]),
            sector=int(row["sector"]),
            cache_relative_path=str(row["cache_relative_path"]),
            input_bytes=int(row["size_bytes"]),
            input_cadences=80,
            retained_cadences=70,
            output_bins=output_bins,
            output_bytes=output_bins * 4,
            output_sha256="b" * 64,
            worker_peak_rss_bytes=2_000_000,
            elapsed_seconds=0.01,
            status="success",
        )

    result = run_benchmark(
        dataset,
        summary,
        tmp_path / "cache",
        output,
        sample_products=12,
        output_bins=128,
        shard_count=6,
        workers_per_shard=6,
        execution_backend="thread",
        processor=fake_processor,
        report_fn=lambda report, path: reports.append((report, path)) or True,
    )

    assert result["status"] == "success"
    assert result["downloads_performed"] == 0
    assert result["derived_arrays_persisted"] == 0
    assert result["aggregate"]["projected_full_inventory_derived_bytes"] == 12 * 128 * 4
    assert result["configuration"]["maximum_concurrent_workers"] == 36
    assert output.stat().st_size < 50_000
    assert len(reports) == 1


def test_run_benchmark_fails_closed_and_does_not_commit_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [_row(index) for index in range(6)]
    dataset, summary = _write_contract(tmp_path, rows)
    monkeypatch.setattr(benchmark, "REPO_ROOT", tmp_path)
    reports: list[object] = []

    def failing_processor(
        row: dict[str, Any], _cache_root: Path, output_bins: int
    ) -> ProductResult:
        return ProductResult(
            target_id=str(row["target_id"]),
            sector=int(row["sector"]),
            cache_relative_path=str(row["cache_relative_path"]),
            input_bytes=int(row["size_bytes"]),
            input_cadences=0,
            retained_cadences=0,
            output_bins=output_bins,
            output_bytes=0,
            output_sha256=None,
            worker_peak_rss_bytes=1_000_000,
            elapsed_seconds=0.0,
            status="failed",
            error="synthetic failure",
        )

    result = run_benchmark(
        dataset,
        summary,
        tmp_path / "cache",
        tmp_path / "failed.json",
        sample_products=6,
        shard_count=6,
        workers_per_shard=6,
        execution_backend="thread",
        processor=failing_processor,
        report_fn=lambda *args: reports.append(args) or True,
    )

    assert result["status"] == "failed"
    assert result["aggregate"]["products_failed"] == 6
    assert reports == []
