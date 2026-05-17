"""Benchmark individual pipeline stage timing.

Records wall-clock time for each pipeline stage (fetch, clean, search, vet,
score) and produces a Markdown timing table.

Public API
----------
StageResult(name, elapsed_seconds, success, error_message)
BenchmarkResult(tic_id, stages, total_seconds, generated_at)
benchmark_pipeline(tic_id, *, mission, run_fn, stage_fns) -> BenchmarkResult
format_benchmark_result(result) -> str
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass(frozen=True)
class StageResult:
    name: str
    elapsed_seconds: float
    success: bool
    error_message: str


@dataclass(frozen=True)
class BenchmarkResult:
    tic_id: int
    stages: tuple[StageResult, ...]
    total_seconds: float
    generated_at: str


_DEFAULT_STAGES = ["fetch", "clean", "search", "vet", "score"]


def _noop_stage(name: str, *args, **kwargs) -> None:
    """No-op placeholder stage function."""


def benchmark_pipeline(
    tic_id: int,
    *,
    mission: str = "TESS",
    run_fn=None,
    stage_fns: dict | None = None,
) -> BenchmarkResult:
    """Time each stage of the pipeline for a given target.

    Two modes:

    1. **Integrated** (``run_fn`` provided): calls ``run_fn(tic_id, mission=mission)``
       and records total wall time as a single "full_pipeline" stage.

    2. **Per-stage** (``stage_fns`` dict provided): each value is a callable
       ``fn(tic_id, **kwargs)``; called in order and timed individually.

    Args:
        tic_id: Target to benchmark.
        mission: TESS | Kepler | K2.
        run_fn: Optional single-call pipeline function.
        stage_fns: Optional ordered dict of stage_name → callable.

    Returns:
        :class:`BenchmarkResult`.
    """
    now_str = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    stages: list[StageResult] = []

    if run_fn is not None:
        t0 = time.perf_counter()
        err = ""
        success = True
        try:
            run_fn(tic_id, mission=mission)
        except Exception as exc:
            err = str(exc)
            success = False
        elapsed = time.perf_counter() - t0
        stages.append(StageResult("full_pipeline", round(elapsed, 4), success, err))

    elif stage_fns is not None:
        for name, fn in stage_fns.items():
            t0 = time.perf_counter()
            err = ""
            success = True
            try:
                fn(tic_id, mission=mission)
            except Exception as exc:
                err = str(exc)
                success = False
            elapsed = time.perf_counter() - t0
            stages.append(StageResult(name, round(elapsed, 4), success, err))

    else:
        # No functions provided — record zero-time placeholders
        for name in _DEFAULT_STAGES:
            stages.append(StageResult(name, 0.0, True, ""))

    total = sum(s.elapsed_seconds for s in stages)

    return BenchmarkResult(
        tic_id=tic_id,
        stages=tuple(stages),
        total_seconds=round(total, 4),
        generated_at=now_str,
    )


def format_benchmark_result(result: BenchmarkResult) -> str:
    """Format benchmark result as Markdown."""
    lines = [
        f"## Pipeline Benchmark — TIC {result.tic_id}",
        "",
        f"- Generated: {result.generated_at}",
        f"- Total time: {result.total_seconds:.3f} s",
        "",
        "| Stage | Time (s) | Status |",
        "|---|---|---|",
    ]
    for s in result.stages:
        status = "✓" if s.success else f"✗ {s.error_message[:40]}"
        lines.append(f"| {s.name} | {s.elapsed_seconds:.4f} | {status} |")
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="pipeline_benchmark",
        description="Benchmark pipeline stage timing for a target.",
    )
    parser.add_argument("tic_id", type=int)
    parser.add_argument("--mission", default="TESS")
    args = parser.parse_args(argv)

    result = benchmark_pipeline(args.tic_id, mission=args.mission)
    print(format_benchmark_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
