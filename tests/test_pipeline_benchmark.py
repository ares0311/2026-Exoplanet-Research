"""Tests for Skills.pipeline_benchmark."""
from __future__ import annotations

import pytest
from Skills.pipeline_benchmark import (
    BenchmarkResult,
    StageResult,
    benchmark_pipeline,
    format_benchmark_result,
)


class TestBenchmarkPipeline:
    def test_returns_benchmark_result(self) -> None:
        r = benchmark_pipeline(1)
        assert isinstance(r, BenchmarkResult)

    def test_tic_id_stored(self) -> None:
        r = benchmark_pipeline(42)
        assert r.tic_id == 42

    def test_no_fns_placeholder_stages(self) -> None:
        r = benchmark_pipeline(1)
        assert len(r.stages) > 0

    def test_total_seconds_sum(self) -> None:
        r = benchmark_pipeline(1)
        total = sum(s.elapsed_seconds for s in r.stages)
        assert r.total_seconds == pytest.approx(total, abs=1e-6)

    def test_run_fn_single_stage(self) -> None:
        def run(tic_id, mission="TESS"):
            pass
        r = benchmark_pipeline(1, run_fn=run)
        assert len(r.stages) == 1
        assert r.stages[0].name == "full_pipeline"

    def test_run_fn_success(self) -> None:
        r = benchmark_pipeline(1, run_fn=lambda tic_id, mission="TESS": None)
        assert r.stages[0].success is True

    def test_run_fn_exception_recorded(self) -> None:
        def bad(tic_id, mission="TESS"):
            raise ValueError("boom")
        r = benchmark_pipeline(1, run_fn=bad)
        assert r.stages[0].success is False
        assert "boom" in r.stages[0].error_message

    def test_stage_fns_called_in_order(self) -> None:
        called: list[str] = []
        fns = {
            "fetch": lambda tic_id, mission="TESS": called.append("fetch"),
            "clean": lambda tic_id, mission="TESS": called.append("clean"),
        }
        benchmark_pipeline(1, stage_fns=fns)
        assert called == ["fetch", "clean"]

    def test_stage_fns_results_recorded(self) -> None:
        fns = {
            "fetch": lambda tic_id, mission="TESS": None,
            "clean": lambda tic_id, mission="TESS": None,
        }
        r = benchmark_pipeline(1, stage_fns=fns)
        assert len(r.stages) == 2

    def test_elapsed_seconds_nonnegative(self) -> None:
        r = benchmark_pipeline(1)
        assert all(s.elapsed_seconds >= 0.0 for s in r.stages)

    def test_generated_at_is_string(self) -> None:
        r = benchmark_pipeline(1)
        assert isinstance(r.generated_at, str)


class TestFormatBenchmarkResult:
    def test_returns_string(self) -> None:
        r = benchmark_pipeline(1)
        assert isinstance(format_benchmark_result(r), str)

    def test_contains_tic_id(self) -> None:
        r = benchmark_pipeline(99)
        assert "99" in format_benchmark_result(r)
