"""Tests for Skills.fetch_tess_lc_snippets (13 tests)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from Skills.fetch_tess_lc_snippets import (
    _mad,
    _median,
    _normalise,
    _phase_fold_bin,
    build_tess_snippet,
    build_tess_snippets,
)

# ---------------------------------------------------------------------------
# Phase-fold helpers
# ---------------------------------------------------------------------------


class TestPhaseFoldBin:
    def test_constant_flux_returns_ones(self) -> None:
        time_bjd = [2457000.0 + i * 0.5 for i in range(210)]
        flux = [1.0] * len(time_bjd)
        bins = _phase_fold_bin(time_bjd, flux, period=5.0, epoch=2457000.0, n_bins=10)
        assert len(bins) == 10
        for v in bins:
            assert abs(v - 1.0) < 1e-6

    def test_output_length_equals_n_bins(self) -> None:
        time_bjd = [2457000.0 + float(i) for i in range(100)]
        flux = [1.0] * 100
        for n in (11, 51, 201):
            bins = _phase_fold_bin(time_bjd, flux, period=5.0, epoch=2457000.0, n_bins=n)
            assert len(bins) == n

    def test_empty_bin_filled_with_one(self) -> None:
        bins = _phase_fold_bin([2457000.0], [0.5], period=10.0, epoch=2457000.0, n_bins=5)
        assert len(bins) == 5
        assert 1.0 in bins


class TestNormalise:
    def test_zero_mad_returns_zeros(self) -> None:
        result = _normalise([1.0] * 10)
        assert all(v == 0.0 for v in result)

    def test_median_helper(self) -> None:
        assert _median([1.0, 2.0, 3.0]) == 2.0
        assert _median([1.0, 2.0]) == 1.5

    def test_mad_helper(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        med = _median(values)
        assert abs(_mad(values, med) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# build_tess_snippet
# ---------------------------------------------------------------------------


class TestBuildTessSnippet:
    def _make_fetcher(self, n_points: int = 500):
        times = [2457000.0 + float(i) * 0.0208333 for i in range(n_points)]
        flux = [1.0 - 0.01 * (1 if abs(i % 100 - 50) < 5 else 0) for i in range(n_points)]
        def fetcher(tic_id: int, period: float, epoch_bjd: float):
            return times, flux
        return fetcher

    def test_ok_flag_on_valid_data(self) -> None:
        result = build_tess_snippet(
            150428135, 1, 9.9, 2458325.5, n_bins=201, lc_fetcher=self._make_fetcher()
        )
        assert result.flag == "OK"
        assert len(result.flux) == 201
        assert result.tic_id == 150428135
        assert result.label == 1

    def test_no_lightkurve_flag_when_none_returned(self) -> None:
        def fetcher(tic_id, period, epoch):
            return None
        result = build_tess_snippet(
            1, 0, 3.0, 2458000.0, n_bins=201, lc_fetcher=fetcher
        )
        assert result.flag in {"NO_LIGHTKURVE", "NO_DATA"}

    def test_short_flag_when_too_few_points(self) -> None:
        def fetcher(tic_id, period, epoch):
            return [2457000.0, 2457001.0], [1.0, 1.0]
        result = build_tess_snippet(
            1, 0, 3.0, 2458000.0, n_bins=201, lc_fetcher=fetcher
        )
        assert result.flag == "SHORT"

    def test_error_flag_on_exception(self) -> None:
        def fetcher(tic_id, period, epoch):
            raise RuntimeError("connection refused")
        result = build_tess_snippet(
            1, 0, 3.0, 2458000.0, n_bins=201, lc_fetcher=fetcher
        )
        assert result.flag.startswith("ERROR")

    def test_result_is_frozen_dataclass(self) -> None:
        result = build_tess_snippet(
            1, 1, 2.0, 2458000.0, n_bins=11, lc_fetcher=self._make_fetcher()
        )
        with pytest.raises(AttributeError):
            result.label = 0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# build_tess_snippets
# ---------------------------------------------------------------------------


class TestBuildTessSnippets:
    def _make_rows(self, n: int = 3) -> list[dict]:
        return [
            {
                "tic_id": 100 + i,
                "label": 1 if i % 2 == 0 else 0,
                "source": "exofop_toi",
                "period_days": 3.5,
                "epoch_bjd": 2458100.0,
            }
            for i in range(n)
        ]

    def _make_fetcher(self, n_points: int = 500):
        times = [2457000.0 + float(j) * 0.02 for j in range(n_points)]
        flux = [1.0] * n_points
        def fetcher(tic_id, period, epoch):
            return times, flux
        return fetcher

    def test_writes_ok_snippets(self, tmp_path: Path) -> None:
        out = tmp_path / "tess_snippets.jsonl"
        n = build_tess_snippets(
            self._make_rows(3),
            n_bins=11,
            output_path=out,
            lc_fetcher=self._make_fetcher(),
            resume=False,
            max_errors=10,
        )
        assert n == 3
        lines = [ln for ln in out.read_text().strip().split("\n") if ln]
        assert len(lines) == 3

    def test_resume_skips_already_written(self, tmp_path: Path) -> None:
        out = tmp_path / "tess_snippets.jsonl"
        existing = [
            json.dumps({"tic_id": 100, "label": 1, "flux": [], "source": "exofop_toi",
                        "period_days": 3.5, "epoch_bjd": 2458100.0, "n_bins": 11}),
            json.dumps({"tic_id": 101, "label": 0, "flux": [], "source": "exofop_toi",
                        "period_days": 3.5, "epoch_bjd": 2458100.0, "n_bins": 11}),
        ]
        out.write_text("\n".join(existing) + "\n")
        n = build_tess_snippets(
            self._make_rows(4),
            n_bins=11,
            output_path=out,
            lc_fetcher=self._make_fetcher(),
            resume=True,
            max_errors=10,
        )
        assert n == 2

    def test_no_resume_overwrites(self, tmp_path: Path) -> None:
        out = tmp_path / "out.jsonl"
        out.write_text('{"tic_id": 999, "label": 1, "flux": []}\n')
        build_tess_snippets(
            self._make_rows(1),
            n_bins=11,
            output_path=out,
            lc_fetcher=self._make_fetcher(),
            resume=False,
            max_errors=10,
        )
        lines = [ln for ln in out.read_text().strip().split("\n") if ln]
        assert all("999" not in ln for ln in lines)

    def test_jsonl_record_fields(self, tmp_path: Path) -> None:
        out = tmp_path / "out.jsonl"
        build_tess_snippets(
            self._make_rows(1),
            n_bins=11,
            output_path=out,
            lc_fetcher=self._make_fetcher(),
            resume=False,
            max_errors=10,
        )
        record = json.loads(out.read_text().strip().split("\n")[0])
        for key in ("tic_id", "label", "flux", "source", "period_days", "epoch_bjd", "n_bins"):
            assert key in record
        assert record["n_bins"] == 11

    def test_skips_zero_period_rows(self, tmp_path: Path) -> None:
        out = tmp_path / "out.jsonl"
        rows = [{"tic_id": 1, "label": 1, "period_days": 0.0, "epoch_bjd": 2458100.0}]
        n = build_tess_snippets(
            rows, n_bins=11, output_path=out,
            lc_fetcher=self._make_fetcher(), resume=False, max_errors=10,
        )
        assert n == 0
