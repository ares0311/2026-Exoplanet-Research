"""Tests for Skills.fetch_kepler_lc_snippets (13 tests)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from Skills.fetch_kepler_lc_snippets import (
    _mad,
    _median,
    _normalise,
    _phase_fold_bin,
    build_kepler_snippet,
    build_kepler_snippets,
)

# ---------------------------------------------------------------------------
# Phase-fold helpers
# ---------------------------------------------------------------------------


class TestPhaseFoldBin:
    def test_constant_flux_returns_ones(self) -> None:
        time_bjd = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0] * 30
        flux = [1.0] * len(time_bjd)
        bins = _phase_fold_bin(time_bjd, flux, period=2.0, epoch=0.0, n_bins=10)
        assert len(bins) == 10
        for v in bins:
            assert abs(v - 1.0) < 1e-6

    def test_output_length_equals_n_bins(self) -> None:
        time_bjd = list(range(100))
        flux = [1.0] * 100
        for n in (11, 51, 201):
            bins = _phase_fold_bin(time_bjd, flux, period=5.0, epoch=0.0, n_bins=n)
            assert len(bins) == n

    def test_empty_bin_filled_with_one(self) -> None:
        # Single point: only one bin occupied
        bins = _phase_fold_bin([0.0], [0.5], period=10.0, epoch=0.0, n_bins=5)
        assert len(bins) == 5
        assert 1.0 in bins  # empty bins default to 1.0


class TestNormalise:
    def test_zero_mad_returns_zeros(self) -> None:
        result = _normalise([1.0] * 10)
        assert all(v == 0.0 for v in result)

    def test_unit_scale(self) -> None:
        flux = [1.0] * 100 + [0.9] * 5
        result = _normalise(flux)
        assert len(result) == 105
        med = _median(result)
        assert abs(med) < 0.5

    def test_median_helper(self) -> None:
        assert _median([1.0, 2.0, 3.0]) == 2.0
        assert _median([1.0, 2.0]) == 1.5

    def test_mad_helper(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        med = _median(values)
        assert abs(_mad(values, med) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# build_kepler_snippet
# ---------------------------------------------------------------------------


class TestBuildKeplerSnippet:
    def _make_fetcher(self, n_points: int = 500):
        """Return a fetcher that yields n_points of flat flux."""
        times = [float(i) * 0.0208333 for i in range(n_points)]  # ~30-min cadence
        flux = [1.0 - 0.01 * (1 if abs(i % 100 - 50) < 5 else 0) for i in range(n_points)]
        def fetcher(kepid: int, period: float, epoch_bjd: float):
            return times, flux
        return fetcher

    def test_ok_flag_on_valid_data(self) -> None:
        fetcher = self._make_fetcher(500)
        result = build_kepler_snippet(
            757450, 1, 2.204, 2454900.0, n_bins=201, lc_fetcher=fetcher
        )
        assert result.flag == "OK"
        assert len(result.flux) == 201
        assert result.kepid == 757450
        assert result.label == 1

    def test_no_lightkurve_flag_when_none_returned(self) -> None:
        def fetcher(kepid, period, epoch):
            return None
        result = build_kepler_snippet(
            1, 0, 3.0, 2454900.0, n_bins=201, lc_fetcher=fetcher
        )
        assert result.flag in {"NO_LIGHTKURVE", "NO_DATA"}

    def test_short_flag_when_too_few_points(self) -> None:
        def fetcher(kepid, period, epoch):
            return [0.0, 1.0], [1.0, 1.0]
        result = build_kepler_snippet(
            1, 0, 3.0, 2454900.0, n_bins=201, lc_fetcher=fetcher
        )
        assert result.flag == "SHORT"

    def test_error_flag_on_exception(self) -> None:
        def fetcher(kepid, period, epoch):
            raise RuntimeError("connection refused")
        result = build_kepler_snippet(
            1, 0, 3.0, 2454900.0, n_bins=201, lc_fetcher=fetcher
        )
        assert result.flag.startswith("ERROR")

    def test_result_is_frozen_dataclass(self) -> None:
        fetcher = self._make_fetcher(500)
        result = build_kepler_snippet(
            1, 1, 2.0, 2454900.0, n_bins=11, lc_fetcher=fetcher
        )
        with pytest.raises(AttributeError):
            result.label = 0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# build_kepler_snippets
# ---------------------------------------------------------------------------


class TestBuildKeplerSnippets:
    def _make_koi_rows(self, n: int = 3) -> list[dict]:
        return [
            {
                "kepid": str(100 + i),
                "koi_disposition": "CONFIRMED" if i % 2 == 0 else "FALSE POSITIVE",
                "koi_period": "3.5",
                "koi_time0bk": "100.0",
            }
            for i in range(n)
        ]

    def _make_fetcher(self, n_points: int = 500):
        times = [float(j) * 0.02 for j in range(n_points)]
        flux = [1.0] * n_points
        def fetcher(kepid, period, epoch):
            return times, flux
        return fetcher

    def test_writes_ok_snippets(self, tmp_path: Path) -> None:
        koi_rows = self._make_koi_rows(3)
        out = tmp_path / "kepler_snippets.jsonl"
        n = build_kepler_snippets(
            koi_rows,
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
        koi_rows = self._make_koi_rows(4)
        out = tmp_path / "kepler_snippets.jsonl"
        # Write the first two
        existing = [
            json.dumps({"kepid": 100, "label": 1, "flux": [], "source": "kepler",
                        "period_days": 3.5, "epoch_bjd": 2454933.0, "tic_id": 0, "n_bins": 11}),
            json.dumps({"kepid": 101, "label": 0, "flux": [], "source": "kepler",
                        "period_days": 3.5, "epoch_bjd": 2454933.0, "tic_id": 0, "n_bins": 11}),
        ]
        out.write_text("\n".join(existing) + "\n")
        n = build_kepler_snippets(
            koi_rows,
            n_bins=11,
            output_path=out,
            lc_fetcher=self._make_fetcher(),
            resume=True,
            max_errors=10,
        )
        # Only 2 new ones written (102 and 103)
        assert n == 2

    def test_no_resume_overwrites(self, tmp_path: Path) -> None:
        out = tmp_path / "out.jsonl"
        out.write_text('{"kepid": 999, "label": 1, "flux": []}\n')
        build_kepler_snippets(
            self._make_koi_rows(1),
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
        build_kepler_snippets(
            self._make_koi_rows(1),
            n_bins=11,
            output_path=out,
            lc_fetcher=self._make_fetcher(),
            resume=False,
            max_errors=10,
        )
        record = json.loads(out.read_text().strip().split("\n")[0])
        for key in ("kepid", "label", "flux", "source", "period_days", "epoch_bjd", "n_bins"):
            assert key in record
        assert record["source"] == "kepler"
        assert record["n_bins"] == 11
