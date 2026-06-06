"""Tests for Skills/download_tess_lightcurves.py (offline / unit tests only)."""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from Skills.download_tess_lightcurves import (
    _DISPOSITION_LABEL,
    _extract_snippet,
    _load_done_tic_ids,
    _load_toi_rows,
    _phase_fold_bin,
    download_and_extract,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_toi_csv(path: Path, rows: list[dict]) -> Path:
    """Write a minimal TOI CSV to *path*."""
    fieldnames = [
        "toi", "tic_id", "tfopwg_disposition", "period_days", "epoch_bjd",
        "duration_hours", "snr",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _sinusoidal_lc(n: int = 500, period: float = 10.0) -> tuple[list[float], list[float]]:
    """Return (time_jd, flux) with a shallow transit at phase 0."""
    import math as m
    time = [2458300.0 + i * 0.0208 for i in range(n)]  # 30-min cadence
    flux = [1.0 - 0.01 * (1.0 if m.cos(2 * m.pi * (t - 2458300.0) / period) > 0.98 else 0.0)
            for t in time]
    return time, flux


def _mock_lc_fn(tic_id: int) -> tuple[list[float], list[float]]:
    return _sinusoidal_lc()


def _failing_lc_fn(tic_id: int) -> tuple[list[float], list[float]]:
    raise ValueError("No LC found")


# ---------------------------------------------------------------------------
# _DISPOSITION_LABEL
# ---------------------------------------------------------------------------


class TestDispositionLabel:
    def test_cp_is_positive(self) -> None:
        assert _DISPOSITION_LABEL["CP"] == 1

    def test_kp_is_positive(self) -> None:
        assert _DISPOSITION_LABEL["KP"] == 1

    def test_fp_is_negative(self) -> None:
        assert _DISPOSITION_LABEL["FP"] == 0

    def test_fa_is_negative(self) -> None:
        assert _DISPOSITION_LABEL["FA"] == 0

    def test_pc_not_in_map(self) -> None:
        assert "PC" not in _DISPOSITION_LABEL


# ---------------------------------------------------------------------------
# _load_toi_rows
# ---------------------------------------------------------------------------


class TestLoadToiRows:
    def test_returns_labelled_rows(self, tmp_path: Path) -> None:
        csv_path = _make_toi_csv(tmp_path / "toi.csv", [
            {"toi": "700.01", "tic_id": "150428135", "tfopwg_disposition": "CP",
             "period_days": "37.4", "epoch_bjd": "2458325.0", "duration_hours": "2.3", "snr": "22"},
        ])
        rows = _load_toi_rows(csv_path)
        assert len(rows) == 1
        assert rows[0]["tic_id"] == 150428135
        assert rows[0]["label"] == 1
        assert rows[0]["period_days"] == pytest.approx(37.4)

    def test_filters_pc_disposition(self, tmp_path: Path) -> None:
        csv_path = _make_toi_csv(tmp_path / "toi.csv", [
            {"toi": "1.01", "tic_id": "111", "tfopwg_disposition": "PC",
             "period_days": "5.0", "epoch_bjd": "2458300.0", "duration_hours": "1.0", "snr": "10"},
        ])
        rows = _load_toi_rows(csv_path)
        assert rows == []

    def test_filters_zero_period(self, tmp_path: Path) -> None:
        csv_path = _make_toi_csv(tmp_path / "toi.csv", [
            {"toi": "1.01", "tic_id": "222", "tfopwg_disposition": "CP",
             "period_days": "0.0", "epoch_bjd": "2458300.0", "duration_hours": "1.0", "snr": "10"},
        ])
        rows = _load_toi_rows(csv_path)
        assert rows == []

    def test_handles_missing_epoch(self, tmp_path: Path) -> None:
        csv_path = _make_toi_csv(tmp_path / "toi.csv", [
            {"toi": "1.01", "tic_id": "333", "tfopwg_disposition": "FP",
             "period_days": "3.1", "epoch_bjd": "", "duration_hours": "1.0", "snr": "5"},
        ])
        rows = _load_toi_rows(csv_path)
        assert len(rows) == 1
        assert rows[0]["epoch_bjd"] == pytest.approx(0.0)

    def test_kp_label_is_positive(self, tmp_path: Path) -> None:
        csv_path = _make_toi_csv(tmp_path / "toi.csv", [
            {"toi": "2.01", "tic_id": "444", "tfopwg_disposition": "KP",
             "period_days": "10.0", "epoch_bjd": "2458320.0", "duration_hours": "2.0", "snr": "30"},
        ])
        rows = _load_toi_rows(csv_path)
        assert rows[0]["label"] == 1

    def test_fa_label_is_negative(self, tmp_path: Path) -> None:
        csv_path = _make_toi_csv(tmp_path / "toi.csv", [
            {"toi": "3.01", "tic_id": "555", "tfopwg_disposition": "FA",
             "period_days": "1.0", "epoch_bjd": "2458300.0", "duration_hours": "0.5", "snr": "2"},
        ])
        rows = _load_toi_rows(csv_path)
        assert rows[0]["label"] == 0


# ---------------------------------------------------------------------------
# _load_done_tic_ids
# ---------------------------------------------------------------------------


class TestLoadDoneTicIds:
    def test_empty_when_file_missing(self, tmp_path: Path) -> None:
        done = _load_done_tic_ids(tmp_path / "nonexistent.jsonl")
        assert done == set()

    def test_reads_ok_records(self, tmp_path: Path) -> None:
        p = tmp_path / "out.jsonl"
        p.write_text(
            json.dumps({"tic_id": 123, "status": "ok"}) + "\n"
            + json.dumps({"tic_id": 456, "status": "error"}) + "\n"
        )
        done = _load_done_tic_ids(p)
        assert 123 in done
        assert 456 in done

    def test_skips_malformed_lines(self, tmp_path: Path) -> None:
        p = tmp_path / "out.jsonl"
        p.write_text("not-json\n" + json.dumps({"tic_id": 789, "status": "ok"}) + "\n")
        done = _load_done_tic_ids(p)
        assert done == {789}


# ---------------------------------------------------------------------------
# _phase_fold_bin
# ---------------------------------------------------------------------------


class TestPhaseFoldBin:
    def test_output_length_matches_n_bins(self) -> None:
        time = list(range(500))
        flux = [1.0] * 500
        centers, means = _phase_fold_bin(time, flux, 10.0, 0.0, 201)
        assert len(centers) == 201
        assert len(means) == 201

    def test_phase_range(self) -> None:
        time = list(range(1000))
        flux = [1.0] * 1000
        centers, _ = _phase_fold_bin(time, flux, 10.0, 0.0, 201)
        assert centers[0] == pytest.approx(-0.5 + 0.5 / 201, rel=1e-3)
        assert centers[-1] < 0.5

    def test_empty_bins_filled_with_one(self) -> None:
        # A very short light curve that won't cover all bins
        time = [2458300.0]
        flux = [0.9]
        _, means = _phase_fold_bin(time, flux, 100.0, 2458300.0, 201)
        # Most bins will be empty → filled with 1.0
        ones = sum(1 for m in means if m == 1.0)
        assert ones > 100


# ---------------------------------------------------------------------------
# _extract_snippet
# ---------------------------------------------------------------------------


class TestExtractSnippet:
    def test_returns_correct_length(self) -> None:
        time, flux = _sinusoidal_lc(500, period=10.0)
        result = _extract_snippet(time, flux, 10.0, 2458300.0, 201)
        assert result is not None
        phase, bins = result
        assert len(phase) == 201
        assert len(bins) == 201

    def test_returns_none_for_insufficient_data(self) -> None:
        result = _extract_snippet([1.0], [1.0], 10.0, 0.0, 201)
        assert result is None

    def test_returns_none_for_zero_period(self) -> None:
        time, flux = _sinusoidal_lc(500)
        result = _extract_snippet(time, flux, 0.0, 2458300.0, 201)
        assert result is None

    def test_flux_values_near_one_for_flat_lc(self) -> None:
        time = [2458300.0 + i * 0.02 for i in range(500)]
        flux = [1.0] * 500
        result = _extract_snippet(time, flux, 10.0, 2458300.0, 201)
        assert result is not None
        _, bins = result
        assert all(math.isfinite(b) for b in bins)
        # All bins should be ≈ 1.0 for flat LC
        assert all(abs(b - 1.0) < 0.01 for b in bins)


# ---------------------------------------------------------------------------
# download_and_extract
# ---------------------------------------------------------------------------


class TestDownloadAndExtract:
    def _make_csv(self, tmp_path: Path) -> Path:
        return _make_toi_csv(tmp_path / "toi.csv", [
            {"toi": "700.01", "tic_id": "150428135", "tfopwg_disposition": "CP",
             "period_days": "37.4", "epoch_bjd": "2458325.0", "duration_hours": "2.3", "snr": "22"},
            {"toi": "100.01", "tic_id": "999", "tfopwg_disposition": "FP",
             "period_days": "3.1", "epoch_bjd": "2458310.0", "duration_hours": "1.0", "snr": "5"},
        ])

    def test_creates_output_file(self, tmp_path: Path) -> None:
        csv_path = self._make_csv(tmp_path)
        out = tmp_path / "snippets.jsonl"
        download_and_extract(csv_path, out, lc_fetch_fn=_mock_lc_fn, sleep_between=0)
        assert out.exists()

    def test_ok_count(self, tmp_path: Path) -> None:
        csv_path = self._make_csv(tmp_path)
        out = tmp_path / "snippets.jsonl"
        result = download_and_extract(csv_path, out, lc_fetch_fn=_mock_lc_fn, sleep_between=0)
        assert result["n_ok"] == 2
        assert result["n_error"] == 0

    def test_error_count_on_failing_fetcher(self, tmp_path: Path) -> None:
        csv_path = self._make_csv(tmp_path)
        out = tmp_path / "snippets.jsonl"
        result = download_and_extract(csv_path, out, lc_fetch_fn=_failing_lc_fn, sleep_between=0)
        assert result["n_error"] == 2
        assert result["n_ok"] == 0

    def test_output_records_have_required_keys(self, tmp_path: Path) -> None:
        csv_path = self._make_csv(tmp_path)
        out = tmp_path / "snippets.jsonl"
        download_and_extract(csv_path, out, lc_fetch_fn=_mock_lc_fn, sleep_between=0)
        records = [json.loads(line) for line in out.read_text().splitlines()]
        for rec in records:
            if rec["status"] == "ok":
                for key in ("tic_id", "label", "disposition", "period_days",
                            "epoch_bjd", "phase", "flux", "n_points", "status"):
                    assert key in rec, f"Missing key: {key}"

    def test_phase_array_length(self, tmp_path: Path) -> None:
        csv_path = self._make_csv(tmp_path)
        out = tmp_path / "snippets.jsonl"
        download_and_extract(csv_path, out, n_bins=201, lc_fetch_fn=_mock_lc_fn, sleep_between=0)
        records = [json.loads(line) for line in out.read_text().splitlines()]
        for rec in records:
            if rec["status"] == "ok":
                assert len(rec["phase"]) == 201
                assert len(rec["flux"]) == 201

    def test_resume_skips_done_tic_ids(self, tmp_path: Path) -> None:
        csv_path = self._make_csv(tmp_path)
        out = tmp_path / "snippets.jsonl"
        # First run: process all
        download_and_extract(csv_path, out, lc_fetch_fn=_mock_lc_fn, sleep_between=0)
        # Second run with resume: should skip all
        result = download_and_extract(
            csv_path, out, resume=True, lc_fetch_fn=_mock_lc_fn, sleep_between=0
        )
        assert result["n_skipped"] == 2
        assert result["n_ok"] == 0

    def test_max_targets_limits_processing(self, tmp_path: Path) -> None:
        csv_path = self._make_csv(tmp_path)
        out = tmp_path / "snippets.jsonl"
        result = download_and_extract(
            csv_path, out, max_targets=1, lc_fetch_fn=_mock_lc_fn, sleep_between=0
        )
        assert result["n_total"] == 1
        assert result["n_ok"] == 1

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        csv_path = self._make_csv(tmp_path)
        out = tmp_path / "nested" / "deep" / "snippets.jsonl"
        download_and_extract(csv_path, out, lc_fetch_fn=_mock_lc_fn, sleep_between=0)
        assert out.exists()

    def test_error_records_written_on_failure(self, tmp_path: Path) -> None:
        csv_path = self._make_csv(tmp_path)
        out = tmp_path / "snippets.jsonl"
        download_and_extract(csv_path, out, lc_fetch_fn=_failing_lc_fn, sleep_between=0)
        records = [json.loads(line) for line in out.read_text().splitlines()]
        assert all(r["status"] == "error" for r in records)
        assert all("reason" in r for r in records)

    def test_label_values_correct(self, tmp_path: Path) -> None:
        csv_path = self._make_csv(tmp_path)
        out = tmp_path / "snippets.jsonl"
        download_and_extract(csv_path, out, lc_fetch_fn=_mock_lc_fn, sleep_between=0)
        records = [json.loads(line) for line in out.read_text().splitlines() if line]
        ok = [r for r in records if r["status"] == "ok"]
        labels = {r["tic_id"]: r["label"] for r in ok}
        assert labels[150428135] == 1   # CP → positive
        assert labels[999] == 0         # FP → negative


# ---------------------------------------------------------------------------
# _cli
# ---------------------------------------------------------------------------


class TestCli:
    def test_exits_1_when_csv_missing(self, tmp_path: Path) -> None:
        from Skills.download_tess_lightcurves import _cli
        rc = _cli([
            "--toi-csv", str(tmp_path / "nonexistent.csv"),
            "--output", str(tmp_path / "out.jsonl"),
        ])
        assert rc == 1
