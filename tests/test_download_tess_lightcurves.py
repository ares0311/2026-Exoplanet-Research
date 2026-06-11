"""Tests for Skills.download_tess_lightcurves."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from Skills.download_tess_lightcurves import (
    _load_checkpoint,
    _save_checkpoint,
    download_tess_lightcurves,
    load_toi_csv,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_toi_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = ["tic_id", "period_days", "epoch_bjd", "tfopwg_disposition"]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _good_row(tic_id: int = 12345, disp: str = "CP") -> dict:
    return {
        "tic_id": tic_id,
        "period_days": 5.0,
        "epoch_bjd": 2458100.5,
        "tfopwg_disposition": disp,
    }


def _fake_ok(row: dict) -> dict:
    return {
        "tic_id": row["tic_id"],
        "label": row["label"],
        "period_days": row["period_days"],
        "epoch_bjd": row["epoch_bjd"],
        "phase": [round(-0.5 + i / 201, 6) for i in range(201)],
        "flux": [1.0] * 201,
        "source": "tess",
        "normalization": "local_median_mad",
    }


def _fake_fail(row: dict) -> None:
    return None


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


class TestCheckpoint:
    def test_load_missing_returns_empty(self, tmp_path: Path) -> None:
        cp = _load_checkpoint(tmp_path / "cp.json")
        assert cp == {"completed": set(), "failed": set()}

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "cp.json"
        _save_checkpoint(path, {"123", "456"}, {"789"})
        cp = _load_checkpoint(path)
        assert cp["completed"] == {"123", "456"}
        assert cp["failed"] == {"789"}

    def test_save_overwrites_existing(self, tmp_path: Path) -> None:
        path = tmp_path / "cp.json"
        _save_checkpoint(path, {"a"}, set())
        _save_checkpoint(path, {"a", "b"}, set())
        cp = _load_checkpoint(path)
        assert "b" in cp["completed"]

    def test_load_corrupt_returns_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "cp.json"
        path.write_text("not-json")
        cp = _load_checkpoint(path)
        assert cp["completed"] == set()
        assert cp["failed"] == set()

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "a" / "b" / "cp.json"
        _save_checkpoint(path, set(), set())
        assert path.exists()


# ---------------------------------------------------------------------------
# load_toi_csv
# ---------------------------------------------------------------------------


class TestLoadToiCsv:
    def test_loads_valid_rows(self, tmp_path: Path) -> None:
        path = tmp_path / "toi.csv"
        _write_toi_csv(path, [_good_row(12345)])
        rows = load_toi_csv(path)
        assert len(rows) == 1
        assert rows[0]["tic_id"] == 12345
        assert rows[0]["epoch_bjd"] == pytest.approx(2458100.5)

    def test_rejects_zero_epoch(self, tmp_path: Path) -> None:
        path = tmp_path / "toi.csv"
        _write_toi_csv(path, [{**_good_row(), "epoch_bjd": 0.0}])
        assert load_toi_csv(path) == []

    def test_rejects_empty_epoch(self, tmp_path: Path) -> None:
        path = tmp_path / "toi.csv"
        _write_toi_csv(path, [{**_good_row(), "epoch_bjd": ""}])
        assert load_toi_csv(path) == []

    def test_cp_maps_to_label_1(self, tmp_path: Path) -> None:
        path = tmp_path / "toi.csv"
        _write_toi_csv(path, [_good_row(disp="CP")])
        assert load_toi_csv(path)[0]["label"] == 1

    def test_kp_maps_to_label_1(self, tmp_path: Path) -> None:
        path = tmp_path / "toi.csv"
        _write_toi_csv(path, [_good_row(disp="KP")])
        assert load_toi_csv(path)[0]["label"] == 1

    def test_fp_maps_to_label_0(self, tmp_path: Path) -> None:
        path = tmp_path / "toi.csv"
        _write_toi_csv(path, [_good_row(disp="FP")])
        assert load_toi_csv(path)[0]["label"] == 0

    def test_fa_maps_to_label_0(self, tmp_path: Path) -> None:
        path = tmp_path / "toi.csv"
        _write_toi_csv(path, [_good_row(disp="FA")])
        assert load_toi_csv(path)[0]["label"] == 0

    def test_skips_bad_tic_id(self, tmp_path: Path) -> None:
        path = tmp_path / "toi.csv"
        _write_toi_csv(path, [{**_good_row(), "tic_id": 0}])
        assert load_toi_csv(path) == []

    def test_source_is_tess(self, tmp_path: Path) -> None:
        path = tmp_path / "toi.csv"
        _write_toi_csv(path, [_good_row()])
        assert load_toi_csv(path)[0]["source"] == "tess"


# ---------------------------------------------------------------------------
# download_tess_lightcurves
# ---------------------------------------------------------------------------


class TestDownloadTessLightcurves:
    def test_writes_jsonl(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "toi.csv"
        _write_toi_csv(csv_path, [_good_row(111), _good_row(222)])
        out = tmp_path / "out.jsonl"
        download_tess_lightcurves(
            csv_path, out,
            checkpoint_path=tmp_path / "cp.json",
            workers=1, _download_fn=_fake_ok,
        )
        lines = out.read_text().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["tic_id"] in {111, 222}

    def test_resume_skips_completed(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "toi.csv"
        _write_toi_csv(csv_path, [_good_row(111), _good_row(222)])
        out = tmp_path / "out.jsonl"
        cp = tmp_path / "cp.json"
        _save_checkpoint(cp, {"111"}, set())

        result = download_tess_lightcurves(
            csv_path, out,
            checkpoint_path=cp, workers=1, _download_fn=_fake_ok,
        )
        assert result["n_skipped"] == 1
        lines = out.read_text().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["tic_id"] == 222

    def test_no_resume_overwrites_output(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "toi.csv"
        _write_toi_csv(csv_path, [_good_row(111)])
        out = tmp_path / "out.jsonl"
        out.write_text('{"tic_id": 999}\n')
        download_tess_lightcurves(
            csv_path, out,
            checkpoint_path=tmp_path / "cp.json",
            workers=1, resume=False, _download_fn=_fake_ok,
        )
        lines = out.read_text().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["tic_id"] == 111

    def test_failed_downloads_tracked_in_checkpoint(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "toi.csv"
        _write_toi_csv(csv_path, [_good_row(999)])
        out = tmp_path / "out.jsonl"
        cp = tmp_path / "cp.json"
        download_tess_lightcurves(
            csv_path, out,
            checkpoint_path=cp, workers=1, _download_fn=_fake_fail,
        )
        state = _load_checkpoint(cp)
        assert "999" in state["failed"]
        assert out.read_text().strip() == ""

    def test_limit_caps_processing(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "toi.csv"
        _write_toi_csv(csv_path, [_good_row(i) for i in range(1, 11)])
        out = tmp_path / "out.jsonl"
        result = download_tess_lightcurves(
            csv_path, out,
            checkpoint_path=tmp_path / "cp.json",
            workers=1, limit=3, _download_fn=_fake_ok,
        )
        assert result["n_attempted"] == 3

    def test_empty_csv_returns_no_data(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "toi.csv"
        _write_toi_csv(csv_path, [])
        result = download_tess_lightcurves(
            csv_path, tmp_path / "out.jsonl",
            checkpoint_path=tmp_path / "cp.json",
            workers=1, _download_fn=_fake_ok,
        )
        assert result["flag"] == "NO_DATA"

    def test_all_zero_epochs_returns_no_data(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "toi.csv"
        _write_toi_csv(csv_path, [{**_good_row(), "epoch_bjd": 0.0}])
        result = download_tess_lightcurves(
            csv_path, tmp_path / "out.jsonl",
            checkpoint_path=tmp_path / "cp.json",
            workers=1, _download_fn=_fake_ok,
        )
        assert result["flag"] == "NO_DATA"

    def test_summary_keys_present(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "toi.csv"
        _write_toi_csv(csv_path, [_good_row()])
        result = download_tess_lightcurves(
            csv_path, tmp_path / "out.jsonl",
            checkpoint_path=tmp_path / "cp.json",
            workers=1, _download_fn=_fake_ok,
        )
        assert {"flag", "n_attempted", "n_succeeded", "n_failed", "n_skipped"} <= result.keys()

    def test_concurrent_workers_produce_correct_count(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "toi.csv"
        _write_toi_csv(csv_path, [_good_row(i) for i in range(1, 9)])
        out = tmp_path / "out.jsonl"
        result = download_tess_lightcurves(
            csv_path, out,
            checkpoint_path=tmp_path / "cp.json",
            workers=4, _download_fn=_fake_ok,
        )
        assert result["n_succeeded"] == 8
        assert len(out.read_text().splitlines()) == 8

    def test_checkpoint_updated_after_each_record(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "toi.csv"
        _write_toi_csv(csv_path, [_good_row(42)])
        cp = tmp_path / "cp.json"
        download_tess_lightcurves(
            csv_path, tmp_path / "out.jsonl",
            checkpoint_path=cp, workers=1, _download_fn=_fake_ok,
        )
        state = _load_checkpoint(cp)
        assert "42" in state["completed"]

    def test_keyboard_interrupt_returns_interrupted_flag(
        self, tmp_path: Path
    ) -> None:
        """KeyboardInterrupt mid-run must return flag='INTERRUPTED' with checkpoint saved."""
        csv_path = tmp_path / "toi.csv"
        _write_toi_csv(csv_path, [_good_row(i) for i in range(1, 6)])
        out = tmp_path / "out.jsonl"
        cp = tmp_path / "cp.json"

        call_count = 0

        def _interrupt_on_second(row: dict) -> dict | None:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise KeyboardInterrupt
            return _fake_ok(row)

        result = download_tess_lightcurves(
            csv_path, out,
            checkpoint_path=cp,
            workers=1,
            _download_fn=_interrupt_on_second,
        )
        assert result["flag"] == "INTERRUPTED"
        assert {"flag", "n_attempted", "n_succeeded", "n_failed", "n_skipped"} <= result.keys()
        # Checkpoint must be saved so the run is resumable
        state = _load_checkpoint(cp)
        assert len(state["completed"]) >= 1
