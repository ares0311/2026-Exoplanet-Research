"""Tests for Skills/star_scanner.py (offline / unit tests only)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# ---------------------------------------------------------------------------
# Inject a stub astroquery into sys.modules so tests run without the package
# ---------------------------------------------------------------------------

_mock_astroquery_mast = MagicMock()
_mock_astroquery = MagicMock()
_mock_astroquery.mast = _mock_astroquery_mast
sys.modules.setdefault("astroquery", _mock_astroquery)
sys.modules.setdefault("astroquery.mast", _mock_astroquery_mast)

from Skills.star_scanner import (  # noqa: E402
    ScanLog,
    priority_score,
    run_background_scan,
    scan_star,
    select_targets,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pipeline_row(
    period: float = 10.0,
    fpp: float = 0.10,
    pathway: str = "tfop_ready",
) -> dict[str, Any]:
    return {
        "candidate_id": "test-001",
        "target_id": "TIC 123",
        "mission": "TESS",
        "period_days": period,
        "epoch_bjd": 2458000.0,
        "duration_hours": 2.0,
        "depth_ppm": 1000,
        "transit_count": 4,
        "snr": 15.0,
        "scorer": "bayesian",
        "posterior": {"planet_candidate": 1.0 - fpp, "eclipsing_binary": 0.0,
                      "background_eclipsing_binary": 0.0, "stellar_variability": 0.0,
                      "instrumental_artifact": 0.0, "known_object": 0.0},
        "scores": {"false_positive_probability": fpp,
                   "detection_confidence": 0.9, "novelty_score": 0.95},
        "pathway": pathway,
    }


def _mock_tic_row(tic_id: int, tmag: float = 12.0,
                  teff: float | None = 4500.0,
                  contratio: float | None = 0.01) -> dict[str, Any]:
    return {"ID": tic_id, "Tmag": tmag, "Teff": teff, "contratio": contratio}


# ---------------------------------------------------------------------------
# TestPriorityScore
# ---------------------------------------------------------------------------


class TestPriorityScore:
    def test_output_in_unit_interval(self) -> None:
        score = priority_score(12.0, teff=4500.0, n_sectors=6, contratio=0.0)
        assert 0.0 <= score <= 1.0

    def test_ideal_star_scores_high(self) -> None:
        score = priority_score(12.5, teff=4000.0, n_sectors=6, contratio=0.0)
        assert score >= 0.85

    def test_very_bright_penalised(self) -> None:
        score_bright = priority_score(7.0)
        score_good = priority_score(12.0)
        assert score_bright < score_good

    def test_very_faint_penalised(self) -> None:
        score_faint = priority_score(17.0)
        score_good = priority_score(12.0)
        assert score_faint < score_good

    def test_m_dwarf_preferred_over_hot_star(self) -> None:
        score_m = priority_score(12.0, teff=4000.0)
        score_hot = priority_score(12.0, teff=9000.0)
        assert score_m > score_hot

    def test_more_sectors_preferred(self) -> None:
        score_many = priority_score(12.0, n_sectors=6)
        score_few = priority_score(12.0, n_sectors=1)
        assert score_many > score_few

    def test_low_contamination_preferred(self) -> None:
        score_clean = priority_score(12.0, contratio=0.0)
        score_dirty = priority_score(12.0, contratio=0.9)
        assert score_clean > score_dirty

    def test_none_teff_gives_neutral(self) -> None:
        score_none = priority_score(12.0, teff=None)
        assert 0.0 < score_none < 1.0

    def test_none_sectors_neutral(self) -> None:
        score = priority_score(12.0, n_sectors=None)
        assert 0.0 < score < 1.0

    def test_none_contratio_neutral(self) -> None:
        score = priority_score(12.0, contratio=None)
        assert 0.0 < score < 1.0

    def test_sectors_capped_at_six(self) -> None:
        score_six = priority_score(12.0, n_sectors=6)
        score_ten = priority_score(12.0, n_sectors=10)
        assert score_six == score_ten

    def test_tmag_13_is_maximum_magnitude_score(self) -> None:
        score_13 = priority_score(13.0, teff=4500.0, n_sectors=6, contratio=0.0)
        score_12 = priority_score(12.0, teff=4500.0, n_sectors=6, contratio=0.0)
        # Both 12 and 13 are in the flat-top region; 13.0 should be ≥ 12.0
        assert score_13 >= score_12 * 0.95


# ---------------------------------------------------------------------------
# TestScanLog
# ---------------------------------------------------------------------------


class TestScanLog:
    def test_new_log_has_no_entries(self, tmp_path: Path) -> None:
        log = ScanLog(tmp_path / "log.json")
        assert log.summary()["total"] == 0

    def test_is_scanned_false_before_record(self, tmp_path: Path) -> None:
        log = ScanLog(tmp_path / "log.json")
        assert not log.is_scanned(12345)

    def test_is_scanned_true_after_record(self, tmp_path: Path) -> None:
        log = ScanLog(tmp_path / "log.json")
        log.record(12345, "scanned_clear", {"n_signals": 0})
        assert log.is_scanned(12345)

    def test_scanned_ids_returns_all_recorded(self, tmp_path: Path) -> None:
        log = ScanLog(tmp_path / "log.json")
        log.record(100, "scanned_clear", {})
        log.record(200, "candidate_found", {"n_signals": 1})
        assert log.scanned_ids() == {100, 200}

    def test_summary_counts_by_status(self, tmp_path: Path) -> None:
        log = ScanLog(tmp_path / "log.json")
        log.record(1, "candidate_found", {"n_signals": 1})
        log.record(2, "scanned_clear", {})
        log.record(3, "error", {"error_message": "oops"})
        s = log.summary()
        assert s["candidate_found"] == 1
        assert s["scanned_clear"] == 1
        assert s["error"] == 1
        assert s["total"] == 3

    def test_log_persisted_to_disk(self, tmp_path: Path) -> None:
        path = tmp_path / "log.json"
        log = ScanLog(path)
        log.record(999, "scanned_clear", {})
        assert path.exists()
        with path.open() as fh:
            data = json.load(fh)
        assert "999" in data["entries"]

    def test_log_loads_existing_file(self, tmp_path: Path) -> None:
        path = tmp_path / "log.json"
        log1 = ScanLog(path)
        log1.record(42, "scanned_clear", {})
        log2 = ScanLog(path)
        assert log2.is_scanned(42)

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        path = tmp_path / "subdir" / "nested" / "log.json"
        log = ScanLog(path)
        log.record(1, "scanned_clear", {})
        assert path.exists()

    def test_record_stores_all_fields(self, tmp_path: Path) -> None:
        log = ScanLog(tmp_path / "log.json")
        log.record(555, "candidate_found", {
            "n_signals": 2,
            "best_period_days": 37.4,
            "best_fpp": 0.08,
            "best_pathway": "tfop_ready",
            "priority_score": 0.9,
        })
        with (tmp_path / "log.json").open() as fh:
            data = json.load(fh)
        entry = data["entries"]["555"]
        assert entry["n_signals"] == 2
        assert entry["best_period_days"] == pytest.approx(37.4)
        assert entry["best_pathway"] == "tfop_ready"


# ---------------------------------------------------------------------------
# TestSelectTargets
# ---------------------------------------------------------------------------


def _make_mock_catalog_result(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return rows


class TestSelectTargets:
    def _catalog_rows(self) -> list[dict[str, Any]]:
        return [
            {"ID": 100, "Tmag": 12.0, "Teff": 4500.0, "contratio": 0.01},
            {"ID": 200, "Tmag": 11.0, "Teff": 5800.0, "contratio": 0.05},
            {"ID": 300, "Tmag": 13.5, "Teff": 3800.0, "contratio": 0.0},
        ]

    def test_returns_list_of_dicts(self) -> None:
        _mock_astroquery_mast.Catalogs.query_criteria.return_value = self._catalog_rows()
        results = select_targets(n=10)
        assert isinstance(results, list)
        assert all("tic_id" in r and "priority" in r for r in results)

    def test_sorted_by_priority_descending(self) -> None:
        _mock_astroquery_mast.Catalogs.query_criteria.return_value = self._catalog_rows()
        results = select_targets(n=10)
        priorities = [r["priority"] for r in results]
        assert priorities == sorted(priorities, reverse=True)

    def test_excludes_specified_ids(self) -> None:
        _mock_astroquery_mast.Catalogs.query_criteria.return_value = self._catalog_rows()
        results = select_targets(n=10, exclude_tic_ids={100, 200})
        tic_ids = {r["tic_id"] for r in results}
        assert 100 not in tic_ids
        assert 200 not in tic_ids

    def test_respects_n_limit(self) -> None:
        _mock_astroquery_mast.Catalogs.query_criteria.return_value = self._catalog_rows()
        results = select_targets(n=2)
        assert len(results) <= 2

    def test_handles_missing_teff_gracefully(self) -> None:
        rows = [{"ID": 500, "Tmag": 12.0, "Teff": None, "contratio": None}]
        _mock_astroquery_mast.Catalogs.query_criteria.return_value = rows
        results = select_targets(n=5)
        assert len(results) == 1
        assert results[0]["teff"] is None


# ---------------------------------------------------------------------------
# TestScanStar
# ---------------------------------------------------------------------------


class TestScanStar:
    def test_candidate_found_when_signals_returned(self, tmp_path: Path) -> None:
        with patch("Skills.star_scanner.run_pipeline") as mock_pipe:
            mock_pipe.return_value = [_make_pipeline_row(fpp=0.10)]
            result = scan_star(150428135)
        assert result["status"] == "candidate_found"
        assert result["n_signals"] == 1
        assert result["best_fpp"] == pytest.approx(0.10)

    def test_scanned_clear_when_no_signals(self, tmp_path: Path) -> None:
        with patch("Skills.star_scanner.run_pipeline") as mock_pipe:
            mock_pipe.return_value = []
            result = scan_star(99999)
        assert result["status"] == "scanned_clear"
        assert result["n_signals"] == 0

    def test_error_status_on_pipeline_exception(self) -> None:
        with patch("Skills.star_scanner.run_pipeline") as mock_pipe:
            mock_pipe.side_effect = RuntimeError("no data found")
            result = scan_star(88888)
        assert result["status"] == "error"
        assert "no data found" in result["error_message"]

    def test_records_to_log_when_provided(self, tmp_path: Path) -> None:
        log = ScanLog(tmp_path / "log.json")
        with patch("Skills.star_scanner.run_pipeline") as mock_pipe:
            mock_pipe.return_value = [_make_pipeline_row()]
            scan_star(111, log=log)
        assert log.is_scanned(111)

    def test_does_not_require_log(self) -> None:
        with patch("Skills.star_scanner.run_pipeline") as mock_pipe:
            mock_pipe.return_value = []
            result = scan_star(222, log=None)
        assert result["status"] == "scanned_clear"

    def test_selects_best_by_lowest_fpp(self) -> None:
        rows = [
            _make_pipeline_row(period=5.0, fpp=0.40),
            _make_pipeline_row(period=10.0, fpp=0.05),
        ]
        with patch("Skills.star_scanner.run_pipeline") as mock_pipe:
            mock_pipe.return_value = rows
            result = scan_star(333)
        assert result["best_period_days"] == pytest.approx(10.0)
        assert result["best_fpp"] == pytest.approx(0.05)

    def test_priority_stored_in_result(self) -> None:
        with patch("Skills.star_scanner.run_pipeline") as mock_pipe:
            mock_pipe.return_value = []
            result = scan_star(444, priority=0.77)
        assert result["priority_score"] == pytest.approx(0.77)

    def test_error_is_logged_when_log_provided(self, tmp_path: Path) -> None:
        log = ScanLog(tmp_path / "log.json")
        with patch("Skills.star_scanner.run_pipeline") as mock_pipe:
            mock_pipe.side_effect = ValueError("timeout")
            scan_star(555, log=log)
        assert log.is_scanned(555)
        assert log.summary()["error"] == 1


# ---------------------------------------------------------------------------
# TestRunBackgroundScan
# ---------------------------------------------------------------------------


class TestRunBackgroundScan:
    def _targets(self) -> list[dict[str, Any]]:
        return [
            {"tic_id": 1001, "tmag": 12.0, "teff": 4500.0,
             "contratio": 0.01, "priority": 0.90},
            {"tic_id": 1002, "tmag": 11.5, "teff": 5000.0,
             "contratio": 0.02, "priority": 0.85},
        ]

    def test_scans_all_selected_targets(self, tmp_path: Path) -> None:
        log_path = tmp_path / "log.json"
        with (
            patch("Skills.star_scanner._load_toi_tic_ids", return_value=set()),
            patch("Skills.star_scanner.select_targets", return_value=self._targets()),
            patch("Skills.star_scanner.run_pipeline", return_value=[]),
        ):
            run_background_scan(log_path, n_targets=10)
        log = ScanLog(log_path)
        assert log.is_scanned(1001)
        assert log.is_scanned(1002)

    def test_skips_already_scanned_via_exclude(self, tmp_path: Path) -> None:
        log_path = tmp_path / "log.json"
        log = ScanLog(log_path)
        log.record(1001, "scanned_clear", {})

        captured_exclude: list[set[int]] = []

        def fake_select(n: int, tmag_range: tuple, exclude_tic_ids: set | None = None) -> list:
            captured_exclude.append(exclude_tic_ids or set())
            return [self._targets()[1]]  # only 1002

        with (
            patch("Skills.star_scanner._load_toi_tic_ids", return_value=set()),
            patch("Skills.star_scanner.select_targets", side_effect=fake_select),
            patch("Skills.star_scanner.run_pipeline", return_value=[]),
        ):
            run_background_scan(log_path, n_targets=10)

        assert 1001 in captured_exclude[0]

    def test_keyboard_interrupt_exits_cleanly(self, tmp_path: Path) -> None:
        log_path = tmp_path / "log.json"
        call_count = 0

        def fake_pipeline(*args: object, **kwargs: object) -> list:
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                raise KeyboardInterrupt
            return []

        with (
            patch("Skills.star_scanner._load_toi_tic_ids", return_value=set()),
            patch("Skills.star_scanner.select_targets", return_value=self._targets()),
            patch("Skills.star_scanner.run_pipeline", side_effect=fake_pipeline),
        ):
            run_background_scan(log_path, n_targets=10)  # must not raise

    def test_toi_ids_excluded(self, tmp_path: Path) -> None:
        log_path = tmp_path / "log.json"
        captured: list[set[int]] = []

        def fake_select(n: int, tmag_range: tuple, exclude_tic_ids: set | None = None) -> list:
            captured.append(exclude_tic_ids or set())
            return []

        with (
            patch("Skills.star_scanner._load_toi_tic_ids", return_value={1001, 9999}),
            patch("Skills.star_scanner.select_targets", side_effect=fake_select),
        ):
            run_background_scan(log_path, n_targets=10)

        assert 1001 in captured[0]
        assert 9999 in captured[0]
