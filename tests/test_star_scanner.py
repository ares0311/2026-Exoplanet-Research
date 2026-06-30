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
        "provenance_score": 0.82,
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

    def test_new_log_is_created_immediately(self, tmp_path: Path) -> None:
        path = tmp_path / "log.json"
        ScanLog(path)
        assert path.exists()
        with path.open() as fh:
            data = json.load(fh)
        assert data["entries"] == {}
        assert data["active"] == {}

    def test_is_scanned_false_before_record(self, tmp_path: Path) -> None:
        log = ScanLog(tmp_path / "log.json")
        assert not log.is_scanned(12345)

    def test_is_scanned_true_after_record(self, tmp_path: Path) -> None:
        log = ScanLog(tmp_path / "log.json")
        log.record(12345, "scanned_clear", {"n_signals": 0})
        assert log.is_scanned(12345)

    def test_mark_started_does_not_make_target_scanned(self, tmp_path: Path) -> None:
        log = ScanLog(tmp_path / "log.json")
        target = {"tic_id": 12345, "priority": 0.75}

        log.mark_started(12345, target, pipeline="QLP", exptime="long")

        assert not log.is_scanned(12345)
        assert log.scanned_ids() == set()
        assert log.summary()["active"] == 1

    def test_record_clears_active_target(self, tmp_path: Path) -> None:
        log = ScanLog(tmp_path / "log.json")
        target = {"tic_id": 12345, "priority": 0.75}

        log.mark_started(12345, target, pipeline="QLP", exptime="long")
        log.record(12345, "scanned_clear", {"n_signals": 0})

        assert log.is_scanned(12345)
        assert log.summary()["active"] == 0

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
            "best_snr": 12.5,
            "best_detection_confidence": 0.8,
            "best_novelty_score": 0.7,
            "best_depth_ppm": 900.0,
            "best_duration_hours": 2.5,
            "best_transit_count": 3,
            "provenance_score": 0.6,
            "signals": [{"candidate_id": "TIC_555_s01"}],
            "priority_score": 0.9,
        })
        with (tmp_path / "log.json").open() as fh:
            data = json.load(fh)
        entry = data["entries"]["555"]
        assert entry["n_signals"] == 2
        assert entry["best_period_days"] == pytest.approx(37.4)
        assert entry["best_pathway"] == "tfop_ready"
        assert entry["best_snr"] == pytest.approx(12.5)
        assert entry["best_detection_confidence"] == pytest.approx(0.8)
        assert entry["best_novelty_score"] == pytest.approx(0.7)
        assert entry["best_depth_ppm"] == pytest.approx(900.0)
        assert entry["best_duration_hours"] == pytest.approx(2.5)
        assert entry["best_transit_count"] == 3
        assert entry["provenance_score"] == pytest.approx(0.6)
        assert entry["signals"] == [{"candidate_id": "TIC_555_s01"}]


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

    @patch("astroquery.mast.Catalogs")
    def test_returns_list_of_dicts(self, mock_catalogs: MagicMock) -> None:
        mock_catalogs.query_region.return_value = self._catalog_rows()
        results = select_targets(n=10)
        assert isinstance(results, list)
        assert all("tic_id" in r and "priority" in r for r in results)

    @patch("astroquery.mast.Catalogs")
    def test_sorted_by_priority_descending(self, mock_catalogs: MagicMock) -> None:
        mock_catalogs.query_region.return_value = self._catalog_rows()
        results = select_targets(n=10)
        priorities = [r["priority"] for r in results]
        assert priorities == sorted(priorities, reverse=True)

    @patch("astroquery.mast.Catalogs")
    def test_excludes_specified_ids(self, mock_catalogs: MagicMock) -> None:
        mock_catalogs.query_region.return_value = self._catalog_rows()
        results = select_targets(n=10, exclude_tic_ids={100, 200})
        tic_ids = {r["tic_id"] for r in results}
        assert 100 not in tic_ids
        assert 200 not in tic_ids

    @patch("astroquery.mast.Catalogs")
    def test_respects_n_limit(self, mock_catalogs: MagicMock) -> None:
        mock_catalogs.query_region.return_value = self._catalog_rows()
        results = select_targets(n=2)
        assert len(results) <= 2

    @patch("astroquery.mast.Catalogs")
    def test_handles_missing_teff_gracefully(self, mock_catalogs: MagicMock) -> None:
        rows = [{"ID": 500, "Tmag": 12.0, "Teff": None, "contratio": None}]
        mock_catalogs.query_region.return_value = rows
        results = select_targets(n=5)
        assert len(results) == 1
        assert results[0]["teff"] is None

    @patch("astroquery.mast.Catalogs")
    def test_uses_bounded_region_queries_not_all_sky_criteria(
        self,
        mock_catalogs: MagicMock,
    ) -> None:
        mock_catalogs.query_region.return_value = self._catalog_rows()
        select_targets(n=2)
        assert mock_catalogs.query_region.called
        assert not mock_catalogs.query_criteria.called

    @patch("astroquery.mast.Catalogs")
    def test_raises_when_all_tic_tiles_fail(self, mock_catalogs: MagicMock) -> None:
        mock_catalogs.query_region.side_effect = RuntimeError("remote closed")
        with pytest.raises(RuntimeError, match="TIC target selection failed"):
            select_targets(n=2, max_tiles=1, retry_attempts=1)


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
        assert result["best_snr"] == pytest.approx(15.0)
        assert result["best_detection_confidence"] == pytest.approx(0.9)
        assert result["best_novelty_score"] == pytest.approx(0.95)
        assert result["best_depth_ppm"] == pytest.approx(1000.0)
        assert result["best_duration_hours"] == pytest.approx(2.0)
        assert result["best_transit_count"] == 4
        assert result["provenance_score"] == pytest.approx(0.82)
        assert result["signals"][0]["snr"] == pytest.approx(15.0)

    def test_scanned_clear_when_no_signals(self, tmp_path: Path) -> None:
        with patch("Skills.star_scanner.run_pipeline") as mock_pipe:
            mock_pipe.return_value = []
            result = scan_star(99999)
        assert result["status"] == "scanned_clear"
        assert result["n_signals"] == 0

    def test_error_status_on_pipeline_exception(self) -> None:
        with patch("Skills.star_scanner.run_pipeline") as mock_pipe:
            mock_pipe.side_effect = RuntimeError("unexpected parser failure")
            result = scan_star(88888)
        assert result["status"] == "error"
        assert "unexpected parser failure" in result["error_message"]

    def test_no_lightcurve_exception_is_no_data(self) -> None:
        with patch("Skills.star_scanner.run_pipeline") as mock_pipe:
            mock_pipe.side_effect = RuntimeError(
                "No TESS light curves found for 'TIC 425884922'"
            )
            result = scan_star(425884922)
        assert result["status"] == "no_data"
        assert "No TESS light curves found" in result["error_message"]

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

    def test_no_data_is_logged_when_log_provided(self, tmp_path: Path) -> None:
        log = ScanLog(tmp_path / "log.json")
        with patch("Skills.star_scanner.run_pipeline") as mock_pipe:
            mock_pipe.side_effect = ValueError("No TESS light curves found")
            scan_star(556, log=log)
        assert log.is_scanned(556)
        assert log.summary()["no_data"] == 1


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
            patch("Skills.star_scanner._load_ctoi_tic_ids", return_value=set()),
            patch("Skills.star_scanner._load_confirmed_host_tic_ids", return_value=frozenset()),
            patch("Skills.star_scanner.select_targets", return_value=self._targets()),
            patch("Skills.star_scanner.run_pipeline", return_value=[]),
        ):
            run_background_scan(log_path, n_targets=10)
        log = ScanLog(log_path)
        assert log.is_scanned(1001)
        assert log.is_scanned(1002)

    def test_background_scan_prints_workers_elapsed_and_eta(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        log_path = tmp_path / "log.json"
        with (
            patch("Skills.star_scanner._load_toi_tic_ids", return_value=set()),
            patch("Skills.star_scanner._load_ctoi_tic_ids", return_value=set()),
            patch("Skills.star_scanner._load_confirmed_host_tic_ids", return_value=frozenset()),
            patch("Skills.star_scanner.select_targets", return_value=self._targets()),
            patch("Skills.star_scanner.run_pipeline", return_value=[]),
        ):
            run_background_scan(log_path, n_targets=10, workers=2, request_delay=0.0)

        out = capsys.readouterr().out
        assert "workers=2" in out
        assert "pipeline=QLP" in out
        assert "exptime=long" in out
        assert "request_delay=0.00s" in out
        assert "[start] TIC" in out
        assert "active=" in out
        assert "elapsed=" in out
        assert "ETA=" in out

    def test_background_scan_passes_fetch_options_to_pipeline(
        self,
        tmp_path: Path,
    ) -> None:
        log_path = tmp_path / "log.json"
        captured: list[dict[str, Any]] = []

        def fake_pipeline(*args: object, **kwargs: object) -> list[object]:
            captured.append(dict(kwargs))
            return []

        with (
            patch("Skills.star_scanner._load_toi_tic_ids", return_value=set()),
            patch("Skills.star_scanner._load_ctoi_tic_ids", return_value=set()),
            patch("Skills.star_scanner._load_confirmed_host_tic_ids", return_value=frozenset()),
            patch("Skills.star_scanner.select_targets", return_value=self._targets()),
            patch("Skills.star_scanner.run_pipeline", side_effect=fake_pipeline),
        ):
            run_background_scan(
                log_path,
                n_targets=10,
                pipeline="QLP",
                exptime="long",
                workers=2,
                request_delay=0.0,
            )

        assert captured
        assert all(row["pipeline"] == "QLP" for row in captured)
        assert all(row["exptime"] == "long" for row in captured)
        for entry in ScanLog(log_path)._data["entries"].values():
            assert entry["pipeline"] == "QLP"
            assert entry["exptime"] == "long"

    def test_skips_already_scanned_via_exclude(self, tmp_path: Path) -> None:
        log_path = tmp_path / "log.json"
        log = ScanLog(log_path)
        log.record(1001, "scanned_clear", {})

        captured_exclude: list[set[int]] = []

        def fake_select(
            n: int,
            tmag_range: tuple,
            exclude_tic_ids: set | None = None,
            **_: object,
        ) -> list:
            captured_exclude.append(exclude_tic_ids or set())
            return [self._targets()[1]]  # only 1002

        with (
            patch("Skills.star_scanner._load_toi_tic_ids", return_value=set()),
            patch("Skills.star_scanner._load_ctoi_tic_ids", return_value=set()),
            patch("Skills.star_scanner._load_confirmed_host_tic_ids", return_value=frozenset()),
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
            patch("Skills.star_scanner._load_ctoi_tic_ids", return_value=set()),
            patch("Skills.star_scanner._load_confirmed_host_tic_ids", return_value=frozenset()),
            patch("Skills.star_scanner.select_targets", return_value=self._targets()),
            patch("Skills.star_scanner.run_pipeline", side_effect=fake_pipeline),
        ):
            run_background_scan(log_path, n_targets=10)  # must not raise

    def test_toi_ids_excluded(self, tmp_path: Path) -> None:
        log_path = tmp_path / "log.json"
        captured: list[set[int]] = []

        def fake_select(
            n: int,
            tmag_range: tuple,
            exclude_tic_ids: set | None = None,
            **_: object,
        ) -> list:
            captured.append(exclude_tic_ids or set())
            return []

        with (
            patch("Skills.star_scanner._load_toi_tic_ids", return_value={1001, 9999}),
            patch("Skills.star_scanner._load_ctoi_tic_ids", return_value=set()),
            patch("Skills.star_scanner._load_confirmed_host_tic_ids", return_value=frozenset()),
            patch("Skills.star_scanner.select_targets", side_effect=fake_select),
        ):
            run_background_scan(log_path, n_targets=10)

        assert 1001 in captured[0]
        assert 9999 in captured[0]
