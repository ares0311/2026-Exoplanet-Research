"""Tests for Skills/star_scanner.py (offline / unit tests only)."""
from __future__ import annotations

import csv
import json
import sys
import time
from datetime import UTC, datetime
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

from Skills.candidate_database import CandidateDatabase  # noqa: E402
from Skills.star_scanner import (  # noqa: E402
    _DEFAULT_SEARCH_CENTERS,
    _EXPANSION_SEARCH_CENTERS,
    TOTAL_SEARCH_TILES,
    PreparedLiveSearchBundle,
    ScanLog,
    _bridge_manual_scan_to_hunter,
    _ledger_record_for_outcome,
    _load_asassn_variable_tic_ids,
    _load_prior_discovery_tic_ids,
    _load_toi_tic_ids,
    _manual_scan_history_entry,
    _search_centers,
    _write_run_report,
    inspect_target_products,
    load_prepared_live_search_bundle,
    main,
    prepare_live_search_snapshot,
    priority_score,
    run_background_scan,
    run_prepared_live_search,
    run_target_scan,
    scan_star,
    select_targets,
)

from exo_toolkit.dataset_manifest import (  # noqa: E402
    load_dataset_manifest,
    sha256_file,
    validate_dataset_manifest,
)
from exo_toolkit.search_lifecycle import HunterStore  # noqa: E402

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

    def test_small_star_preferred_over_large_star(self) -> None:
        # Same-size planet transits far deeper (and more detectably) around
        # a small star than a large one.
        score_small = priority_score(12.0, radius_rsun=0.4)
        score_large = priority_score(12.0, radius_rsun=3.0)
        assert score_small > score_large

    def test_none_radius_gives_neutral(self) -> None:
        score = priority_score(12.0, radius_rsun=None)
        assert 0.0 < score < 1.0

    def test_radius_capped_at_point_seven_rsun(self) -> None:
        score_small = priority_score(12.0, radius_rsun=0.7)
        score_smaller = priority_score(12.0, radius_rsun=0.2)
        assert score_small == score_smaller

    def test_ideal_star_with_radius_still_scores_high(self) -> None:
        score = priority_score(
            12.5, teff=4000.0, n_sectors=6, contratio=0.0, radius_rsun=0.5
        )
        assert score >= 0.85

    def test_ideal_star_without_radius_info_still_scores_high(self) -> None:
        # Existing behavior: an otherwise-ideal star with no radius data
        # available must not be penalised below the historical threshold.
        score = priority_score(12.5, teff=4000.0, n_sectors=6, contratio=0.0)
        assert score >= 0.85


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
            {"ID": 100, "ra": 10.0, "dec": -5.0, "Tmag": 12.0, "Teff": 4500.0, "contratio": 0.01},
            {"ID": 200, "ra": 20.0, "dec": 5.0, "Tmag": 11.0, "Teff": 5800.0, "contratio": 0.05},
            {"ID": 300, "ra": 30.0, "dec": 15.0, "Tmag": 13.5, "Teff": 3800.0, "contratio": 0.0},
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

    @patch("astroquery.mast.Catalogs")
    def test_default_stops_before_querying_every_tile(
        self, mock_catalogs: MagicMock
    ) -> None:
        # Default (full_sweep=False) behavior: with a small target pool and
        # every tile returning fresh unique candidates, the search should
        # stop long before exhausting the full configured grid.
        counter = {"n": 0}

        def _rows(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
            counter["n"] += 1
            base = counter["n"] * 1000
            return [{"ID": base + i, "Tmag": 12.0, "Teff": 4500.0, "contratio": 0.0}
                    for i in range(5)]

        mock_catalogs.query_region.side_effect = _rows
        select_targets(n=2, max_tiles=126)
        assert mock_catalogs.query_region.call_count < 126

    @patch("astroquery.mast.Catalogs")
    def test_full_sweep_queries_every_configured_tile(
        self, mock_catalogs: MagicMock
    ) -> None:
        mock_catalogs.query_region.return_value = self._catalog_rows()
        select_targets(n=2, max_tiles=10, full_sweep=True)
        assert mock_catalogs.query_region.call_count == 10

    @patch("astroquery.mast.Catalogs")
    def test_search_log_populated_on_default_path(
        self, mock_catalogs: MagicMock
    ) -> None:
        mock_catalogs.query_region.return_value = self._catalog_rows()
        log: dict[str, Any] = {}
        select_targets(n=2, max_tiles=5, search_log=log)
        assert log["tiles_configured"] == 5
        assert log["tiles_queried"] >= 1
        assert log["full_sweep"] is False
        assert "raw_candidates_before_exclusion" in log
        assert "candidates_after_exclusion" in log
        assert "sky_coverage_deg2" in log
        assert log["elapsed_seconds"] >= 0.0

    @patch("astroquery.mast.Catalogs")
    def test_search_log_reflects_full_sweep(self, mock_catalogs: MagicMock) -> None:
        mock_catalogs.query_region.return_value = self._catalog_rows()
        log: dict[str, Any] = {}
        select_targets(n=2, max_tiles=10, full_sweep=True, search_log=log)
        assert log["tiles_queried"] == 10
        assert log["full_sweep"] is True

    @patch("astroquery.mast.Catalogs")
    def test_search_log_counts_excluded_candidates(
        self, mock_catalogs: MagicMock
    ) -> None:
        mock_catalogs.query_region.return_value = self._catalog_rows()
        log: dict[str, Any] = {}
        select_targets(n=10, max_tiles=3, exclude_tic_ids={100}, search_log=log)
        assert log["excluded_count"] == 1
        assert 100 not in {t["tic_id"] for t in select_targets(
            n=10, max_tiles=3, exclude_tic_ids={100}
        )}

    @patch("astroquery.mast.Catalogs")
    def test_radius_extracted_from_catalog_row(self, mock_catalogs: MagicMock) -> None:
        rows = [{"ID": 700, "Tmag": 12.0, "Teff": 4500.0, "contratio": 0.0, "rad": 0.45}]
        mock_catalogs.query_region.return_value = rows
        results = select_targets(n=5)
        assert results[0]["radius_rsun"] == pytest.approx(0.45)

    @patch("astroquery.mast.Catalogs")
    def test_missing_radius_is_none(self, mock_catalogs: MagicMock) -> None:
        rows = [{"ID": 800, "Tmag": 12.0, "Teff": 4500.0, "contratio": 0.0}]
        mock_catalogs.query_region.return_value = rows
        results = select_targets(n=5)
        assert results[0]["radius_rsun"] is None

    @patch("astroquery.mast.conf")
    @patch("astroquery.mast.Catalogs")
    def test_query_timeout_bounds_mast_conf(
        self, mock_catalogs: MagicMock, mock_conf: MagicMock
    ) -> None:
        # astroquery.mast's own default timeout is 600s and is never
        # otherwise overridden in this project; a single slow tile query
        # could stall an entire full_sweep for up to ten minutes. Every
        # tile query must bound it explicitly (as an int -- astropy's
        # ConfigItem for this setting rejects float values).
        mock_catalogs.query_region.return_value = self._catalog_rows()
        select_targets(n=2, max_tiles=1, query_timeout_seconds=15.0)
        assert mock_conf.timeout == 15
        assert isinstance(mock_conf.timeout, int)

    @patch("astroquery.mast.conf")
    @patch("astroquery.mast.Catalogs")
    def test_query_timeout_default_applied_without_override(
        self, mock_catalogs: MagicMock, mock_conf: MagicMock
    ) -> None:
        mock_catalogs.query_region.return_value = self._catalog_rows()
        select_targets(n=2, max_tiles=1)
        assert mock_conf.timeout == 30

    @patch("astroquery.mast.Catalogs")
    def test_max_tiles_within_base_grid_never_uses_expansion_ring(
        self, mock_catalogs: MagicMock
    ) -> None:
        mock_catalogs.query_region.return_value = self._catalog_rows()
        log: dict[str, Any] = {}
        select_targets(n=2, max_tiles=126, full_sweep=True, search_log=log)
        assert log["tiles_configured"] == 126

    @patch("astroquery.mast.Catalogs")
    def test_max_tiles_beyond_base_grid_draws_from_expansion_ring(
        self, mock_catalogs: MagicMock
    ) -> None:
        mock_catalogs.query_region.return_value = self._catalog_rows()
        log: dict[str, Any] = {}
        select_targets(n=2, max_tiles=200, full_sweep=True, search_log=log)
        assert log["tiles_configured"] == 200
        assert log["tiles_queried"] == 200

    @patch("astroquery.mast.Catalogs")
    def test_max_tiles_caps_at_total_available(self, mock_catalogs: MagicMock) -> None:
        mock_catalogs.query_region.return_value = self._catalog_rows()
        log: dict[str, Any] = {}
        select_targets(n=2, max_tiles=10_000, full_sweep=True, search_log=log)
        assert log["tiles_configured"] == TOTAL_SEARCH_TILES


class TestSearchCenters:
    def test_base_and_expansion_rings_are_disjoint(self) -> None:
        assert not set(_DEFAULT_SEARCH_CENTERS) & set(_EXPANSION_SEARCH_CENTERS)

    def test_total_search_tiles_matches_combined_ring_lengths(self) -> None:
        assert len(_DEFAULT_SEARCH_CENTERS) + len(
            _EXPANSION_SEARCH_CENTERS
        ) == TOTAL_SEARCH_TILES

    def test_search_centers_within_base_grid_unchanged(self) -> None:
        assert _search_centers(126) == _DEFAULT_SEARCH_CENTERS

    def test_search_centers_beyond_base_grid_appends_expansion_ring(self) -> None:
        combined = _search_centers(TOTAL_SEARCH_TILES)
        assert combined[:126] == _DEFAULT_SEARCH_CENTERS
        assert combined[126:] == _EXPANSION_SEARCH_CENTERS

    def test_search_centers_caps_at_total_available(self) -> None:
        assert _search_centers(TOTAL_SEARCH_TILES + 500) == (
            _DEFAULT_SEARCH_CENTERS + _EXPANSION_SEARCH_CENTERS
        )


# ---------------------------------------------------------------------------
# TestAsassnVariableExclusion
# ---------------------------------------------------------------------------


class TestAsassnVariableExclusion:
    def test_empty_candidates_returns_empty_without_import(self) -> None:
        assert _load_asassn_variable_tic_ids([]) == frozenset()

    def test_fails_open_when_module_cannot_load(self) -> None:
        with patch(
            "importlib.util.spec_from_file_location", return_value=None
        ):
            assert _load_asassn_variable_tic_ids([123]) == frozenset()

    def test_strict_raises_when_module_cannot_load(self) -> None:
        with (
            patch("importlib.util.spec_from_file_location", return_value=None),
            pytest.raises(RuntimeError, match="Cannot load"),
        ):
            _load_asassn_variable_tic_ids([123], strict=True)

    def test_fails_open_on_network_error(self) -> None:
        # The real preflight module loads and its contract parses fine, but
        # the live TAP query itself must never actually touch the network
        # in a default test -- force it to fail immediately instead.
        with patch(
            "urllib.request.urlopen", side_effect=OSError("network disabled in tests")
        ):
            result = _load_asassn_variable_tic_ids([123456789])
        assert result == frozenset()


class _FakeProductTable(list[dict[str, Any]]):
    colnames = ["dataURI", "size", "sequence_number", "productFilename"]


class TestLiveSearchPreparation:
    @staticmethod
    def _target(tic_id: int, *, priority: float = 0.9) -> dict[str, Any]:
        return {
            "tic_id": tic_id,
            "ra_deg": 10.0 + tic_id / 1000,
            "dec_deg": -20.0,
            "tmag": 12.5,
            "teff": 4500.0,
            "contratio": 0.01,
            "priority": priority,
        }

    def test_inspect_target_products_preserves_exact_metadata(self) -> None:
        table = _FakeProductTable(
            [
                {
                    "dataURI": "mast:TESS/product-a.fits",
                    "size": 100,
                    "sequence_number": 1,
                    "productFilename": "product-a.fits",
                },
                {
                    "dataURI": "mast:TESS/product-a.fits",
                    "size": 100,
                    "sequence_number": 1,
                    "productFilename": "product-a.fits",
                },
                {
                    "dataURI": "mast:TESS/product-b.fits",
                    "size": 250,
                    "sequence_number": 2,
                    "productFilename": "product-b.fits",
                },
            ]
        )
        search_result = MagicMock()
        search_result.table = table
        search_fn = MagicMock(return_value=search_result)

        result = inspect_target_products(
            self._target(101),
            mission="TESS",
            pipeline="QLP",
            exptime="long",
            search_fn=search_fn,
        )

        search_fn.assert_called_once_with(
            "TIC 101", mission="TESS", author="QLP", exptime="long"
        )
        assert result["product_count"] == 2
        assert result["total_bytes"] == 350
        assert result["sectors"] == (1, 2)
        assert [product["uri"] for product in result["products"]] == [
            "mast:TESS/product-a.fits",
            "mast:TESS/product-b.fits",
        ]

    def test_prepare_snapshot_writes_valid_policy_bundle(self, tmp_path: Path) -> None:
        queue = tmp_path / "data_selection" / "target_priority_queue.csv"
        snapshot = tmp_path / "data_selection" / "queue.batch-v1.csv"
        batch = tmp_path / "data_selection" / "batch_manifests" / "batch-v1.json"
        dataset = tmp_path / "metadata" / "dataset_manifests" / "dataset-v1.json"

        def inspector(target: dict[str, Any], **_: object) -> dict[str, Any]:
            result = dict(target)
            result.update(
                {
                    "products": (
                        {
                            "uri": f"mast:TESS/{target['tic_id']}.fits",
                            "filename": f"{target['tic_id']}.fits",
                            "size_bytes": 1_000_000,
                            "sector": 1,
                        },
                    ),
                    "product_count": 1,
                    "total_bytes": 1_000_000,
                    "sectors": (1,),
                    "n_sectors": 1,
                    "priority": 0.9,
                }
            )
            return result

        result = prepare_live_search_snapshot(
            [self._target(102), self._target(101)],
            queue_path=queue,
            immutable_snapshot_path=snapshot,
            batch_manifest_path=batch,
            dataset_manifest_path=dataset,
            repo_root=tmp_path,
            workers=2,
            inspector_fn=inspector,
            batch_id="batch-v1",
            dataset_id="dataset-v1",
        )

        with queue.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
            header = list(rows[0])
        assert header == [
            "target_id", "project", "source", "catalog_ids", "ra_deg", "dec_deg",
            "data_products_available", "estimated_download_gb", "search_category",
            "scientific_novelty", "prior_significance", "followup_leverage",
            "data_quality", "method_advantage", "publication_value",
            "community_integration", "new_followup_balance", "storage_cost_penalty",
            "total_priority", "status", "notes", "citations",
        ]
        assert [row["target_id"] for row in rows] == ["TIC 101", "TIC 102"]
        assert queue.read_text() == snapshot.read_text()

        batch_payload = json.loads(batch.read_text())
        assert batch_payload["expected_raw_files"] == 2
        assert batch_payload["target_queue_snapshot"]["sha256"] == sha256_file(snapshot)
        assert [item["target_id"] for item in batch_payload["product_inventory"]] == [
            "TIC 101",
            "TIC 102",
        ]
        manifest = load_dataset_manifest(dataset)
        assert manifest.role == "live_search"
        assert manifest.local_path == "data_selection/queue.batch-v1.csv"
        assert manifest.row_count == manifest.group_count == 2
        assert validate_dataset_manifest(dataset, repo_root=tmp_path).ok
        assert result["target_count"] == 2

        snapshot.write_text(snapshot.read_text() + "\n")
        assert not validate_dataset_manifest(dataset, repo_root=tmp_path).ok

    @pytest.mark.parametrize("failure", ["empty", "no_products", "missing_size"])
    def test_prepare_snapshot_fails_closed(
        self, tmp_path: Path, failure: str
    ) -> None:
        queue = tmp_path / "data_selection" / "queue.csv"
        snapshot = tmp_path / "data_selection" / "snapshot.csv"
        batch = tmp_path / "data_selection" / "batch.json"
        dataset = tmp_path / "metadata" / "dataset.json"

        targets = [] if failure == "empty" else [self._target(101)]

        def inspector(target: dict[str, Any], **_: object) -> dict[str, Any]:
            product_count = 0 if failure == "no_products" else 1
            size = None if failure == "missing_size" else 100
            return {
                **target,
                "products": (
                    ()
                    if not product_count
                    else (
                        {
                            "uri": "mast:x",
                            "filename": "x",
                            "size_bytes": size,
                            "sector": 1,
                        },
                    )
                ),
                "product_count": product_count,
                "total_bytes": int(size or 0),
                "sectors": () if not product_count else (1,),
                "n_sectors": product_count,
                "priority": 0.9,
            }

        with pytest.raises(RuntimeError):
            prepare_live_search_snapshot(
                targets,
                queue_path=queue,
                immutable_snapshot_path=snapshot,
                batch_manifest_path=batch,
                dataset_manifest_path=dataset,
                repo_root=tmp_path,
                inspector_fn=inspector,
            )
        assert not dataset.exists()

    def test_prepare_only_routes_before_scanning(self, tmp_path: Path) -> None:
        log_path = tmp_path / "scan.json"
        target = self._target(101)
        with (
            patch("Skills.star_scanner._load_toi_tic_ids", return_value={1}),
            patch("Skills.star_scanner._load_ctoi_tic_ids", return_value={2}),
            patch(
                "Skills.star_scanner._load_confirmed_host_tic_ids",
                return_value=frozenset({3}),
            ),
            patch("Skills.star_scanner.select_targets", return_value=[target]),
            patch("Skills.star_scanner.prepare_live_search_snapshot") as prepare,
            patch("Skills.star_scanner.run_pipeline") as pipeline,
        ):
            exit_code = main(
                [
                    "--prepare-only",
                    "--max-stars", "1",
                    "--log", str(log_path),
                    "--queue-path", str(tmp_path / "queue.csv"),
                    "--queue-snapshot-path", str(tmp_path / "snapshot.csv"),
                    "--batch-manifest-path", str(tmp_path / "batch.json"),
                    "--dataset-manifest-path", str(tmp_path / "dataset.json"),
                    "--no-git-report",
                ]
            )

        assert exit_code == 0
        prepare.assert_called_once()
        pipeline.assert_not_called()
        assert not log_path.exists()

    def test_prior_discovery_logs_are_unioned(self, tmp_path: Path) -> None:
        (tmp_path / "discovery_run_001.json").write_text(
            json.dumps({"entries": {"101": {}, "102": {}}})
        )
        (tmp_path / "discovery_run_002_qlp.json").write_text(
            json.dumps({"entries": {"102": {}, "103": {}}})
        )

        assert _load_prior_discovery_tic_ids(tmp_path, strict=True) == {
            101,
            102,
            103,
        }

    def test_prepare_only_fails_before_selection_on_exclusion_error(
        self, tmp_path: Path
    ) -> None:
        with (
            patch("Skills.star_scanner._load_toi_tic_ids", return_value={1}),
            patch(
                "Skills.star_scanner._load_ctoi_tic_ids",
                side_effect=RuntimeError("CTOI unavailable"),
            ),
            patch("Skills.star_scanner.select_targets") as select,
            pytest.raises(RuntimeError, match="CTOI unavailable"),
        ):
            run_background_scan(
                tmp_path / "scan.json",
                prepare_only=True,
                repo_root=tmp_path,
            )
        select.assert_not_called()

    def test_toi_loader_strict_mode_rejects_empty_or_unknown_schema(self) -> None:
        with (
            patch("Skills.star_scanner._url_text", return_value="other\n1\n"),
            pytest.raises(RuntimeError, match="TIC ID column"),
        ):
            _load_toi_tic_ids(strict=True)

        with (
            patch("Skills.star_scanner._url_text", return_value="TIC ID\n"),
            pytest.raises(RuntimeError, match="no TIC IDs"),
        ):
            _load_toi_tic_ids(strict=True)

    def test_run_report_uses_injected_git_runner(self, tmp_path: Path) -> None:
        fake_runner = MagicMock()
        with patch(
            "Skills.star_scanner.run_and_commit_report", return_value=True
        ) as commit:
            _write_run_report(
                started_at="2026-07-11T00:00:00+00:00",
                elapsed_seconds=12.0,
                items_processed=20,
                items_written=18,
                items_failed=0,
                output_paths=("data_selection/target_priority_queue.csv",),
                report_dir=tmp_path,
                notes="metadata only",
                git_run_fn=fake_runner,
            )

        report, path = commit.call_args.args
        assert report.script == "star_scanner"
        assert report.items_processed == 20
        assert report.items_written == 18
        assert path == tmp_path / "star_scanner.jsonl"
        assert commit.call_args.kwargs["run_fn"] is fake_runner

    def test_committed_prepared_bundle_cross_validates(self) -> None:
        root = Path(__file__).resolve().parent.parent
        bundle = load_prepared_live_search_bundle(
            dataset_manifest_path=(
                root / "metadata/dataset_manifests/tess_live_search_v1.json"
            ),
            batch_manifest_path=(
                root / "data_selection/batch_manifests/tess_live_search_v1.json"
            ),
            repo_root=root,
        )

        assert bundle.dataset_id == "tess_live_search_v1"
        assert len(bundle.targets) == 18
        assert sum(len(target["raw_uris"]) for target in bundle.targets) == 103

    def test_ledger_record_requires_exact_frozen_uri_tuple(self) -> None:
        target = {
            "tic_id": 101,
            "target_id": "TIC 101",
            "priority": 20.0,
            "raw_uris": ("mast:a", "mast:b"),
        }
        context = {
            "fetch_provenance": {
                "raw_uris": ["mast:a", "mast:b"],
                "sectors_or_quarters": [1, 2],
            },
            "preprocess_version": "clean-v1",
        }
        record = _ledger_record_for_outcome(
            target=target,
            source_dataset_id="tess_live_search_v1",
            row=None,
            pipeline_context=context,
            min_snr=5.0,
            max_peaks=5,
            max_period_grid_points=20_000,
            scorer="bayesian",
            pipeline="QLP",
            exptime="long",
            model_path=None,
        )
        assert record.schema_version == 2
        assert record.raw_uris == ("mast:a", "mast:b")
        assert record.time_window == "sectors 1,2"
        assert record.candidate_id == "TIC_101_null"

        context["fetch_provenance"]["raw_uris"] = ["mast:a"]
        with pytest.raises(RuntimeError, match="differ from frozen inventory"):
            _ledger_record_for_outcome(
                target=target,
                source_dataset_id="tess_live_search_v1",
                row=None,
                pipeline_context=context,
                min_snr=5.0,
                max_peaks=5,
                max_period_grid_points=20_000,
                scorer="bayesian",
                pipeline="QLP",
                exptime="long",
                model_path=None,
            )

    def test_ledger_record_preserves_candidate_context(self) -> None:
        target = {
            "tic_id": 101,
            "target_id": "TIC 101",
            "priority": 20.0,
            "raw_uris": ("mast:a",),
        }
        row = _make_pipeline_row()
        row.update(
            {
                "fetch_provenance": {
                    "raw_uris": ["mast:a"],
                    "sectors_or_quarters": [1],
                },
                "score_quantile": 0.92,
                "calibration_dataset_id": "t1_2_k2pandc_calibration",
                "threshold_version": "no_decision_threshold_v1",
                "candidate_context_id": "context_v1",
                "false_discovery_estimate": 0.08,
                "false_discovery_reference_n": 100,
                "false_discovery_reference_negatives": 8,
            }
        )
        record = _ledger_record_for_outcome(
            target=target,
            source_dataset_id="tess_live_search_v1",
            row=row,
            pipeline_context={},
            min_snr=5.0,
            max_peaks=5,
            max_period_grid_points=20_000,
            scorer="full-ensemble",
            pipeline="QLP",
            exptime="long",
            model_path=None,
        )
        assert record.score_quantiles == {
            "full_ensemble_planet_probability": pytest.approx(0.92)
        }
        assert record.model_versions["calibration_dataset_id"] == (
            "t1_2_k2pandc_calibration"
        )
        assert record.model_versions["threshold_version"] == (
            "no_decision_threshold_v1"
        )
        assert record.model_scores["false_discovery_estimate"] == pytest.approx(0.08)

    def test_prepared_shard_writes_only_its_target_to_schema_v2(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        targets = tuple(
            {
                "tic_id": tic_id,
                "target_id": f"TIC {tic_id}",
                "priority": 20.0,
                "raw_uris": (f"mast:{tic_id}",),
            }
            for tic_id in (101, 102)
        )
        bundle = PreparedLiveSearchBundle(
            dataset_id="tess_live_search_v1",
            batch_id="tess_live_search_v1",
            targets=targets,
        )

        def fake_scan(tic_id: int, **_: object) -> dict[str, Any]:
            time.sleep(0.04)
            provenance = {
                "raw_uris": [f"mast:{tic_id}"],
                "sectors_or_quarters": [1],
            }
            return {
                "status": "candidate_found",
                "n_signals": 1,
                "best_period_days": 3.0,
                "best_fpp": 0.1,
                "best_pathway": "github_only_reproducibility",
                "error_message": None,
                "_pipeline_context": {
                    "fetch_provenance": provenance,
                    "preprocess_version": "clean-v1",
                },
                "_ledger_rows": [
                    {
                        "candidate_id": f"TIC_{tic_id}_s01",
                        "fetch_provenance": provenance,
                        "posterior": {"planet_candidate": 0.8},
                        "scores": {"false_positive_probability": 0.2},
                    }
                ],
            }

        with patch("Skills.star_scanner.scan_star", side_effect=fake_scan):
            summary = run_prepared_live_search(
                bundle,
                log_path=tmp_path / "scan.json",
                candidate_db_path=tmp_path / "candidates.sqlite3",
                workers=2,
                request_delay=0.0,
                shard_index=0,
                shard_count=2,
                heartbeat_interval_seconds=0.01,
            )

        db_path = tmp_path / "candidates.shard0of2.sqlite3"
        with CandidateDatabase(db_path) as database:
            assert database.provenanced_count() == 1
            assert database.provenanced_target_ids("tess_live_search_v1") == {
                "TIC 102"
            }
        assert summary["items_processed"] == 1
        assert summary["items_written"] == 1
        assert summary["items_failed"] == 0
        output = capsys.readouterr().out
        assert "batch_total=2" in output
        assert "[start] TIC 102" in output
        assert "[heartbeat]" in output
        assert "elapsed=" in output
        assert "ETA=" in output

        with patch("Skills.star_scanner.scan_star") as scan:
            resumed = run_prepared_live_search(
                bundle,
                log_path=tmp_path / "scan.json",
                candidate_db_path=tmp_path / "candidates.sqlite3",
                workers=2,
                request_delay=0.0,
                shard_index=0,
                shard_count=2,
            )
        scan.assert_not_called()
        assert resumed["items_processed"] == 0

    def test_execute_prepared_cli_never_uses_dynamic_selection(
        self, tmp_path: Path
    ) -> None:
        bundle = PreparedLiveSearchBundle(
            dataset_id="tess_live_search_v1",
            batch_id="tess_live_search_v1",
            targets=(),
        )
        summary = {
            "items_processed": 0,
            "items_written": 0,
            "items_failed": 0,
            "output_paths": (str(tmp_path / "scan.json"),),
        }
        with (
            patch(
                "Skills.star_scanner.load_prepared_live_search_bundle",
                return_value=bundle,
            ) as load,
            patch(
                "Skills.star_scanner.run_prepared_live_search",
                return_value=summary,
            ) as execute,
            patch("Skills.star_scanner.select_targets") as select,
            patch(
                "Skills.star_scanner._bridge_manual_scan_to_hunter"
            ) as bridge,
        ):
            code = main(
                [
                    "--execute-prepared-batch",
                    "--log",
                    str(tmp_path / "scan.json"),
                    "--candidate-db-path",
                    str(tmp_path / "candidates.sqlite3"),
                    "--shard-count",
                    "3",
                    "--shard-index",
                    "1",
                    "--no-git-report",
                ]
            )

        assert code == 0
        load.assert_called_once()
        execute.assert_called_once()
        assert execute.call_args.kwargs["heartbeat_interval_seconds"] == 30.0
        select.assert_not_called()
        # The prepared-batch shard-launcher path must stay untouched by the
        # manual-scan Hunter durable-history bridge.
        bridge.assert_not_called()


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


class TestRunTargetScan:
    def test_target_scan_prints_progress_and_clears_active(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        log_path = tmp_path / "target_log.json"
        with patch("Skills.star_scanner.run_pipeline") as mock_pipe:
            mock_pipe.return_value = [_make_pipeline_row()]
            result = run_target_scan(
                log_path,
                201252011,
                pipeline="QLP",
                exptime="long",
                max_period_grid_points=20_000,
            )

        out = capsys.readouterr().out
        assert result["status"] == "candidate_found"
        assert "[start] TIC 201252011" in out
        assert "[1/1] TIC 201252011" in out
        assert "pipeline=QLP" in out
        assert "exptime=long" in out
        assert "period_grid≤20000" in out
        assert "elapsed=" in out
        assert "ETA=" in out

        log = ScanLog(log_path)
        assert log.is_scanned(201252011)
        assert log.summary()["active"] == 0
        assert log.summary()["candidate_found"] == 1


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

    def test_scanned_this_run_lists_exactly_this_runs_targets(
        self, tmp_path: Path
    ) -> None:
        log_path = tmp_path / "log.json"
        with (
            patch("Skills.star_scanner._load_toi_tic_ids", return_value=set()),
            patch("Skills.star_scanner._load_ctoi_tic_ids", return_value=set()),
            patch("Skills.star_scanner._load_confirmed_host_tic_ids", return_value=frozenset()),
            patch("Skills.star_scanner.select_targets", return_value=self._targets()),
            patch("Skills.star_scanner.run_pipeline", return_value=[]),
        ):
            summary = run_background_scan(log_path, n_targets=10)
        scanned_tic_ids = {row["tic_id"] for row in summary["scanned_this_run"]}
        assert scanned_tic_ids == {1001, 1002}

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

    def test_ctoi_load_failure_warns_and_continues(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Regression: a CTOI/ExoFOP outage during a live scan used to
        # silently disable that exclusion category with nothing in the
        # console transcript -- asymmetric with the TOI branch just above,
        # which already warns. The scan must still complete.
        log_path = tmp_path / "log.json"
        with (
            patch("Skills.star_scanner._load_toi_tic_ids", return_value=set()),
            patch(
                "Skills.star_scanner._load_ctoi_tic_ids",
                side_effect=RuntimeError("CTOI source unavailable"),
            ),
            patch(
                "Skills.star_scanner._load_confirmed_host_tic_ids",
                return_value=frozenset(),
            ),
            patch("Skills.star_scanner.select_targets", return_value=self._targets()),
            patch("Skills.star_scanner.run_pipeline", return_value=[]),
        ):
            run_background_scan(log_path, n_targets=10)

        out = capsys.readouterr().out
        assert "Warning: could not load CTOI list" in out
        assert "CTOI source unavailable" in out
        log = ScanLog(log_path)
        assert log.is_scanned(1001)
        assert log.is_scanned(1002)

    def test_confirmed_host_load_failure_warns_and_continues(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        log_path = tmp_path / "log.json"
        with (
            patch("Skills.star_scanner._load_toi_tic_ids", return_value=set()),
            patch("Skills.star_scanner._load_ctoi_tic_ids", return_value=set()),
            patch(
                "Skills.star_scanner._load_confirmed_host_tic_ids",
                side_effect=RuntimeError("NEA archive unavailable"),
            ),
            patch("Skills.star_scanner.select_targets", return_value=self._targets()),
            patch("Skills.star_scanner.run_pipeline", return_value=[]),
        ):
            run_background_scan(log_path, n_targets=10)

        out = capsys.readouterr().out
        assert "Warning: could not load confirmed transiting planet hosts" in out
        assert "NEA archive unavailable" in out
        log = ScanLog(log_path)
        assert log.is_scanned(1001)
        assert log.is_scanned(1002)

    def test_full_sweep_forwarded_to_select_targets(self, tmp_path: Path) -> None:
        log_path = tmp_path / "log.json"
        captured: dict[str, Any] = {}

        def fake_select(*args: Any, **kwargs: Any) -> list:
            captured["full_sweep"] = kwargs.get("full_sweep")
            return []

        with (
            patch("Skills.star_scanner._load_toi_tic_ids", return_value=set()),
            patch("Skills.star_scanner._load_ctoi_tic_ids", return_value=set()),
            patch("Skills.star_scanner._load_confirmed_host_tic_ids", return_value=frozenset()),
            patch("Skills.star_scanner.select_targets", side_effect=fake_select),
        ):
            run_background_scan(log_path, n_targets=10, full_sweep=True)

        assert captured["full_sweep"] is True

    def test_exclude_known_variables_removes_matched_targets(
        self, tmp_path: Path
    ) -> None:
        log_path = tmp_path / "log.json"
        with (
            patch("Skills.star_scanner._load_toi_tic_ids", return_value=set()),
            patch("Skills.star_scanner._load_ctoi_tic_ids", return_value=set()),
            patch("Skills.star_scanner._load_confirmed_host_tic_ids", return_value=frozenset()),
            patch("Skills.star_scanner.select_targets", return_value=self._targets()),
            patch(
                "Skills.star_scanner._load_asassn_variable_tic_ids",
                return_value=frozenset({1001}),
            ),
            patch("Skills.star_scanner.run_pipeline", return_value=[]),
        ):
            run_background_scan(log_path, n_targets=10, exclude_known_variables=True)

        log = ScanLog(log_path)
        assert not log.is_scanned(1001)
        assert log.is_scanned(1002)

    def test_exclude_known_variables_off_by_default(self, tmp_path: Path) -> None:
        log_path = tmp_path / "log.json"
        called = {"n": 0}

        def fake_asassn(*args: Any, **kwargs: Any) -> frozenset:
            called["n"] += 1
            return frozenset()

        with (
            patch("Skills.star_scanner._load_toi_tic_ids", return_value=set()),
            patch("Skills.star_scanner._load_ctoi_tic_ids", return_value=set()),
            patch("Skills.star_scanner._load_confirmed_host_tic_ids", return_value=frozenset()),
            patch("Skills.star_scanner.select_targets", return_value=self._targets()),
            patch(
                "Skills.star_scanner._load_asassn_variable_tic_ids",
                side_effect=fake_asassn,
            ),
            patch("Skills.star_scanner.run_pipeline", return_value=[]),
        ):
            run_background_scan(log_path, n_targets=10)

        assert called["n"] == 0

    def test_search_log_written_to_path(self, tmp_path: Path) -> None:
        log_path = tmp_path / "log.json"
        search_log_path = tmp_path / "search_manifest.json"
        with (
            patch("Skills.star_scanner._load_toi_tic_ids", return_value=set()),
            patch("Skills.star_scanner._load_ctoi_tic_ids", return_value=set()),
            patch("Skills.star_scanner._load_confirmed_host_tic_ids", return_value=frozenset()),
            patch("Skills.star_scanner.select_targets", return_value=self._targets()),
            patch("Skills.star_scanner.run_pipeline", return_value=[]),
        ):
            run_background_scan(
                log_path, n_targets=10, search_log_path=search_log_path
            )

        assert search_log_path.exists()
        manifest = json.loads(search_log_path.read_text())
        assert manifest["final_target_count"] == 2
        assert manifest["requested_target_count"] == 10
        assert manifest["asassn_variables_excluded"] == 0

    def test_search_log_not_written_when_path_omitted(self, tmp_path: Path) -> None:
        log_path = tmp_path / "log.json"
        with (
            patch("Skills.star_scanner._load_toi_tic_ids", return_value=set()),
            patch("Skills.star_scanner._load_ctoi_tic_ids", return_value=set()),
            patch("Skills.star_scanner._load_confirmed_host_tic_ids", return_value=frozenset()),
            patch("Skills.star_scanner.select_targets", return_value=self._targets()),
            patch("Skills.star_scanner.run_pipeline", return_value=[]),
        ):
            run_background_scan(log_path, n_targets=10)
        assert not (tmp_path / "search_manifest.json").exists()


# ---------------------------------------------------------------------------
# Hunter durable-history bridge
# ---------------------------------------------------------------------------


class TestManualScanHistoryEntry:
    def test_candidate_found_maps_through_unchanged(self) -> None:
        entry = _manual_scan_history_entry(
            1, mission="TESS", result={"status": "candidate_found", "best_fpp": 0.1}
        )
        assert entry["target_id"] == "TIC 1"
        assert entry["status"] == "candidate_found"
        assert entry["metrics"]["best_fpp"] == 0.1
        assert "error_message" not in entry

    def test_scanned_clear_maps_to_no_signal(self) -> None:
        entry = _manual_scan_history_entry(1, mission="TESS", result={"status": "scanned_clear"})
        assert entry["status"] == "no_signal"

    def test_no_data_maps_through_unchanged(self) -> None:
        entry = _manual_scan_history_entry(1, mission="TESS", result={"status": "no_data"})
        assert entry["status"] == "no_data"

    def test_error_maps_to_failed_with_error_message(self) -> None:
        entry = _manual_scan_history_entry(
            1,
            mission="TESS",
            result={"status": "error", "error_message": "network failure"},
        )
        assert entry["status"] == "failed"
        assert entry["error_message"] == "network failure"


class TestBridgeManualScanToHunter:
    def test_empty_entries_is_a_noop(self, tmp_path: Path) -> None:
        assert (
            _bridge_manual_scan_to_hunter(
                script="star_scanner",
                log_path=tmp_path / "log.json",
                mission="TESS",
                entries=[],
                started_at=datetime.now(UTC),
                completed_at=datetime.now(UTC),
                hunter_db_path=tmp_path / "hunter.sqlite3",
                method_or_data="TESS QLP manual scan",
            )
            is None
        )

    def test_real_scan_durably_recorded_in_hunter_store(self, tmp_path: Path) -> None:
        log_path = tmp_path / "log.json"
        log = ScanLog(log_path)
        log.record(555, "candidate_found", {"best_fpp": 0.05})
        hunter_db = tmp_path / "hunter.sqlite3"

        summary = _bridge_manual_scan_to_hunter(
            script="star_scanner",
            log_path=log_path,
            mission="TESS",
            entries=[
                _manual_scan_history_entry(
                    555, mission="TESS", result={"status": "candidate_found"}
                )
            ],
            started_at=datetime(2026, 7, 24, tzinfo=UTC),
            completed_at=datetime.now(UTC),
            hunter_db_path=hunter_db,
            method_or_data="TESS QLP manual scan",
        )

        assert summary is not None
        assert summary["sources_created"] == 1
        store = HunterStore(hunter_db)
        assert "TIC 555" in store.searched_target_ids()

    def test_missing_log_file_warns_but_returns_none(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        result = _bridge_manual_scan_to_hunter(
            script="star_scanner",
            log_path=tmp_path / "does_not_exist.json",
            mission="TESS",
            entries=[
                _manual_scan_history_entry(
                    1, mission="TESS", result={"status": "scanned_clear"}
                )
            ],
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            hunter_db_path=tmp_path / "hunter.sqlite3",
            method_or_data="TESS QLP manual scan",
        )
        assert result is None
        assert "Warning" in capsys.readouterr().err


class TestCliHunterBridgeWiring:
    def test_target_mode_bridges_by_default(self, tmp_path: Path) -> None:
        log_path = tmp_path / "log.json"
        hunter_db = tmp_path / "hunter.sqlite3"
        with (
            patch("Skills.star_scanner.run_pipeline", return_value=[]),
            patch(
                "Skills.star_scanner.run_and_commit_report", return_value=True
            ),
        ):
            exit_code = main(
                [
                    "--target",
                    "42",
                    "--log",
                    str(log_path),
                    "--hunter-db",
                    str(hunter_db),
                ]
            )
        assert exit_code == 0
        assert hunter_db.exists()
        assert "TIC 42" in HunterStore(hunter_db).searched_target_ids()

    def test_target_mode_no_hunter_bridge_flag_skips_it(self, tmp_path: Path) -> None:
        log_path = tmp_path / "log.json"
        hunter_db = tmp_path / "hunter.sqlite3"
        with (
            patch("Skills.star_scanner.run_pipeline", return_value=[]),
            patch(
                "Skills.star_scanner.run_and_commit_report", return_value=True
            ),
        ):
            exit_code = main(
                [
                    "--target",
                    "42",
                    "--log",
                    str(log_path),
                    "--hunter-db",
                    str(hunter_db),
                    "--no-hunter-bridge",
                ]
            )
        assert exit_code == 0
        assert not hunter_db.exists()

    def test_background_scan_mode_bridges_scanned_targets(self, tmp_path: Path) -> None:
        log_path = tmp_path / "log.json"
        hunter_db = tmp_path / "hunter.sqlite3"
        targets = [
            {"tic_id": 1001, "tmag": 12.0, "teff": 4500.0,
             "contratio": 0.01, "priority": 0.90},
        ]
        with (
            patch("Skills.star_scanner._load_toi_tic_ids", return_value=set()),
            patch("Skills.star_scanner._load_ctoi_tic_ids", return_value=set()),
            patch(
                "Skills.star_scanner._load_confirmed_host_tic_ids",
                return_value=frozenset(),
            ),
            patch("Skills.star_scanner.select_targets", return_value=targets),
            patch("Skills.star_scanner.run_pipeline", return_value=[]),
            patch(
                "Skills.star_scanner.run_and_commit_report", return_value=True
            ),
        ):
            exit_code = main(
                [
                    "--log",
                    str(log_path),
                    "--max-stars",
                    "10",
                    "--hunter-db",
                    str(hunter_db),
                ]
            )
        assert exit_code == 0
        assert hunter_db.exists()
        assert "TIC 1001" in HunterStore(hunter_db).searched_target_ids()

    def test_prepare_only_mode_never_bridges(self, tmp_path: Path) -> None:
        log_path = tmp_path / "log.json"
        hunter_db = tmp_path / "hunter.sqlite3"
        target = {
            "tic_id": 101, "tmag": 12.0, "teff": 4500.0,
            "contratio": 0.01, "radius_rsun": 1.0, "priority": 0.9,
        }
        with (
            patch("Skills.star_scanner._load_toi_tic_ids", return_value=set()),
            patch("Skills.star_scanner._load_ctoi_tic_ids", return_value=set()),
            patch(
                "Skills.star_scanner._load_confirmed_host_tic_ids",
                return_value=frozenset(),
            ),
            patch("Skills.star_scanner.select_targets", return_value=[target]),
            patch("Skills.star_scanner.prepare_live_search_snapshot"),
            patch(
                "Skills.star_scanner._bridge_manual_scan_to_hunter"
            ) as bridge,
            patch(
                "Skills.star_scanner.run_and_commit_report", return_value=True
            ),
        ):
            main(
                [
                    "--prepare-only",
                    "--log",
                    str(log_path),
                    "--queue-path",
                    str(tmp_path / "queue.csv"),
                    "--queue-snapshot-path",
                    str(tmp_path / "snapshot.csv"),
                    "--batch-manifest-path",
                    str(tmp_path / "batch.json"),
                    "--dataset-manifest-path",
                    str(tmp_path / "dataset.json"),
                    "--hunter-db",
                    str(hunter_db),
                    "--no-git-report",
                ]
            )
        bridge.assert_not_called()
        assert not hunter_db.exists()
