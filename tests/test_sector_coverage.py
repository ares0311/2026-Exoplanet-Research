"""Tests for Skills/sector_coverage.py."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Skills.sector_coverage import (  # noqa: E402
    SectorCoverage,
    format_coverage_table,
    get_sector_coverage,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_search(sectors: list[int], exptimes: list[float]) -> MagicMock:
    """Build a mock search result with the given sector numbers and exptimes."""
    rows = []
    for sec, exp in zip(sectors, exptimes, strict=True):
        row = {"sequence_number": sec, "exptime": exp}
        rows.append(row)

    mock_result = MagicMock()
    mock_result.__len__.return_value = len(rows)

    table = MagicMock()
    table.colnames = ["sequence_number", "exptime"]
    table.__iter__ = MagicMock(return_value=iter(rows))
    mock_result.table = table

    return lambda *args, **kwargs: mock_result


def _empty_search() -> MagicMock:
    """Build a mock search result with no rows."""
    mock_result = MagicMock()
    mock_result.__len__.return_value = 0
    return lambda *args, **kwargs: mock_result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGetSectorCoverage:
    def test_returns_sector_coverage_object(self) -> None:
        cov = get_sector_coverage(
            "TIC 100", search_fn=_mock_search([1, 2, 3], [120.0, 120.0, 120.0])
        )
        assert isinstance(cov, SectorCoverage)

    def test_sectors_sorted(self) -> None:
        cov = get_sector_coverage(
            "TIC 100", search_fn=_mock_search([5, 1, 3], [120.0, 120.0, 120.0])
        )
        assert cov.sectors == [1, 3, 5]

    def test_n_sectors_matches_sectors_list(self) -> None:
        cov = get_sector_coverage(
            "TIC 100", search_fn=_mock_search([1, 2], [120.0, 120.0])
        )
        assert cov.n_sectors == len(cov.sectors)

    def test_empty_result_gives_zero_sectors(self) -> None:
        cov = get_sector_coverage("TIC 999", search_fn=_empty_search())
        assert cov.n_sectors == 0
        assert cov.sectors == []

    def test_target_id_prefixed_with_tic(self) -> None:
        cov = get_sector_coverage("150428135", search_fn=_empty_search())
        assert cov.target_id.startswith("TIC")

    def test_cadence_label_2min(self) -> None:
        cov = get_sector_coverage(
            "TIC 100", search_fn=_mock_search([1], [120.0])
        )
        assert "2min" in cov.cadence_labels

    def test_cadence_label_10min(self) -> None:
        cov = get_sector_coverage(
            "TIC 100", search_fn=_mock_search([1], [600.0])
        )
        assert "10min" in cov.cadence_labels

    def test_to_dict_contains_required_keys(self) -> None:
        cov = get_sector_coverage(
            "TIC 100", search_fn=_mock_search([1, 2], [120.0, 120.0])
        )
        d = cov.to_dict()
        for key in ("target_id", "pipeline", "sectors", "n_sectors", "cadence_labels"):
            assert key in d


class TestFormatCoverageTable:
    def test_returns_string(self) -> None:
        cov = SectorCoverage("TIC 1", "SPOC", [1, 2], 2, ["2min", "2min"])
        assert isinstance(format_coverage_table([cov]), str)

    def test_empty_list_returns_placeholder(self) -> None:
        assert "no targets" in format_coverage_table([])
