"""Tests for Skills.multi_sector_phase_compare."""
from __future__ import annotations

from Skills.multi_sector_phase_compare import (
    PhaseComparisonResult,
    SectorPhaseMetrics,
    compare_sector_phase_folds,
    format_phase_comparison,
)


def _sector(
    sector: int,
    *,
    depth_ppm: float = 1_000.0,
    phase_shift_days: float = 0.0,
    period: float = 2.0,
    epoch: float = 0.0,
    duration: float = 0.2,
) -> dict:
    time: list[float] = []
    flux: list[float] = []
    cadence = 0.02
    start = (sector - 1) * 10.0 - 1.0
    for index in range(101):
        t = start + index * cadence
        nearest_transit = round((t - epoch - phase_shift_days) / period)
        mid = epoch + phase_shift_days + nearest_transit * period
        in_transit = abs(t - mid) <= duration / 2.0
        time.append(t)
        flux.append(1.0 - depth_ppm / 1_000_000.0 if in_transit else 1.0)
    return {"sector": sector, "time": time, "flux": flux}


def test_compare_returns_result() -> None:
    result = compare_sector_phase_folds(
        [_sector(1), _sector(2)],
        2.0,
        0.0,
        duration_days=0.2,
    )
    assert isinstance(result, PhaseComparisonResult)


def test_sector_metrics_are_recorded() -> None:
    result = compare_sector_phase_folds(
        [_sector(3), _sector(4)],
        2.0,
        0.0,
        duration_days=0.2,
    )
    assert all(isinstance(metric, SectorPhaseMetrics) for metric in result.sectors)
    assert [metric.sector for metric in result.sectors] == [3, 4]


def test_consistent_depths_flag_consistent() -> None:
    result = compare_sector_phase_folds(
        [_sector(1, depth_ppm=900.0), _sector(2, depth_ppm=1_000.0)],
        2.0,
        0.0,
        duration_days=0.2,
        depth_tolerance_ppm=500.0,
    )
    assert result.flag == "CONSISTENT"


def test_depth_mismatch_flagged() -> None:
    result = compare_sector_phase_folds(
        [_sector(1, depth_ppm=500.0), _sector(2, depth_ppm=2_000.0)],
        2.0,
        0.0,
        duration_days=0.2,
        depth_tolerance_ppm=500.0,
    )
    assert result.flag == "DEPTH_MISMATCH"
    assert result.max_depth_delta_ppm is not None
    assert result.max_depth_delta_ppm > 500.0


def test_phase_shift_flagged_when_depths_match() -> None:
    result = compare_sector_phase_folds(
        [_sector(1), _sector(2, phase_shift_days=0.08)],
        2.0,
        0.0,
        duration_days=0.2,
        phase_tolerance=0.015,
    )
    assert result.flag == "PHASE_SHIFT"


def test_invalid_period_is_insufficient() -> None:
    result = compare_sector_phase_folds(
        [_sector(1), _sector(2)],
        0.0,
        0.0,
        duration_days=0.2,
    )
    assert result.flag == "INSUFFICIENT_SECTORS"


def test_empty_input_is_insufficient() -> None:
    result = compare_sector_phase_folds([], 2.0, 0.0, duration_days=0.2)
    assert result.flag == "INSUFFICIENT_SECTORS"


def test_empty_sector_marks_partial() -> None:
    result = compare_sector_phase_folds(
        [_sector(1), {"sector": 2, "time": [], "flux": []}],
        2.0,
        0.0,
        duration_days=0.2,
    )
    assert result.flag == "PARTIAL"


def test_inverted_signal_flagged() -> None:
    inverted = _sector(1, depth_ppm=-500.0)
    result = compare_sector_phase_folds(
        [inverted, _sector(2, depth_ppm=-400.0)],
        2.0,
        0.0,
        duration_days=0.2,
    )
    assert result.flag == "WEAK_OR_INVERTED"


def test_default_sector_numbers_are_assigned() -> None:
    first = _sector(1)
    second = _sector(2)
    first.pop("sector")
    second.pop("sector")
    result = compare_sector_phase_folds([first, second], 2.0, 0.0, duration_days=0.2)
    assert [metric.sector for metric in result.sectors] == [1, 2]


def test_format_contains_table_and_flag() -> None:
    result = compare_sector_phase_folds(
        [_sector(1), _sector(2)],
        2.0,
        0.0,
        duration_days=0.2,
    )
    text = format_phase_comparison(result)
    assert "Multi-Sector Phase-Fold Comparison" in text
    assert "Overall flag" in text
    assert "| Sector |" in text


def test_depth_estimate_is_positive_for_transit() -> None:
    result = compare_sector_phase_folds(
        [_sector(1, depth_ppm=1_200.0), _sector(2, depth_ppm=1_200.0)],
        2.0,
        0.0,
        duration_days=0.2,
    )
    assert result.sectors[0].depth_ppm is not None
    assert result.sectors[0].depth_ppm > 0.0
