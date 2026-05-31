"""Tests for Skills/exoplanet_archive_formatter.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from exoplanet_archive_formatter import format_archive_record, format_archive_record_markdown


def _make_candidate(**kwargs) -> dict:
    base = {
        "tic_id": "150428135",
        "period_days": 37.4,
        "epoch_bjd": 2458325.6,
        "depth_ppm": 1200.0,
        "duration_hours": 3.5,
        "false_positive_probability": 0.05,
        "pathway": "tfop_ready",
    }
    base.update(kwargs)
    return base


class TestExoplanetArchiveFormatter:
    def test_basic_record(self) -> None:
        r = format_archive_record(_make_candidate())
        assert r.flag == "OK"
        assert r.tic_id == "150428135"

    def test_missing_tic_id(self) -> None:
        r = format_archive_record({"period_days": 5.0})
        assert r.flag == "MISSING_TIC_ID"

    def test_low_fpp_disposition_pc(self) -> None:
        r = format_archive_record(_make_candidate(false_positive_probability=0.02))
        assert r.disposition == "PC"

    def test_high_fpp_disposition_fp(self) -> None:
        r = format_archive_record(_make_candidate(false_positive_probability=0.80))
        assert r.disposition == "FP"

    def test_mid_fpp_disposition_apc(self) -> None:
        r = format_archive_record(_make_candidate(false_positive_probability=0.25))
        assert r.disposition == "APC"

    def test_period_extracted(self) -> None:
        r = format_archive_record(_make_candidate())
        assert r.period_days == 37.4

    def test_epoch_extracted(self) -> None:
        r = format_archive_record(_make_candidate())
        assert r.epoch_bjd == 2458325.6

    def test_depth_extracted(self) -> None:
        r = format_archive_record(_make_candidate())
        assert r.depth_ppm == 1200.0

    def test_pathway_stored(self) -> None:
        r = format_archive_record(_make_candidate())
        assert r.pathway == "tfop_ready"

    def test_nested_signal(self) -> None:
        cand = {
            "tic_id": "999",
            "signal": {"period_days": 5.0, "epoch_bjd": 2459000.0, "depth_ppm": 500.0},
        }
        r = format_archive_record(cand)
        assert r.period_days == 5.0

    def test_nested_scores_fpp(self) -> None:
        cand = {
            "tic_id": "888",
            "scores": {"false_positive_probability": 0.03},
        }
        r = format_archive_record(cand)
        assert r.fpp == 0.03

    def test_format_markdown_returns_string(self) -> None:
        r = format_archive_record(_make_candidate())
        s = format_archive_record_markdown(r)
        assert isinstance(s, str)
        assert "TIC" in s
