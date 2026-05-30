"""Tests for Skills/candidate_disposition_scorer.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from candidate_disposition_scorer import (
    format_disposition_result,
    score_disposition,
)


class TestScoreDisposition:
    def test_clean_signal_pc(self) -> None:
        r = score_disposition(0.05, 0.0, 0.0, 0.0)
        assert r.disposition == "PC"
        assert r.flag == "OK"

    def test_strong_odd_even_eb(self) -> None:
        r = score_disposition(0.05, 5.0, 5.0, 0.0)
        assert r.disposition in ("EB", "FP", "PC")

    def test_large_centroid_neb(self) -> None:
        r = score_disposition(0.1, 0.0, 0.0, 10.0)
        assert r.disposition == "NEB"

    def test_high_fpp(self) -> None:
        r = score_disposition(0.9, 0.0, 0.0, 0.0)
        assert r.pc_score < 1.0

    def test_invalid_fpp_low(self) -> None:
        r = score_disposition(-0.1, 0.0, 0.0, 0.0)
        assert r.flag == "INVALID_FPP"

    def test_invalid_fpp_high(self) -> None:
        r = score_disposition(1.5, 0.0, 0.0, 0.0)
        assert r.flag == "INVALID_FPP"

    def test_flag_ok_valid(self) -> None:
        r = score_disposition(0.1, 0.0, 0.0, 0.0)
        assert r.flag == "OK"

    def test_all_scores_in_range(self) -> None:
        r = score_disposition(0.3, 2.0, 2.0, 1.0)
        assert 0.0 <= r.pc_score <= 1.0
        assert 0.0 <= r.fp_score <= 1.0
        assert 0.0 <= r.eb_score <= 1.0
        assert 0.0 <= r.neb_score <= 1.0

    def test_disposition_is_string(self) -> None:
        r = score_disposition(0.1, 0.0, 0.0, 0.0)
        assert isinstance(r.disposition, str)

    def test_centroid_drives_neb_score(self) -> None:
        r_no = score_disposition(0.1, 0.0, 0.0, 0.0)
        r_yes = score_disposition(0.1, 0.0, 0.0, 4.0)
        assert r_yes.neb_score > r_no.neb_score

    def test_combined_flags_fp(self) -> None:
        r = score_disposition(0.8, 4.0, 4.0, 0.0)
        assert r.pc_score < 0.5

    def test_format_contains_disposition(self) -> None:
        r = score_disposition(0.05, 0.0, 0.0, 0.0)
        s = format_disposition_result(r)
        assert r.disposition in s
