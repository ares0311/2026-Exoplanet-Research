"""Tests for Skills/candidate_fingerprinter.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from candidate_fingerprinter import fingerprint, format_fingerprint_result


class TestCandidateFingerprinter:
    def test_basic_fingerprint(self) -> None:
        r = fingerprint("12345", 5.0, 2459000.0, 1000.0)
        assert r.flag == "OK"
        assert len(r.fingerprint) == 64

    def test_deterministic(self) -> None:
        r1 = fingerprint("12345", 5.0, 2459000.0, 1000.0)
        r2 = fingerprint("12345", 5.0, 2459000.0, 1000.0)
        assert r1.fingerprint == r2.fingerprint

    def test_different_period_different_hash(self) -> None:
        r1 = fingerprint("12345", 5.0, 2459000.0, 1000.0)
        r2 = fingerprint("12345", 6.0, 2459000.0, 1000.0)
        assert r1.fingerprint != r2.fingerprint

    def test_different_tic_different_hash(self) -> None:
        r1 = fingerprint("12345", 5.0, 2459000.0, 1000.0)
        r2 = fingerprint("99999", 5.0, 2459000.0, 1000.0)
        assert r1.fingerprint != r2.fingerprint

    def test_rounding_tolerance(self) -> None:
        r1 = fingerprint("12345", 5.00001, 2459000.0, 1000.0)
        r2 = fingerprint("12345", 5.00002, 2459000.0, 1000.0)
        # Both round to 5.0000 (4 decimals) => same hash
        assert r1.fingerprint == r2.fingerprint

    def test_invalid_tic_id(self) -> None:
        r = fingerprint("", 5.0, 2459000.0, 1000.0)
        assert r.flag == "INVALID_TIC_ID"

    def test_invalid_period(self) -> None:
        r = fingerprint("12345", 0.0, 2459000.0, 1000.0)
        assert r.flag == "INVALID_PERIOD"

    def test_invalid_epoch(self) -> None:
        import math
        r = fingerprint("12345", 5.0, math.nan, 1000.0)
        assert r.flag == "INVALID_EPOCH"

    def test_invalid_depth(self) -> None:
        r = fingerprint("12345", 5.0, 2459000.0, 0.0)
        assert r.flag == "INVALID_DEPTH"

    def test_short_fp_is_prefix(self) -> None:
        r = fingerprint("12345", 5.0, 2459000.0, 1000.0)
        assert r.fingerprint.startswith(r.short_fp)

    def test_short_fp_length(self) -> None:
        r = fingerprint("12345", 5.0, 2459000.0, 1000.0)
        assert len(r.short_fp) == 12

    def test_format_returns_string(self) -> None:
        r = fingerprint("12345", 5.0, 2459000.0, 1000.0)
        s = format_fingerprint_result(r)
        assert isinstance(s, str)
        assert "Fingerprint" in s
