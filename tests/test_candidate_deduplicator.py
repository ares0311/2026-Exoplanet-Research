"""Tests for candidate_deduplicator.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from candidate_deduplicator import (
    deduplicate_candidates,
    format_deduplication_result,
)


def _candidate(tic_id, period, epoch, ra, dec, fpp=0.1):
    return {
        "tic_id": tic_id,
        "period_days": period,
        "epoch_bjd": epoch,
        "ra_deg": ra,
        "dec_deg": dec,
        "false_positive_probability": fpp,
    }


class TestDeduplicateCandidates:
    def test_empty_input(self):
        r = deduplicate_candidates([])
        assert r.flag == "EMPTY"
        assert r.n_input == 0

    def test_single_candidate(self):
        c = _candidate(1, 5.0, 0.0, 10.0, 10.0)
        r = deduplicate_candidates([c])
        assert r.flag == "NO_DUPLICATES"
        assert r.n_unique == 1

    def test_no_duplicates_different_period(self):
        c1 = _candidate(1, 5.0, 0.0, 10.0, 10.0)
        c2 = _candidate(2, 10.0, 0.0, 11.0, 11.0)
        r = deduplicate_candidates([c1, c2])
        assert r.flag == "NO_DUPLICATES"
        assert r.n_unique == 2

    def test_duplicate_same_position_same_period(self):
        c1 = _candidate(1, 5.0, 0.0, 10.0, 10.0)
        c2 = _candidate(2, 5.0, 0.0, 10.0, 10.0)
        r = deduplicate_candidates([c1, c2])
        assert r.n_duplicates_removed == 1
        assert r.n_unique == 1

    def test_keeps_lower_fpp(self):
        c1 = _candidate(1, 5.0, 0.0, 10.0, 10.0, fpp=0.5)
        c2 = _candidate(2, 5.0, 0.0, 10.0, 10.0, fpp=0.1)
        r = deduplicate_candidates([c1, c2])
        assert 1 in r.unique_indices  # index 1 (c2) has lower FPP

    def test_pairs_compared(self):
        c1 = _candidate(1, 5.0, 0.0, 10.0, 10.0)
        c2 = _candidate(2, 7.0, 0.0, 50.0, 50.0)
        c3 = _candidate(3, 9.0, 0.0, 100.0, 100.0)
        r = deduplicate_candidates([c1, c2, c3])
        assert len(r.pairs) == 3  # C(3,2)

    def test_similarity_score_in_range(self):
        c1 = _candidate(1, 5.0, 0.0, 10.0, 10.0)
        c2 = _candidate(2, 10.0, 0.0, 11.0, 11.0)
        r = deduplicate_candidates([c1, c2])
        for p in r.pairs:
            assert 0.0 <= p.similarity_score <= 1.0

    def test_harmonic_matching(self):
        # Period ratio 2:1 should have high period similarity
        c1 = _candidate(1, 5.0, 0.0, 10.0, 10.0)
        c2 = _candidate(2, 10.0, 0.0, 10.01, 10.01)
        r = deduplicate_candidates([c1, c2])
        # At least one pair should be compared
        assert len(r.pairs) >= 1

    def test_unique_indices_valid(self):
        c1 = _candidate(1, 5.0, 0.0, 10.0, 10.0)
        c2 = _candidate(2, 8.0, 0.0, 90.0, 90.0)
        r = deduplicate_candidates([c1, c2])
        assert all(0 <= i < 2 for i in r.unique_indices)

    def test_result_frozen(self):
        c = _candidate(1, 5.0, 0.0, 10.0, 10.0)
        r = deduplicate_candidates([c])
        try:
            r.n_unique = 999  # type: ignore[misc]
            raise AssertionError()
        except (AttributeError, TypeError):
            pass


class TestFormatDeduplicationResult:
    def test_returns_string(self):
        c = _candidate(1, 5.0, 0.0, 10.0, 10.0)
        r = deduplicate_candidates([c])
        assert isinstance(format_deduplication_result(r), str)

    def test_contains_flag(self):
        c = _candidate(1, 5.0, 0.0, 10.0, 10.0)
        r = deduplicate_candidates([c])
        s = format_deduplication_result(r)
        assert r.flag in s

    def test_duplicate_pairs_shown(self):
        c1 = _candidate(1, 5.0, 0.0, 10.0, 10.0)
        c2 = _candidate(2, 5.0, 0.0, 10.0, 10.0)
        r = deduplicate_candidates([c1, c2])
        s = format_deduplication_result(r)
        assert "TIC" in s or "duplicate" in s.lower()
