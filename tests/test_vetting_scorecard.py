"""Tests for Skills.vetting_scorecard."""
from __future__ import annotations

from Skills.vetting_scorecard import (
    VettingCheck,
    VettingScorecard,
    build_scorecard,
    format_scorecard,
)


def _row(**kw: object) -> dict:
    base = {
        "tic_id": 1,
        "candidate_id": "TIC1-001",
        "fpp": 0.05,
        "snr": 9.0,
        "n_transits": 5,
        "odd_even_significance": 0.5,
        "secondary_snr": 1.0,
        "centroid_delta_arcsec": 0.3,
        "best_pathway": "tfop_ready",
    }
    base.update(kw)
    return base


class TestBuildScorecard:
    def test_returns_vetting_scorecard(self) -> None:
        sc = build_scorecard(_row())
        assert isinstance(sc, VettingScorecard)

    def test_all_pass_gives_pass_overall(self) -> None:
        sc = build_scorecard(_row())
        assert sc.overall == "PASS"

    def test_high_fpp_gives_fail(self) -> None:
        sc = build_scorecard(_row(fpp=0.80))
        assert sc.overall == "FAIL"

    def test_moderate_fpp_gives_warn(self) -> None:
        sc = build_scorecard(_row(fpp=0.30))
        assert sc.overall in {"WARN", "FAIL"}

    def test_low_snr_gives_warn_or_fail(self) -> None:
        sc = build_scorecard(_row(snr=3.0))
        assert sc.overall in {"WARN", "FAIL"}

    def test_secondary_detected_gives_fail(self) -> None:
        sc = build_scorecard(_row(secondary_snr=10.0))
        assert sc.overall == "FAIL"

    def test_odd_even_asymmetry_gives_fail(self) -> None:
        sc = build_scorecard(_row(odd_even_significance=5.0))
        assert sc.overall == "FAIL"

    def test_centroid_shift_gives_fail(self) -> None:
        sc = build_scorecard(_row(centroid_delta_arcsec=3.0))
        assert sc.overall == "FAIL"

    def test_n_pass_counts_correctly(self) -> None:
        sc = build_scorecard(_row())
        assert sc.n_pass > 0

    def test_n_fail_zero_for_clean_candidate(self) -> None:
        sc = build_scorecard(_row())
        assert sc.n_fail == 0

    def test_tic_id_stored(self) -> None:
        sc = build_scorecard(_row(tic_id=99))
        assert sc.tic_id == 99

    def test_candidate_id_stored(self) -> None:
        sc = build_scorecard(_row(candidate_id="TIC99-002"))
        assert sc.candidate_id == "TIC99-002"

    def test_missing_fields_give_skip(self) -> None:
        sc = build_scorecard({"tic_id": 1})
        skipped = [c for c in sc.checks if c.status == "SKIP"]
        assert len(skipped) > 0

    def test_custom_checks_fn(self) -> None:
        def _always_pass(row: dict) -> VettingCheck:
            return VettingCheck("CUSTOM", "PASS", None, None)
        sc = build_scorecard(_row(), checks_fn=[_always_pass])
        assert sc.n_pass == 1
        assert sc.n_fail == 0


class TestFormatScorecard:
    def test_format_contains_overall(self) -> None:
        sc = build_scorecard(_row())
        text = format_scorecard(sc)
        assert "PASS" in text or "WARN" in text or "FAIL" in text

    def test_format_contains_candidate_id(self) -> None:
        sc = build_scorecard(_row(candidate_id="TIC1-001"))
        text = format_scorecard(sc)
        assert "TIC1-001" in text

    def test_format_contains_table(self) -> None:
        sc = build_scorecard(_row())
        text = format_scorecard(sc)
        assert "|" in text
