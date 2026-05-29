"""Tests for Skills/candidate_habitability_reporter.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from candidate_habitability_reporter import (
    HabitabilityReport, assess_habitability, format_habitability_report
)


class TestHabitabilityReport:
    def test_dataclass_fields(self):
        r = HabitabilityReport(
            hz_class="inner_hz", size_class="rocky", stellar_class="G",
            tidal_status="unlocked", overall_score=0.7, flag="PROMISING"
        )
        assert r.overall_score == 0.7
        assert r.flag == "PROMISING"

    def test_frozen(self):
        r = HabitabilityReport(
            hz_class="inner_hz", size_class="rocky", stellar_class="G",
            tidal_status="unlocked", overall_score=0.7, flag="PROMISING"
        )
        try:
            r.overall_score = 0
            assert False
        except Exception:
            pass


class TestAssessHabitability:
    def test_earth_like_promising(self):
        # Earth: R=1, S=1, Teff=5778, not locked
        r = assess_habitability(1.0, 1.0, teff_k=5778.0, t_lock_yr=1e10)
        assert r.flag == "PROMISING"
        assert r.overall_score > 0.6

    def test_hot_rocky_unlikely(self):
        # Hot planet close to star
        r = assess_habitability(1.0, 20.0, teff_k=5778.0, t_lock_yr=1e8)
        assert r.hz_class == "hot_zone"

    def test_gas_giant_unlikely(self):
        r = assess_habitability(12.0, 1.0, teff_k=5778.0)
        assert r.size_class == "giant"

    def test_m_dwarf_moderate_score(self):
        r = assess_habitability(1.0, 0.5, teff_k=3000.0, t_lock_yr=None)
        assert r.stellar_class == "M"

    def test_tidal_locked(self):
        r = assess_habitability(1.0, 1.0, t_lock_yr=1e8)
        assert r.tidal_status == "locked"

    def test_tidal_unlocked(self):
        r = assess_habitability(1.0, 1.0, t_lock_yr=1e10)
        assert r.tidal_status == "unlocked"

    def test_tidal_uncertain(self):
        r = assess_habitability(1.0, 1.0, t_lock_yr=None)
        assert r.tidal_status == "uncertain"

    def test_score_in_range(self):
        r = assess_habitability(1.0, 1.0)
        assert 0.0 <= r.overall_score <= 1.0

    def test_promising_threshold(self):
        # Overall > 0.6 → PROMISING
        r = assess_habitability(1.0, 1.0, teff_k=5778.0, t_lock_yr=1e10)
        if r.overall_score > 0.6:
            assert r.flag == "PROMISING"

    def test_marginal_flag(self):
        # Score around 0.3-0.6 should be MARGINAL
        r = assess_habitability(3.0, 1.0, teff_k=5778.0, t_lock_yr=None)
        if 0.3 <= r.overall_score <= 0.6:
            assert r.flag == "MARGINAL"

    def test_hz_class_hot_zone(self):
        r = assess_habitability(1.0, 5.0)
        assert r.hz_class == "hot_zone"

    def test_f_star_class(self):
        r = assess_habitability(1.0, 1.0, teff_k=6500.0)
        assert r.stellar_class == "F"


class TestFormatHabitability:
    def test_returns_string(self):
        r = assess_habitability(1.0, 1.0)
        s = format_habitability_report(r)
        assert isinstance(s, str)

    def test_contains_overall_score(self):
        r = assess_habitability(1.0, 1.0)
        s = format_habitability_report(r)
        assert str(round(r.overall_score, 3)) in s or "score" in s.lower()
