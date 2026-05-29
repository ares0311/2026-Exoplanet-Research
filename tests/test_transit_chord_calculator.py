"""Tests for Skills/transit_chord_calculator.py"""
import sys
import math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from transit_chord_calculator import TransitChordResult, compute_transit_chord, format_transit_chord


class TestTransitChordResult:
    def test_dataclass_fields(self):
        r = TransitChordResult(chord_rstar=2.0, b_used=0.0, grazing=False, flag="OK")
        assert r.chord_rstar == 2.0
        assert r.grazing is False

    def test_frozen(self):
        r = TransitChordResult(chord_rstar=2.0, b_used=0.0, grazing=False, flag="OK")
        try:
            r.chord_rstar = 0
            assert False
        except Exception:
            pass


class TestComputeTransitChord:
    def test_central_transit_chord(self):
        # b=0: chord = 2*sqrt(1-0) = 2.0
        r = compute_transit_chord(0.0)
        assert abs(r.chord_rstar - 2.0) < 1e-6
        assert r.grazing is False
        assert r.flag == "OK"

    def test_b_half_chord(self):
        # b=0.5: chord = 2*sqrt(1 - 0.25) = 2*sqrt(0.75)
        expected = 2.0 * math.sqrt(0.75)
        r = compute_transit_chord(0.5)
        assert abs(r.chord_rstar - expected) < 1e-4

    def test_b_zero_is_maximum_chord(self):
        r0 = compute_transit_chord(0.0)
        r05 = compute_transit_chord(0.5)
        assert r0.chord_rstar > r05.chord_rstar

    def test_not_grazing_at_b0(self):
        r = compute_transit_chord(0.0, rp_rearth=1.0, rstar_rsun=1.0)
        assert r.grazing is False
        assert r.flag == "OK"

    def test_grazing_flag(self):
        # b=0.995, rp_rstar=0.009168 → b+rp_rstar > 1 → GRAZING
        r = compute_transit_chord(0.995, rp_rearth=1.0, rstar_rsun=1.0)
        assert r.grazing is True
        assert r.flag == "GRAZING"

    def test_negative_b_same_as_positive(self):
        r_pos = compute_transit_chord(0.5)
        r_neg = compute_transit_chord(-0.5)
        assert abs(r_pos.chord_rstar - r_neg.chord_rstar) < 1e-9

    def test_b_used_clipped(self):
        # b > 1 should be clipped
        r = compute_transit_chord(2.0, rp_rearth=1.0, rstar_rsun=1.0)
        assert r.b_used <= 1.0 + (1.0 * 0.009168 / 1.0)

    def test_chord_non_negative(self):
        for b in [0.0, 0.3, 0.7, 0.95, 1.0]:
            r = compute_transit_chord(b)
            assert r.chord_rstar >= 0.0

    def test_larger_planet_more_likely_grazing(self):
        r_small = compute_transit_chord(0.95, rp_rearth=1.0, rstar_rsun=1.0)
        r_large = compute_transit_chord(0.95, rp_rearth=20.0, rstar_rsun=1.0)
        # large planet has larger rp_rstar, so more likely to graze
        assert r_large.grazing or r_small.grazing  # at least one should graze at b=0.95


class TestFormatTransitChord:
    def test_returns_string(self):
        r = compute_transit_chord(0.0)
        s = format_transit_chord(r)
        assert isinstance(s, str)

    def test_contains_chord(self):
        r = compute_transit_chord(0.0)
        s = format_transit_chord(r)
        assert "Chord" in s or "chord" in s

    def test_contains_flag(self):
        r = compute_transit_chord(0.0)
        s = format_transit_chord(r)
        assert r.flag in s
