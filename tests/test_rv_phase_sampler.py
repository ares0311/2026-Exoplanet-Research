"""Tests for Skills/rv_phase_sampler.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from rv_phase_sampler import format_rv_phases, sample_rv_phases


def test_basic_ok():
    r = sample_rv_phases(10.0, 8)
    assert r.flag == "OK"
    assert r.n_obs == 8
    assert len(r.phases) == 8


def test_phases_in_unit_interval():
    r = sample_rv_phases(5.0, 6)
    for p in r.phases:
        assert 0.0 <= p < 1.0


def test_phase_offset_pattern():
    """Phases should be at (i + 0.1) / n_obs."""
    n = 4
    r = sample_rv_phases(10.0, n)
    expected = [(i + 0.1) / n % 1.0 for i in range(n)]
    for obs_p, exp_p in zip(r.phases, expected, strict=True):
        assert abs(obs_p - exp_p) < 1e-5


def test_bjd_times_length():
    r = sample_rv_phases(7.0, 5)
    assert len(r.bjd_times) == 5


def test_bjd_times_from_reference():
    ref = 2460000.0
    period = 10.0
    r = sample_rv_phases(period, 4, reference_bjd=ref)
    for bjd, phase in zip(r.bjd_times, r.phases, strict=True):
        assert abs(bjd - (ref + phase * period)) < 1e-4


def test_coverage_fraction_range():
    r = sample_rv_phases(10.0, 8)
    assert 0.0 <= r.phase_coverage <= 1.0


def test_max_gap_decreases_with_more_obs():
    r4 = sample_rv_phases(10.0, 4)
    r8 = sample_rv_phases(10.0, 8)
    assert r8.max_gap < r4.max_gap


def test_invalid_negative_period():
    r = sample_rv_phases(-1.0, 4)
    assert r.flag == "INVALID"
    assert len(r.phases) == 0


def test_invalid_zero_n_obs():
    r = sample_rv_phases(5.0, 0)
    assert r.flag == "INVALID"


def test_invalid_nan_period():
    r = sample_rv_phases(float("nan"), 4)
    assert r.flag == "INVALID"


def test_single_observation():
    r = sample_rv_phases(10.0, 1)
    assert r.flag == "OK"
    assert len(r.phases) == 1
    assert abs(r.phases[0] - 0.1) < 1e-6


def test_format_contains_keywords():
    r = sample_rv_phases(10.0, 6)
    text = format_rv_phases(r)
    assert "RV Phase Sampler" in text
    assert "Phase coverage" in text
    assert "OK" in text


def test_format_invalid():
    r = sample_rv_phases(-1.0, 4)
    text = format_rv_phases(r)
    assert "INVALID" in text
