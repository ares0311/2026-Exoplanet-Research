"""Tests for Skills/orbital_decay_timescale.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from orbital_decay_timescale import compute_orbital_decay_timescale, format_orbital_decay_result


class TestComputeOrbitalDecayTimescale:
    def test_ok_flag(self) -> None:
        r = compute_orbital_decay_timescale(3.0, 1.0)
        assert r.flag == "OK"

    def test_timescale_positive(self) -> None:
        r = compute_orbital_decay_timescale(3.0, 1.0)
        assert r.decay_timescale_gyr > 0.0

    def test_semi_major_axis_positive(self) -> None:
        r = compute_orbital_decay_timescale(3.0, 1.0)
        assert r.semi_major_axis_au > 0.0

    def test_dp_dt_negative(self) -> None:
        r = compute_orbital_decay_timescale(3.0, 1.0)
        assert r.dp_dt_s_per_s < 0.0

    def test_closer_orbit_faster_decay(self) -> None:
        r_close = compute_orbital_decay_timescale(1.0, 1.0)
        r_far = compute_orbital_decay_timescale(10.0, 1.0)
        assert r_close.decay_timescale_gyr < r_far.decay_timescale_gyr

    def test_larger_q_slower_decay(self) -> None:
        r_low = compute_orbital_decay_timescale(3.0, 1.0, tidal_quality_factor=1e5)
        r_high = compute_orbital_decay_timescale(3.0, 1.0, tidal_quality_factor=1e8)
        assert r_high.decay_timescale_gyr > r_low.decay_timescale_gyr

    def test_decay_class_set(self) -> None:
        r = compute_orbital_decay_timescale(3.0, 1.0)
        assert r.decay_class in ("STABLE", "MARGINAL", "DECAYING", "RAPID_DECAY")

    def test_hot_jupiter_class(self) -> None:
        r = compute_orbital_decay_timescale(1.0, 5.0, tidal_quality_factor=1e5)
        assert r.decay_class in ("DECAYING", "RAPID_DECAY", "MARGINAL")

    def test_invalid_period(self) -> None:
        r = compute_orbital_decay_timescale(0.0, 1.0)
        assert r.flag == "INVALID_PERIOD"

    def test_invalid_mass(self) -> None:
        r = compute_orbital_decay_timescale(3.0, 0.0)
        assert r.flag == "INVALID_PLANET_MASS"

    def test_invalid_stellar_mass(self) -> None:
        r = compute_orbital_decay_timescale(3.0, 1.0, stellar_mass_msun=0.0)
        assert r.flag == "INVALID_STELLAR_MASS"

    def test_result_frozen(self) -> None:
        r = compute_orbital_decay_timescale(3.0, 1.0)
        try:
            r.decay_timescale_gyr = 0.0  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except (AttributeError, TypeError):
            pass

    def test_format_returns_string(self) -> None:
        r = compute_orbital_decay_timescale(3.0, 1.0)
        s = format_orbital_decay_result(r)
        assert isinstance(s, str)
        assert r.flag in s
