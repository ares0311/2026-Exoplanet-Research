"""Tests for Skills/multi_planet_ttv_amplitude.py."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from multi_planet_ttv_amplitude import (
    TTVAmplitudeResult,
    compute_ttv_amplitude,
    format_ttv_amplitude_result,
)


class TestComputeTTVAmplitude:
    def test_returns_result_type(self):
        r = compute_ttv_amplitude(inner_period_days=10.0, outer_period_days=20.0,
                                   perturber_mass_mearth=10.0)
        assert isinstance(r, TTVAmplitudeResult)

    def test_flag_ok(self):
        r = compute_ttv_amplitude(inner_period_days=10.0, outer_period_days=20.0,
                                   perturber_mass_mearth=10.0)
        assert r.flag == "OK"

    def test_ttv_amplitude_positive(self):
        r = compute_ttv_amplitude(inner_period_days=10.0, outer_period_days=20.0,
                                   perturber_mass_mearth=10.0)
        assert r.ttv_amplitude_minutes > 0.0

    def test_libration_period_positive(self):
        r = compute_ttv_amplitude(inner_period_days=10.0, outer_period_days=20.0,
                                   perturber_mass_mearth=10.0)
        assert r.libration_period_days > 0.0

    def test_ttv_class_string(self):
        r = compute_ttv_amplitude(inner_period_days=10.0, outer_period_days=20.0,
                                   perturber_mass_mearth=10.0)
        assert r.ttv_class in ("STRONG", "MODERATE", "WEAK")

    def test_more_massive_perturber_larger_ttv(self):
        r1 = compute_ttv_amplitude(inner_period_days=10.0, outer_period_days=20.0,
                                    perturber_mass_mearth=100.0)
        r2 = compute_ttv_amplitude(inner_period_days=10.0, outer_period_days=20.0,
                                    perturber_mass_mearth=1.0)
        assert r1.ttv_amplitude_minutes > r2.ttv_amplitude_minutes

    def test_near_resonance_larger_ttv(self):
        # Near 2:1 resonance
        r_near = compute_ttv_amplitude(inner_period_days=10.0, outer_period_days=20.5,
                                        perturber_mass_mearth=10.0)
        # Far from resonance
        r_far = compute_ttv_amplitude(inner_period_days=10.0, outer_period_days=50.0,
                                       perturber_mass_mearth=10.0)
        assert r_near.ttv_amplitude_minutes > r_far.ttv_amplitude_minutes

    def test_invalid_inner_period(self):
        r = compute_ttv_amplitude(inner_period_days=0.0, outer_period_days=20.0,
                                   perturber_mass_mearth=10.0)
        assert r.flag != "OK"
        assert math.isnan(r.ttv_amplitude_minutes)

    def test_invalid_outer_period(self):
        r = compute_ttv_amplitude(inner_period_days=10.0, outer_period_days=0.0,
                                   perturber_mass_mearth=10.0)
        assert r.flag != "OK"

    def test_invalid_perturber_mass(self):
        r = compute_ttv_amplitude(inner_period_days=10.0, outer_period_days=20.0,
                                   perturber_mass_mearth=0.0)
        assert r.flag != "OK"

    def test_outer_must_be_larger_than_inner(self):
        r = compute_ttv_amplitude(inner_period_days=20.0, outer_period_days=10.0,
                                   perturber_mass_mearth=10.0)
        assert r.flag != "OK"

    def test_resonance_order_positive(self):
        r = compute_ttv_amplitude(inner_period_days=10.0, outer_period_days=20.0,
                                   perturber_mass_mearth=10.0)
        assert r.resonance_order >= 1

    def test_frozen_dataclass(self):
        r = compute_ttv_amplitude(inner_period_days=10.0, outer_period_days=20.0,
                                   perturber_mass_mearth=10.0)
        try:
            r.ttv_amplitude_minutes = 0.0  # type: ignore[misc]
            raise AssertionError("frozen dataclass should not allow setattr")
        except (AttributeError, TypeError):
            pass


class TestFormatTTVAmplitudeResult:
    def test_ok_returns_table(self):
        r = compute_ttv_amplitude(inner_period_days=10.0, outer_period_days=20.0,
                                   perturber_mass_mearth=10.0)
        out = format_ttv_amplitude_result(r)
        assert "TTV amplitude" in out
        assert "|" in out

    def test_invalid_returns_flag(self):
        r = compute_ttv_amplitude(inner_period_days=0.0, outer_period_days=20.0,
                                   perturber_mass_mearth=10.0)
        out = format_ttv_amplitude_result(r)
        assert "flag=" in out
