"""Tests for Skills/exomoon_hill_sphere_checker.py."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from exomoon_hill_sphere_checker import (
    ExomoonStabilityResult,
    check_exomoon_stability,
    format_exomoon_stability_result,
)


class TestCheckExomoonStability:
    def test_returns_result_type(self):
        r = check_exomoon_stability(1.0, 1.0, 5.0)
        assert isinstance(r, ExomoonStabilityResult)

    def test_flag_ok(self):
        r = check_exomoon_stability(1.0, 1.0, 5.0)
        assert r.flag == "OK"

    def test_hill_radius_positive(self):
        r = check_exomoon_stability(1.0, 1.0, 5.0)
        assert r.hill_radius_au > 0.0

    def test_stable_orbit_limit_less_than_hill(self):
        r = check_exomoon_stability(1.0, 1.0, 5.0)
        assert r.stable_orbit_limit_au < r.hill_radius_au

    def test_roche_limit_positive(self):
        r = check_exomoon_stability(1.0, 1.0, 5.0)
        assert r.roche_limit_rjup > 0.0

    def test_max_moon_period_days_positive(self):
        r = check_exomoon_stability(1.0, 1.0, 5.0)
        assert r.max_moon_period_days > 0.0

    def test_stability_class_string(self):
        r = check_exomoon_stability(1.0, 1.0, 5.0)
        assert r.stability_class in ("WIDE_STABLE", "MARGINAL", "ROCHE_LIMITED")

    def test_close_orbit_more_constrained(self):
        r_close = check_exomoon_stability(1.0, 1.0, 0.5)
        r_far = check_exomoon_stability(1.0, 1.0, 5.0)
        assert r_close.hill_radius_au < r_far.hill_radius_au

    def test_retrograde_larger_stable_region(self):
        r_pro = check_exomoon_stability(1.0, 1.0, 5.0, prograde=True)
        r_ret = check_exomoon_stability(1.0, 1.0, 5.0, prograde=False)
        assert r_ret.stable_orbit_limit_au > r_pro.stable_orbit_limit_au

    def test_invalid_stellar_mass(self):
        r = check_exomoon_stability(0.0, 1.0, 5.0)
        assert r.flag != "OK"
        assert math.isnan(r.hill_radius_au)

    def test_invalid_planet_mass(self):
        r = check_exomoon_stability(1.0, 0.0, 5.0)
        assert r.flag != "OK"

    def test_invalid_distance(self):
        r = check_exomoon_stability(1.0, 1.0, 0.0)
        assert r.flag != "OK"

    def test_frozen_dataclass(self):
        r = check_exomoon_stability(1.0, 1.0, 5.0)
        try:
            r.hill_radius_au = 0.0  # type: ignore[misc]
            raise AssertionError("frozen dataclass should not allow setattr")
        except (AttributeError, TypeError):
            pass


class TestFormatExomoonStabilityResult:
    def test_ok_returns_table(self):
        r = check_exomoon_stability(1.0, 1.0, 5.0)
        out = format_exomoon_stability_result(r)
        assert "Hill radius" in out
        assert "|" in out

    def test_invalid_returns_flag(self):
        r = check_exomoon_stability(0.0, 1.0, 5.0)
        out = format_exomoon_stability_result(r)
        assert "flag=" in out
