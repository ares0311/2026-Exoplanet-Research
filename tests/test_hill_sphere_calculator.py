"""Tests for Skills/hill_sphere_calculator.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from hill_sphere_calculator import HillSphereResult, compute_hill_sphere, format_hill_sphere_result


class TestHillSphereResult:
    def test_dataclass_fields(self):
        r = HillSphereResult(r_hill_au=0.01, r_hill_rp=100.0, stability_flag="stable", flag="OK")
        assert r.r_hill_au == 0.01
        assert r.r_hill_rp == 100.0

    def test_frozen(self):
        r = HillSphereResult(r_hill_au=0.01, r_hill_rp=100.0, stability_flag="stable")
        try:
            r.r_hill_au = 0
            assert False, "Should be frozen"
        except Exception:
            pass


class TestComputeHillSphere:
    def test_earth_hill_sphere(self):
        # Earth at 1 AU: Hill sphere ~0.010 AU
        r = compute_hill_sphere(1.0, 1.0, m_star_msun=1.0, r_planet_rearth=1.0)
        assert abs(r.r_hill_au - 0.01) < 0.002
        assert r.flag == "OK"

    def test_stable_flag(self):
        r = compute_hill_sphere(1.0, 1.0)
        assert r.stability_flag == "stable"

    def test_unstable_flag(self):
        # Very close-in tiny planet → small Hill sphere
        r = compute_hill_sphere(0.001, 0.001, m_star_msun=1.0)
        assert r.stability_flag == "unstable"

    def test_zero_a_error(self):
        r = compute_hill_sphere(0.0, 1.0)
        assert r.flag == "ERROR"

    def test_zero_mass_error(self):
        r = compute_hill_sphere(1.0, 0.0)
        assert r.flag == "ERROR"

    def test_hill_scales_with_a(self):
        r1 = compute_hill_sphere(1.0, 1.0)
        r2 = compute_hill_sphere(2.0, 1.0)
        ratio = r2.r_hill_au / r1.r_hill_au
        assert abs(ratio - 2.0) < 0.01

    def test_hill_scales_with_mass_cbrt(self):
        r1 = compute_hill_sphere(1.0, 1.0)
        r2 = compute_hill_sphere(1.0, 8.0)
        ratio = r2.r_hill_au / r1.r_hill_au
        assert abs(ratio - 2.0) < 0.01  # (8/1)^(1/3) = 2

    def test_r_hill_rp_positive(self):
        r = compute_hill_sphere(1.0, 1.0, r_planet_rearth=2.0)
        assert r.r_hill_rp > 0

    def test_negative_r_planet_error(self):
        r = compute_hill_sphere(1.0, 1.0, r_planet_rearth=0.0)
        assert r.flag == "ERROR"

    def test_large_planet_hill(self):
        # Jupiter-mass planet at 5.2 AU — Hill sphere ~0.35 AU
        r = compute_hill_sphere(5.2, 318.0, m_star_msun=1.0, r_planet_rearth=11.2)
        assert r.r_hill_au > 0.1


class TestFormatHillSphere:
    def test_returns_string(self):
        r = compute_hill_sphere(1.0, 1.0)
        s = format_hill_sphere_result(r)
        assert isinstance(s, str)

    def test_contains_stability(self):
        r = compute_hill_sphere(1.0, 1.0)
        s = format_hill_sphere_result(r)
        assert r.stability_flag in s
