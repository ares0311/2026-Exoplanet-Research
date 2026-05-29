"""Tests for Skills/orbital_period_from_density.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from orbital_period_from_density import OrbitalPeriodResult, compute_orbital_distance, format_orbital_period


class TestOrbitalPeriodResult:
    def test_dataclass_fields(self):
        r = OrbitalPeriodResult(a_rstar=5.0, a_au=0.05, period_days=10.0, flag="OK")
        assert r.a_rstar == 5.0
        assert r.flag == "OK"

    def test_frozen(self):
        r = OrbitalPeriodResult(a_rstar=5.0, a_au=0.05, period_days=10.0, flag="OK")
        try:
            r.a_rstar = 0
            assert False
        except Exception:
            pass


class TestComputeOrbitalDistance:
    def test_solar_density_1d_period(self):
        # Solar density ~1.41 g/cm^3, P=1 day
        r = compute_orbital_distance(1.41, 1.0, rstar_rsun=1.0)
        assert r.flag == "OK"
        assert r.a_rstar > 0
        assert r.a_au > 0

    def test_a_rstar_positive(self):
        r = compute_orbital_distance(1.0, 10.0)
        assert r.a_rstar > 0

    def test_a_au_positive(self):
        r = compute_orbital_distance(1.0, 10.0)
        assert r.a_au > 0

    def test_period_stored(self):
        r = compute_orbital_distance(1.0, 7.3)
        assert r.period_days == 7.3

    def test_zero_density_error(self):
        r = compute_orbital_distance(0.0, 10.0)
        assert r.flag == "ERROR"

    def test_zero_period_error(self):
        r = compute_orbital_distance(1.0, 0.0)
        assert r.flag == "ERROR"

    def test_zero_rstar_error(self):
        r = compute_orbital_distance(1.0, 10.0, rstar_rsun=0.0)
        assert r.flag == "ERROR"

    def test_long_period_flag(self):
        r = compute_orbital_distance(1.0, 400.0)
        assert r.flag == "LONG_PERIOD"

    def test_a_rstar_scales_with_period(self):
        # a/R* ∝ P^(2/3)
        r1 = compute_orbital_distance(1.0, 10.0)
        r2 = compute_orbital_distance(1.0, 80.0)
        ratio = r2.a_rstar / r1.a_rstar
        expected = (80.0 / 10.0) ** (2.0 / 3.0)
        assert abs(ratio - expected) < 0.01

    def test_a_rstar_scales_with_density(self):
        # a/R* ∝ rho^(1/3)
        r1 = compute_orbital_distance(1.0, 10.0)
        r2 = compute_orbital_distance(8.0, 10.0)
        ratio = r2.a_rstar / r1.a_rstar
        assert abs(ratio - 2.0) < 0.01  # (8/1)^(1/3) = 2

    def test_a_au_scales_with_rstar(self):
        r1 = compute_orbital_distance(1.0, 10.0, rstar_rsun=1.0)
        r2 = compute_orbital_distance(1.0, 10.0, rstar_rsun=2.0)
        assert abs(r2.a_au / r1.a_au - 2.0) < 0.01


class TestFormatOrbitalPeriod:
    def test_returns_string(self):
        r = compute_orbital_distance(1.0, 10.0)
        s = format_orbital_period(r)
        assert isinstance(s, str)

    def test_contains_au(self):
        r = compute_orbital_distance(1.0, 10.0)
        s = format_orbital_period(r)
        assert "AU" in s or "au" in s.lower()
