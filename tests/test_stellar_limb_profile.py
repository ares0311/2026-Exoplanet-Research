"""Tests for Skills/stellar_limb_profile.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from stellar_limb_profile import LimbProfileResult, compute_limb_profile, format_limb_profile


class TestLimbProfileResult:
    def test_dataclass_fields(self):
        r = LimbProfileResult(
            mu_values=(0.0, 1.0), intensity_values=(0.3, 1.0),
            u1=0.4, u2=0.3, flag="OK"
        )
        assert r.u1 == 0.4
        assert r.flag == "OK"

    def test_frozen(self):
        r = LimbProfileResult(
            mu_values=(0.0, 1.0), intensity_values=(0.3, 1.0),
            u1=0.4, u2=0.3, flag="OK"
        )
        try:
            r.u1 = 0
            assert False
        except Exception:
            pass


class TestComputeLimbProfile:
    def test_n_points_length(self):
        r = compute_limb_profile(n_points=10)
        assert len(r.mu_values) == 10
        assert len(r.intensity_values) == 10

    def test_mu_range(self):
        r = compute_limb_profile(n_points=11)
        assert r.mu_values[0] == 0.0
        assert abs(r.mu_values[-1] - 1.0) < 0.01

    def test_intensity_at_mu_1(self):
        # At mu=1: I = 1 - u1*(1-1) - u2*(1-1)^2 = 1.0
        r = compute_limb_profile(u1=0.4, u2=0.3, n_points=2)
        assert abs(r.intensity_values[-1] - 1.0) < 1e-9

    def test_intensity_at_mu_0(self):
        # At mu=0: I = 1 - u1 - u2
        r = compute_limb_profile(u1=0.4, u2=0.3, n_points=2)
        expected = 1.0 - 0.4 - 0.3
        assert abs(r.intensity_values[0] - expected) < 1e-9

    def test_monotonic_with_valid_coeffs(self):
        r = compute_limb_profile(u1=0.4, u2=0.3, n_points=10)
        intensities = list(r.intensity_values)
        assert all(intensities[i] <= intensities[i + 1] for i in range(len(intensities) - 1))

    def test_flag_ok(self):
        r = compute_limb_profile(u1=0.4, u2=0.3, n_points=10)
        assert r.flag == "OK"

    def test_unphysical_flag(self):
        # Extreme coefficients to make intensity < 0 at mu=0
        r = compute_limb_profile(u1=0.9, u2=0.9, n_points=5)
        # 1 - 0.9 - 0.9 = -0.8 < 0
        assert r.flag == "UNPHYSICAL"

    def test_u1_u2_stored(self):
        r = compute_limb_profile(u1=0.5, u2=0.2, n_points=5)
        assert r.u1 == 0.5
        assert r.u2 == 0.2

    def test_min_n_points_is_2(self):
        r = compute_limb_profile(n_points=1)
        assert len(r.mu_values) == 2

    def test_tuples(self):
        r = compute_limb_profile(n_points=5)
        assert isinstance(r.mu_values, tuple)
        assert isinstance(r.intensity_values, tuple)


class TestFormatLimbProfile:
    def test_returns_string(self):
        r = compute_limb_profile()
        s = format_limb_profile(r)
        assert isinstance(s, str)

    def test_contains_u1_u2(self):
        r = compute_limb_profile(u1=0.4, u2=0.3)
        s = format_limb_profile(r)
        assert "0.4" in s
        assert "0.3" in s
