"""Tests for Skills/stellar_convective_zone.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from stellar_convective_zone import ConvectiveZoneResult, estimate_convective_zone


class TestStellarConvectiveZone:
    def test_fully_convective_m_dwarf(self) -> None:
        r = estimate_convective_zone(3000.0)
        assert r.flag == "OK"
        assert r.convective_type == "FULLY_CONVECTIVE"
        assert r.rcz_over_rstar == 1.0

    def test_deep_cz_k_dwarf(self) -> None:
        r = estimate_convective_zone(4500.0)
        assert r.flag == "OK"
        assert r.convective_type == "DEEP_CZ"
        assert 0.22 < r.rcz_over_rstar < 1.0

    def test_shallow_cz_f_dwarf(self) -> None:
        r = estimate_convective_zone(6500.0)
        assert r.flag == "OK"
        assert r.convective_type == "SHALLOW_CZ"

    def test_radiative_a_star(self) -> None:
        r = estimate_convective_zone(9000.0)
        assert r.flag == "OK"
        assert r.convective_type == "RADIATIVE"
        assert r.rcz_over_rstar == 0.01

    def test_invalid_teff_zero(self) -> None:
        r = estimate_convective_zone(0.0)
        assert r.flag == "INVALID_TEFF"

    def test_invalid_teff_negative(self) -> None:
        r = estimate_convective_zone(-100.0)
        assert r.flag == "INVALID_TEFF"

    def test_invalid_teff_nan(self) -> None:
        r = estimate_convective_zone(float("nan"))
        assert r.flag == "INVALID_TEFF"

    def test_dynamo_active_solar(self) -> None:
        r = estimate_convective_zone(5778.0)
        assert r.dynamo_active

    def test_dynamo_inactive_a_star(self) -> None:
        r = estimate_convective_zone(9000.0)
        assert not r.dynamo_active

    def test_boundary_3900k(self) -> None:
        r = estimate_convective_zone(3900.0)
        assert r.flag == "OK"

    def test_boundary_6000k(self) -> None:
        r = estimate_convective_zone(6000.0)
        assert r.flag == "OK"

    def test_result_frozen(self) -> None:
        r = estimate_convective_zone(5778.0)
        assert isinstance(r, ConvectiveZoneResult)
        try:
            object.__setattr__(r, "flag", "x")
            raise AssertionError()
        except Exception:
            pass
