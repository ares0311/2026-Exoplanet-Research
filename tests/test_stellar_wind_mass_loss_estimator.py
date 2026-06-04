"""Tests for Skills/stellar_wind_mass_loss_estimator.py."""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from stellar_wind_mass_loss_estimator import estimate_mass_loss_rate, format_wind_result


class TestEstimateMassLossRate:
    def test_solar_rotation(self) -> None:
        r = estimate_mass_loss_rate(prot_days=25.0)
        assert r.flag == "OK"
        assert r.mass_loss_msun_per_yr > 0
        assert r.method == "ROTATION"

    def test_solar_relative_near_one(self) -> None:
        r = estimate_mass_loss_rate(prot_days=25.0)
        # Solar rotation ~25 d → relative to sun ≈ 1
        assert 0.1 < r.mass_loss_relative_to_sun < 10.0

    def test_fast_rotator_more_loss(self) -> None:
        r_fast = estimate_mass_loss_rate(prot_days=5.0)
        r_slow = estimate_mass_loss_rate(prot_days=30.0)
        assert r_fast.mass_loss_msun_per_yr > r_slow.mass_loss_msun_per_yr

    def test_xray_method_preferred(self) -> None:
        # Low X-ray (valid Wood 2005 range) — both supplied but xray preferred
        r = estimate_mass_loss_rate(prot_days=25.0, lx_ergs=1e26)
        assert r.method == "XRAY"

    def test_xray_valid_range_reliable(self) -> None:
        # log(Lx/Lsun) < -3.8 → reliable
        r = estimate_mass_loss_rate(lx_ergs=1e26)
        assert r.reliable

    def test_xray_saturated_not_reliable(self) -> None:
        r = estimate_mass_loss_rate(lx_ergs=1e30)
        assert r.method == "XRAY"
        assert not r.reliable

    def test_default_method_fallback(self) -> None:
        r = estimate_mass_loss_rate()
        assert r.method == "DEFAULT"
        assert r.mass_loss_msun_per_yr > 0

    def test_invalid_mass(self) -> None:
        r = estimate_mass_loss_rate(mass_msun=0.0)
        assert r.flag == "INVALID_MASS"

    def test_higher_mass_more_loss(self) -> None:
        r1 = estimate_mass_loss_rate(prot_days=25.0, mass_msun=1.0)
        r2 = estimate_mass_loss_rate(prot_days=25.0, mass_msun=2.0)
        assert r2.mass_loss_msun_per_yr > r1.mass_loss_msun_per_yr

    def test_reliable_in_prot_range(self) -> None:
        r = estimate_mass_loss_rate(prot_days=20.0)
        assert r.reliable

    def test_format_output(self) -> None:
        r = estimate_mass_loss_rate(prot_days=25.0)
        s = format_wind_result(r)
        assert "|" in s
        assert "mass" in s.lower() or "solar" in s.lower()

    def test_result_finite(self) -> None:
        r = estimate_mass_loss_rate(prot_days=15.0)
        assert math.isfinite(r.mass_loss_msun_per_yr)
        assert math.isfinite(r.mass_loss_relative_to_sun)
