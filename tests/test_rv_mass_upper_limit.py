"""Tests for Skills/rv_mass_upper_limit.py"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from rv_mass_upper_limit import compute_rv_mass_limit, format_rv_mass_limit


class TestRvMassUpperLimit:
    def test_basic_ok(self) -> None:
        r = compute_rv_mass_limit(5.0, 1.0, 1.0)
        assert r.flag == "OK"
        assert r.mass_upper_limit_mearth > 0.0

    def test_k_amplitude_positive(self) -> None:
        r = compute_rv_mass_limit(5.0, 1.0, 1.0)
        assert r.k_amplitude_ms > 0.0

    def test_better_precision_lower_limit(self) -> None:
        r1 = compute_rv_mass_limit(5.0, 1.0, 1.0)
        r2 = compute_rv_mass_limit(5.0, 1.0, 0.1)
        assert r2.mass_upper_limit_mearth < r1.mass_upper_limit_mearth

    def test_more_obs_lower_limit(self) -> None:
        r1 = compute_rv_mass_limit(5.0, 1.0, 1.0, n_obs=5)
        r2 = compute_rv_mass_limit(5.0, 1.0, 1.0, n_obs=50)
        assert r2.mass_upper_limit_mearth < r1.mass_upper_limit_mearth

    def test_longer_period_higher_mass_limit(self) -> None:
        r1 = compute_rv_mass_limit(1.0, 1.0, 1.0)
        r2 = compute_rv_mass_limit(100.0, 1.0, 1.0)
        assert r2.mass_upper_limit_mearth > r1.mass_upper_limit_mearth

    def test_invalid_period(self) -> None:
        r = compute_rv_mass_limit(0.0, 1.0, 1.0)
        assert "INVALID" in r.flag
        assert math.isnan(r.mass_upper_limit_mearth)

    def test_invalid_stellar_mass(self) -> None:
        r = compute_rv_mass_limit(5.0, 0.0, 1.0)
        assert "INVALID" in r.flag

    def test_invalid_rv_precision(self) -> None:
        r = compute_rv_mass_limit(5.0, 1.0, 0.0)
        assert "INVALID" in r.flag

    def test_invalid_eccentricity(self) -> None:
        r = compute_rv_mass_limit(5.0, 1.0, 1.0, eccentricity=1.0)
        assert r.flag == "INVALID_ECCENTRICITY"

    def test_mass_sini_stored(self) -> None:
        r = compute_rv_mass_limit(5.0, 1.0, 1.0)
        assert r.mass_sini_mearth > 0.0

    def test_eccentricity_lowers_limit(self) -> None:
        # Higher eccentricity amplifies K → lower mass detectable at same RV precision
        r0 = compute_rv_mass_limit(5.0, 1.0, 1.0, eccentricity=0.0)
        r5 = compute_rv_mass_limit(5.0, 1.0, 1.0, eccentricity=0.5)
        assert r5.mass_upper_limit_mearth < r0.mass_upper_limit_mearth

    def test_format_returns_string(self) -> None:
        r = compute_rv_mass_limit(5.0, 1.0, 1.0)
        s = format_rv_mass_limit(r)
        assert isinstance(s, str)
        assert "RV" in s
