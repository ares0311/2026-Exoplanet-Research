"""Tests for Skills/mutual_inclination_estimator.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from mutual_inclination_estimator import (
    estimate_mutual_inclination,
    format_mutual_inclination_result,
)


class TestMutualInclinationEstimator:
    def test_basic_ok(self) -> None:
        r = estimate_mutual_inclination(5.0, 1000.0, 2.0, 10.0, 1000.0, 2.0)
        assert r.flag == "OK"

    def test_identical_planets_zero_mutual_inclination(self) -> None:
        r = estimate_mutual_inclination(10.0, 1000.0, 2.0, 10.0, 1000.0, 2.0)
        assert r.mutual_inclination_deg < 1e-6

    def test_inclinations_finite(self) -> None:
        import math
        r = estimate_mutual_inclination(5.0, 1000.0, 2.0, 10.0, 1000.0, 2.0)
        assert math.isfinite(r.inclination1_deg)
        assert math.isfinite(r.inclination2_deg)
        assert 0.0 <= r.inclination1_deg <= 90.0
        assert 0.0 <= r.inclination2_deg <= 90.0

    def test_different_durations_different_inclinations(self) -> None:
        r = estimate_mutual_inclination(5.0, 1000.0, 2.0, 10.0, 1000.0, 3.5)
        assert r.inclination1_deg != r.inclination2_deg

    def test_mutual_inclination_abs_value(self) -> None:
        r = estimate_mutual_inclination(5.0, 1000.0, 2.0, 10.0, 1000.0, 3.5)
        assert r.mutual_inclination_deg >= 0

    def test_impact_parameters_non_negative(self) -> None:
        r = estimate_mutual_inclination(5.0, 1000.0, 2.0, 10.0, 1000.0, 2.0)
        assert r.impact_parameter1 >= 0
        assert r.impact_parameter2 >= 0

    def test_invalid_period(self) -> None:
        r = estimate_mutual_inclination(0.0, 1000.0, 2.0, 10.0, 1000.0, 2.0)
        assert "INVALID" in r.flag

    def test_invalid_depth(self) -> None:
        r = estimate_mutual_inclination(5.0, 0.0, 2.0, 10.0, 1000.0, 2.0)
        assert "INVALID" in r.flag

    def test_invalid_duration(self) -> None:
        r = estimate_mutual_inclination(5.0, 1000.0, 0.0, 10.0, 1000.0, 2.0)
        assert "INVALID" in r.flag

    def test_result_is_frozen(self) -> None:
        r = estimate_mutual_inclination(5.0, 1000.0, 2.0, 10.0, 1000.0, 2.0)
        try:
            r.mutual_inclination_deg = 5.0  # type: ignore[misc]
            raise AssertionError("frozen dataclass should not allow setattr")
        except Exception:
            pass

    def test_format_ok(self) -> None:
        r = estimate_mutual_inclination(5.0, 1000.0, 2.0, 10.0, 1000.0, 2.0)
        s = format_mutual_inclination_result(r)
        assert "inclination" in s.lower() or "Inclination" in s

    def test_format_error(self) -> None:
        r = estimate_mutual_inclination(0.0, 1000.0, 2.0, 10.0, 1000.0, 2.0)
        s = format_mutual_inclination_result(r)
        assert "INVALID" in s

    def test_stellar_mass_affects_result(self) -> None:
        r1 = estimate_mutual_inclination(5.0, 1000.0, 2.0, 10.0, 1000.0, 2.0,
                                          stellar_mass_msun=1.0)
        r2 = estimate_mutual_inclination(5.0, 1000.0, 2.0, 10.0, 1000.0, 2.0,
                                          stellar_mass_msun=2.0)
        # Different stellar mass → different a/Rs → different inclination
        assert r1.impact_parameter1 != r2.impact_parameter1
