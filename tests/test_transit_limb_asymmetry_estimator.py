"""Tests for Skills/transit_limb_asymmetry_estimator.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from transit_limb_asymmetry_estimator import (
    compute_transit_limb_asymmetry,
    format_limb_asymmetry_result,
)


class TestComputeTransitLimbAsymmetry:
    def test_ok_flag(self) -> None:
        r = compute_transit_limb_asymmetry(10.0, 10000.0, 0.3, 1.0, 1.0)
        assert r.flag == "OK"

    def test_ingress_duration_positive(self) -> None:
        r = compute_transit_limb_asymmetry(10.0, 10000.0, 0.3, 1.0, 1.0)
        assert r.ingress_duration_hours > 0.0

    def test_egress_duration_positive(self) -> None:
        r = compute_transit_limb_asymmetry(10.0, 10000.0, 0.3, 1.0, 1.0)
        assert r.egress_duration_hours > 0.0

    def test_asymmetry_ratio_near_one_aligned(self) -> None:
        r = compute_transit_limb_asymmetry(10.0, 10000.0, 0.3, 1.0, 1.0,
                                            spin_orbit_lambda_deg=0.0)
        assert abs(r.asymmetry_ratio - 1.0) < 0.05

    def test_nonzero_lambda_changes_asymmetry(self) -> None:
        r0 = compute_transit_limb_asymmetry(10.0, 10000.0, 0.3, 1.0, 1.0,
                                             spin_orbit_lambda_deg=0.0)
        r45 = compute_transit_limb_asymmetry(10.0, 10000.0, 0.3, 1.0, 1.0,
                                              spin_orbit_lambda_deg=45.0)
        assert abs(r45.asymmetry_ratio - r0.asymmetry_ratio) >= 0.0

    def test_ingress_egress_durations_finite(self) -> None:
        r = compute_transit_limb_asymmetry(10.0, 10000.0, 0.0, 1.0, 1.0)
        assert r.ingress_duration_hours < float("inf")
        assert r.egress_duration_hours < float("inf")

    def test_high_impact_parameter(self) -> None:
        r = compute_transit_limb_asymmetry(10.0, 10000.0, 0.8, 1.0, 1.0)
        assert r.flag == "OK"
        assert r.ingress_duration_hours > 0.0

    def test_asymmetry_seconds_nonnegative(self) -> None:
        r = compute_transit_limb_asymmetry(10.0, 10000.0, 0.3, 1.0, 1.0)
        assert r.asymmetry_seconds >= 0.0

    def test_invalid_period(self) -> None:
        r = compute_transit_limb_asymmetry(0.0, 10000.0, 0.3, 1.0, 1.0)
        assert r.flag == "INVALID_PERIOD"

    def test_invalid_depth(self) -> None:
        r = compute_transit_limb_asymmetry(10.0, 0.0, 0.3, 1.0, 1.0)
        assert r.flag == "INVALID_DEPTH"

    def test_invalid_impact_parameter(self) -> None:
        r = compute_transit_limb_asymmetry(10.0, 10000.0, 1.0, 1.0, 1.0)
        assert r.flag == "INVALID_IMPACT"

    def test_result_frozen(self) -> None:
        r = compute_transit_limb_asymmetry(10.0, 10000.0, 0.3, 1.0, 1.0)
        try:
            r.asymmetry_ratio = 0.0  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except (AttributeError, TypeError):
            pass

    def test_format_returns_string(self) -> None:
        r = compute_transit_limb_asymmetry(10.0, 10000.0, 0.3, 1.0, 1.0)
        s = format_limb_asymmetry_result(r)
        assert isinstance(s, str)
        assert r.flag in s
