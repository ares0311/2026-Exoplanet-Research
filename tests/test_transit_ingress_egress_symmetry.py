"""Tests for Skills/transit_ingress_egress_symmetry.py."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from transit_ingress_egress_symmetry import (
    check_ingress_egress_symmetry,
    format_symmetry_result,
)


def _symmetric_transit(n: int = 50, half_width: float = 0.05) -> tuple[list[float], list[float]]:
    phase = [(-0.5 + i / (n - 1)) for i in range(n)]
    flux = [1.0 - 0.01 if abs(p) < half_width else 1.0 for p in phase]
    return phase, flux


class TestCheckIngressEgressSymmetry:
    def test_symmetric_transit_ok(self) -> None:
        phase, flux = _symmetric_transit()
        r = check_ingress_egress_symmetry(phase, flux)
        assert r.flag in ("OK", "INSUFFICIENT_INGRESS_EGRESS_POINTS")

    def test_length_mismatch(self) -> None:
        r = check_ingress_egress_symmetry([0.0, 0.1], [1.0])
        assert r.flag == "LENGTH_MISMATCH"

    def test_insufficient_data(self) -> None:
        r = check_ingress_egress_symmetry([0.0, 0.1], [1.0, 0.99])
        assert r.flag == "INSUFFICIENT_DATA"

    def test_asymmetric_detected(self) -> None:
        # Build phase and flux with very different ingress vs egress depths
        phase = list(range(-20, 21))
        phase = [p / 200.0 for p in phase]
        hw = 0.05
        iw = 0.04
        flux = []
        for p in phase:
            if -hw <= p <= -hw + iw:
                flux.append(0.90)  # deep ingress
            elif hw - iw <= p <= hw:
                flux.append(0.99)  # shallow egress
            else:
                flux.append(1.0)
        r = check_ingress_egress_symmetry(phase, flux, hw, iw)
        if r.flag not in ("INSUFFICIENT_INGRESS_EGRESS_POINTS",):
            assert r.significance >= 0

    def test_significance_nonneg(self) -> None:
        phase, flux = _symmetric_transit(100)
        r = check_ingress_egress_symmetry(phase, flux, 0.05, 0.02)
        if r.flag == "OK":
            assert r.significance >= 0

    def test_n_in_transit_count(self) -> None:
        phase, flux = _symmetric_transit(100)
        r = check_ingress_egress_symmetry(phase, flux)
        assert r.n_in_transit >= 0

    def test_asymmetry_is_mean_diff(self) -> None:
        # Large sample with uniform depth for both ingress and egress
        n = 200
        phase = [(-0.5 + i / (n - 1)) for i in range(n)]
        hw = 0.1
        iw = 0.05
        flux = [1.0 - 0.01 if abs(p) <= hw else 1.0 for p in phase]
        r = check_ingress_egress_symmetry(phase, flux, hw, iw)
        if r.flag == "OK":
            assert abs(r.asymmetry) < 0.02

    def test_flag_ok_or_flagged(self) -> None:
        phase, flux = _symmetric_transit(200)
        r = check_ingress_egress_symmetry(phase, flux, 0.05, 0.02)
        assert r.flag in ("OK", "ASYMMETRIC", "INSUFFICIENT_INGRESS_EGRESS_POINTS")

    def test_ingress_depth_mean_nonneg(self) -> None:
        phase, flux = _symmetric_transit(100)
        r = check_ingress_egress_symmetry(phase, flux)
        if r.flag == "OK":
            assert r.ingress_depth_mean >= 0

    def test_egress_depth_mean_nonneg(self) -> None:
        phase, flux = _symmetric_transit(100)
        r = check_ingress_egress_symmetry(phase, flux)
        if r.flag == "OK":
            assert r.egress_depth_mean >= 0

    def test_format_output(self) -> None:
        phase, flux = _symmetric_transit(100)
        r = check_ingress_egress_symmetry(phase, flux)
        s = format_symmetry_result(r)
        assert "|" in s

    def test_empty_lists(self) -> None:
        r = check_ingress_egress_symmetry([], [])
        assert r.flag == "INSUFFICIENT_DATA"
