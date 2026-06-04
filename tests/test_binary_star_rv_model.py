"""Tests for Skills/binary_star_rv_model.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from binary_star_rv_model import compute_binary_rv_model, format_binary_rv_result


class TestComputeBinaryRvModel:
    def test_ok_flag_sb1(self) -> None:
        r = compute_binary_rv_model(50000.0, 1.5)
        assert r.flag == "OK"

    def test_ok_flag_sb2(self) -> None:
        r = compute_binary_rv_model(50000.0, 1.5, mass_ratio=2.0)
        assert r.flag == "OK"

    def test_n_points_correct(self) -> None:
        r = compute_binary_rv_model(50000.0, 1.5, n_points=50)
        assert len(r.phases) == 50
        assert len(r.rv_primary_ms) == 50

    def test_sb1_no_secondary(self) -> None:
        r = compute_binary_rv_model(50000.0, 1.5)
        assert r.rv_secondary_ms is None
        assert r.k2_ms is None

    def test_sb2_has_secondary(self) -> None:
        r = compute_binary_rv_model(50000.0, 1.5, mass_ratio=1.0)
        assert r.rv_secondary_ms is not None
        assert len(r.rv_secondary_ms) == len(r.rv_primary_ms)

    def test_sb2_k2_equals_k1_for_equal_mass(self) -> None:
        r = compute_binary_rv_model(50000.0, 1.5, mass_ratio=1.0)
        assert r.k2_ms is not None
        assert abs(r.k2_ms - r.k1_ms) < 1e-6

    def test_sb2_primary_secondary_antiphase(self) -> None:
        r = compute_binary_rv_model(50000.0, 1.5, gamma_ms=0.0, mass_ratio=1.0, n_points=100)
        rv1_sum = sum(r.rv_primary_ms)
        rv2_sum = sum(r.rv_secondary_ms)  # type: ignore[arg-type]
        assert abs(rv1_sum + rv2_sum) < 1.0

    def test_circular_rv_range(self) -> None:
        r = compute_binary_rv_model(50000.0, 1.5, eccentricity=0.0, gamma_ms=0.0)
        assert abs(r.rv_max_primary_ms - 50000.0) < 1.0
        assert abs(r.rv_min_primary_ms + 50000.0) < 1.0

    def test_invalid_period(self) -> None:
        r = compute_binary_rv_model(50000.0, 0.0)
        assert r.flag == "INVALID_PERIOD"

    def test_invalid_eccentricity(self) -> None:
        r = compute_binary_rv_model(50000.0, 1.5, eccentricity=1.0)
        assert r.flag == "INVALID_ECCENTRICITY"

    def test_invalid_k1(self) -> None:
        r = compute_binary_rv_model(0.0, 1.5)
        assert r.flag == "INVALID_K1"

    def test_result_frozen(self) -> None:
        r = compute_binary_rv_model(50000.0, 1.5)
        try:
            r.k1_ms = 0.0  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except (AttributeError, TypeError):
            pass

    def test_format_returns_string(self) -> None:
        r = compute_binary_rv_model(50000.0, 1.5)
        s = format_binary_rv_result(r)
        assert isinstance(s, str)
        assert r.flag in s
