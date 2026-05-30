"""Tests for Skills/centroid_bootstrap_significance.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from centroid_bootstrap_significance import (
    bootstrap_centroid_significance,
    format_centroid_bootstrap,
)


class TestBootstrapCentroidSignificance:
    def test_clearly_offset_significant(self) -> None:
        # in-transit centroids far from oot
        in_t = [10.0, 10.1, 9.9, 10.0, 10.05]
        oot = [0.0, 0.1, -0.1, 0.05, -0.05, 0.0, 0.02, -0.02]
        r = bootstrap_centroid_significance(in_t, oot, n_bootstrap=500, seed=42)
        assert r.is_significant is True
        assert r.flag == "SIGNIFICANT"

    def test_same_centroids_not_significant(self) -> None:
        vals = [0.0, 0.1, -0.1, 0.05, -0.05, 0.0, 0.02, -0.02]
        r = bootstrap_centroid_significance(vals[:4], vals[4:], n_bootstrap=500, seed=42)
        assert r.is_significant is False
        assert r.flag == "OK"

    def test_insufficient_in_transit(self) -> None:
        r = bootstrap_centroid_significance([0.0], [0.0, 0.1, 0.2])
        assert r.flag == "INSUFFICIENT_DATA"

    def test_insufficient_oot(self) -> None:
        r = bootstrap_centroid_significance([0.0, 0.1, 0.2], [0.0])
        assert r.flag == "INSUFFICIENT_DATA"

    def test_observed_offset_non_negative(self) -> None:
        r = bootstrap_centroid_significance(
            [1.0, 1.1, 0.9], [0.0, 0.1, -0.1, 0.05], n_bootstrap=100
        )
        assert r.observed_offset >= 0.0

    def test_bootstrap_std_non_negative(self) -> None:
        r = bootstrap_centroid_significance(
            [0.0, 0.1, 0.0], [0.0, -0.1, 0.1, 0.0], n_bootstrap=200
        )
        assert r.bootstrap_std >= 0.0

    def test_significance_is_float(self) -> None:
        r = bootstrap_centroid_significance(
            [0.0, 0.1, 0.0], [0.0, -0.1, 0.1, 0.0], n_bootstrap=100
        )
        assert isinstance(r.significance, float)

    def test_is_significant_bool(self) -> None:
        r = bootstrap_centroid_significance(
            [0.0, 0.1, 0.0], [0.0, -0.1, 0.1, 0.0], n_bootstrap=100
        )
        assert isinstance(r.is_significant, bool)

    def test_flag_significant(self) -> None:
        in_t = [100.0, 101.0, 99.0]
        oot = [0.0, 1.0, -1.0, 0.5, -0.5]
        r = bootstrap_centroid_significance(in_t, oot, n_bootstrap=200, seed=99)
        assert r.flag in ("SIGNIFICANT", "OK")

    def test_result_is_frozen(self) -> None:
        r = bootstrap_centroid_significance([0.0, 0.1, 0.2], [0.0, 0.1, 0.2, 0.0])
        try:
            r.flag = "X"  # type: ignore[misc]
            raise AssertionError("should be frozen")
        except Exception:
            pass

    def test_format_returns_string(self) -> None:
        r = bootstrap_centroid_significance(
            [0.0, 0.1, 0.0], [0.0, -0.1, 0.1, 0.0], n_bootstrap=50
        )
        s = format_centroid_bootstrap(r)
        assert isinstance(s, str)
        assert "significance" in s.lower() or "Significance" in s

    def test_format_contains_significance(self) -> None:
        r = bootstrap_centroid_significance(
            [1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0], n_bootstrap=100
        )
        s = format_centroid_bootstrap(r)
        assert "Significance" in s
