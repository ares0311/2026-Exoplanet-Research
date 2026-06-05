"""Tests for Skills/rv_precision_estimator.py"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from rv_precision_estimator import (
    estimate_rv_precision,
    format_rv_precision_result,
)


class TestRvPrecisionEstimator:
    def test_basic_ok(self) -> None:
        r = estimate_rv_precision(5778.0, 2.0, 8.0)
        assert r.flag == "OK"
        assert r.total_precision_ms > 0

    def test_photon_noise_positive(self) -> None:
        r = estimate_rv_precision(5778.0, 2.0, 8.0)
        assert r.photon_noise_ms > 0
        assert math.isfinite(r.photon_noise_ms)

    def test_total_ge_photon_noise(self) -> None:
        r = estimate_rv_precision(5778.0, 2.0, 8.0)
        assert r.total_precision_ms >= r.photon_noise_ms

    def test_brighter_star_lower_photon_noise(self) -> None:
        r_bright = estimate_rv_precision(5778.0, 2.0, 5.0)
        r_faint = estimate_rv_precision(5778.0, 2.0, 10.0)
        assert r_bright.photon_noise_ms < r_faint.photon_noise_ms

    def test_high_vsini_higher_noise(self) -> None:
        r_slow = estimate_rv_precision(5778.0, 1.0, 8.0)
        r_fast = estimate_rv_precision(5778.0, 20.0, 8.0)
        assert r_fast.photon_noise_ms > r_slow.photon_noise_ms

    def test_n_photons_positive(self) -> None:
        r = estimate_rv_precision(5778.0, 2.0, 8.0)
        assert r.n_photons_per_pixel >= 1.0

    def test_jitter_positive(self) -> None:
        r = estimate_rv_precision(5778.0, 2.0, 8.0)
        assert r.stellar_jitter_ms > 0

    def test_invalid_teff(self) -> None:
        r = estimate_rv_precision(0.0, 2.0, 8.0)
        assert r.flag == "INVALID_TEFF"
        assert math.isnan(r.photon_noise_ms)

    def test_invalid_vsini(self) -> None:
        r = estimate_rv_precision(5778.0, -1.0, 8.0)
        assert r.flag == "INVALID_VSINI"

    def test_invalid_resolution(self) -> None:
        r = estimate_rv_precision(5778.0, 2.0, 8.0, instrument_resolution=0.0)
        assert r.flag == "INVALID_RESOLUTION"

    def test_invalid_exposure(self) -> None:
        r = estimate_rv_precision(5778.0, 2.0, 8.0, exposure_time_min=0.0)
        assert r.flag == "INVALID_EXPOSURE"

    def test_result_is_frozen(self) -> None:
        r = estimate_rv_precision(5778.0, 2.0, 8.0)
        try:
            r.total_precision_ms = 0.0  # type: ignore[misc]
            raise AssertionError("frozen dataclass should not allow setattr")
        except Exception:
            pass

    def test_format_ok(self) -> None:
        r = estimate_rv_precision(5778.0, 2.0, 8.0)
        s = format_rv_precision_result(r)
        assert "m/s" in s
        assert "photon" in s.lower() or "Photon" in s
