"""Tests for Skills/rv_jitter_model.py."""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from rv_jitter_model import estimate_rv_jitter, format_rv_jitter_result


class TestEstimateRvJitter:
    def test_no_diagnostics(self) -> None:
        r = estimate_rv_jitter()
        assert r.flag == "NO_DIAGNOSTICS"

    def test_prot_solar(self) -> None:
        r = estimate_rv_jitter(prot_days=25.0)
        assert r.flag == "OK"
        assert r.jitter_ms > 0
        assert r.dominant_source == "ROTATION"

    def test_fast_rotator_more_jitter(self) -> None:
        r_fast = estimate_rv_jitter(prot_days=2.0)
        r_slow = estimate_rv_jitter(prot_days=30.0)
        assert r_fast.jitter_ms > r_slow.jitter_ms

    def test_vsini(self) -> None:
        r = estimate_rv_jitter(vsini_kms=5.0)
        assert r.flag == "OK"
        assert r.jitter_ms > 0

    def test_bv_color(self) -> None:
        r = estimate_rv_jitter(bv_color=0.65)
        assert r.flag == "OK"
        assert r.dominant_source == "GRANULATION"

    def test_teff_k(self) -> None:
        r = estimate_rv_jitter(teff_k=5778.0)
        assert r.flag == "OK"
        assert r.dominant_source == "OSCILLATION"

    def test_activity_level_quiet(self) -> None:
        r = estimate_rv_jitter(teff_k=5778.0, prot_days=25.0, bv_color=0.65)
        assert r.activity_level in ("QUIET", "MODERATE", "ACTIVE")

    def test_high_vsini_active(self) -> None:
        r = estimate_rv_jitter(vsini_kms=50.0)
        assert r.activity_level == "ACTIVE"

    def test_jitter_err_positive(self) -> None:
        r = estimate_rv_jitter(prot_days=20.0)
        assert r.jitter_err_ms > 0

    def test_multi_diagnostic_finite(self) -> None:
        r = estimate_rv_jitter(prot_days=15.0, bv_color=0.70, teff_k=5500.0)
        assert math.isfinite(r.jitter_ms)

    def test_format_output(self) -> None:
        r = estimate_rv_jitter(prot_days=25.0)
        s = format_rv_jitter_result(r)
        assert "|" in s
        assert "jitter" in s.lower() or "RV" in s

    def test_jitter_positive(self) -> None:
        r = estimate_rv_jitter(prot_days=10.0)
        assert r.jitter_ms > 0
