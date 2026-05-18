"""Tests for Skills.stellar_activity_filter."""
from __future__ import annotations

from Skills.stellar_activity_filter import (
    ActivityFilterResult,
    apply_activity_mask,
    filter_stellar_activity,
    format_activity_result,
)


def _flat_lc(n=500, noise=1e-4):
    import random
    rng = random.Random(42)
    time = [2458000.0 + i * (2.0 / 1440) for i in range(n)]
    flux = [1.0 + rng.gauss(0, noise) for _ in range(n)]
    return time, flux


def _lc_with_flares(n=500):
    time, flux = _flat_lc(n, noise=1e-4)
    flux_list = list(flux)
    # Inject a large flare at index 100
    for i in range(98, 105):
        flux_list[i] += 0.05
    return time, flux_list


class TestFilterStellarActivity:
    def test_returns_result(self) -> None:
        t, f = _flat_lc()
        r = filter_stellar_activity(t, f)
        assert isinstance(r, ActivityFilterResult)

    def test_empty_returns_quiet(self) -> None:
        r = filter_stellar_activity([], [])
        assert r.flag == "QUIET"

    def test_flat_lc_quiet(self) -> None:
        t, f = _flat_lc(noise=1e-5)
        r = filter_stellar_activity(t, f, sigma_upper=5.0)
        assert r.flag == "QUIET"

    def test_flare_detected(self) -> None:
        t, f = _lc_with_flares()
        r = filter_stellar_activity(t, f, sigma_upper=3.0)
        assert r.n_flagged > 0

    def test_fraction_in_range(self) -> None:
        t, f = _flat_lc()
        r = filter_stellar_activity(t, f)
        assert 0.0 <= r.activity_fraction <= 1.0

    def test_flagged_indices_in_range(self) -> None:
        t, f = _lc_with_flares()
        r = filter_stellar_activity(t, f)
        assert all(0 <= i < len(t) for i in r.flagged_indices)

    def test_baseline_rms_positive(self) -> None:
        t, f = _flat_lc(noise=1e-3)
        r = filter_stellar_activity(t, f)
        assert r.baseline_rms_ppm > 0

    def test_flag_values_valid(self) -> None:
        t, f = _flat_lc()
        r = filter_stellar_activity(t, f)
        assert r.flag in {"QUIET", "ACTIVE", "VERY_ACTIVE"}


class TestApplyActivityMask:
    def test_removes_flagged_cadences(self) -> None:
        t, f = _lc_with_flares()
        r = filter_stellar_activity(t, f)
        t_c, f_c, _ = apply_activity_mask(t, f, None, r)
        assert len(t_c) == len(t) - r.n_flagged

    def test_flux_err_passed_through(self) -> None:
        t, f = _lc_with_flares()
        e = [1e-4] * len(f)
        r = filter_stellar_activity(t, f)
        _, _, e_c = apply_activity_mask(t, f, e, r)
        assert e_c is not None
        assert len(e_c) == len(f) - r.n_flagged


class TestFormatActivityResult:
    def test_returns_string(self) -> None:
        t, f = _flat_lc()
        r = filter_stellar_activity(t, f)
        assert isinstance(format_activity_result(r), str)

    def test_contains_flag(self) -> None:
        t, f = _flat_lc()
        r = filter_stellar_activity(t, f)
        assert r.flag in format_activity_result(r)
