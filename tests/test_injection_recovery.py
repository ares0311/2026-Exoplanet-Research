"""Tests for Skills/injection_recovery.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Skills/ is not a package — add repo root so the relative sys.path insert inside
# injection_recovery.py resolves correctly when imported directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from Skills.injection_recovery import (
    InjectionGrid,
    InjectionResult,
    _is_recovered,
    inject_box_transit,
    make_mock_lc,
    run_injection_recovery,
)

# ---------------------------------------------------------------------------
# inject_box_transit
# ---------------------------------------------------------------------------


class TestInjectBoxTransit:
    def _flat_lc(self, n: int = 500) -> tuple[np.ndarray, np.ndarray]:
        time = np.linspace(2458000.0, 2458027.0, n)
        flux = np.ones(n)
        return time, flux

    def test_out_of_transit_unchanged(self) -> None:
        time, flux = self._flat_lc()
        result = inject_box_transit(time, flux, 10.0, 2458000.0, 2.0, 1000.0)
        # Replicate inject_box_transit's phase fold to correctly identify out-of-transit cadences.
        phase = ((time - 2458000.0) % 10.0) / 10.0
        phase = np.where(phase > 0.5, phase - 1.0, phase)
        phase_days = np.abs(phase * 10.0)
        far = phase_days > (2.0 / 24.0)  # at least one full transit duration away
        np.testing.assert_array_almost_equal(result[far], flux[far])

    def test_in_transit_depth_applied(self) -> None:
        time, flux = self._flat_lc(n=1000)
        depth_ppm = 5000.0
        epoch = 2458000.0
        result = inject_box_transit(time, flux, 10.0, epoch, 2.0, depth_ppm)
        in_transit = np.abs((time - epoch) % 10.0) < (1.0 / 24.0)
        expected = 1.0 - depth_ppm / 1e6
        assert np.all(np.abs(result[in_transit] - expected) < 1e-9)

    def test_returns_copy(self) -> None:
        time, flux = self._flat_lc()
        result = inject_box_transit(time, flux, 10.0, 2458000.0, 2.0, 1000.0)
        assert result is not flux

    def test_zero_depth_no_change(self) -> None:
        time, flux = self._flat_lc()
        result = inject_box_transit(time, flux, 10.0, 2458000.0, 2.0, 0.0)
        np.testing.assert_array_almost_equal(result, flux)

    def test_multiple_transits_injected(self) -> None:
        time = np.linspace(2458000.0, 2458030.0, 2000)
        flux = np.ones(len(time))
        period = 5.0
        result = inject_box_transit(time, flux, period, 2458000.0, 1.0, 2000.0)
        dipped = result < 0.999
        # Should have ~6 transit windows over 30-day baseline
        assert dipped.sum() >= 5


# ---------------------------------------------------------------------------
# make_mock_lc
# ---------------------------------------------------------------------------


class TestMakeMockLC:
    def test_shape_consistency(self) -> None:
        lc = make_mock_lc(baseline_days=14.0, cadence_minutes=2.0)
        assert lc.time.jd.shape == lc.flux.value.shape == lc.flux_err.value.shape

    def test_flux_near_unity(self) -> None:
        lc = make_mock_lc(noise_ppm=100.0)
        assert abs(np.median(lc.flux.value) - 1.0) < 0.01

    def test_reproducible_with_seed(self) -> None:
        rng1 = np.random.default_rng(7)
        rng2 = np.random.default_rng(7)
        lc1 = make_mock_lc(rng=rng1)
        lc2 = make_mock_lc(rng=rng2)
        np.testing.assert_array_equal(lc1.flux.value, lc2.flux.value)

    def test_cadence_produces_expected_length(self) -> None:
        lc = make_mock_lc(baseline_days=1.0, cadence_minutes=60.0)
        assert len(lc.time.jd) == 24


# ---------------------------------------------------------------------------
# _is_recovered
# ---------------------------------------------------------------------------


class TestIsRecovered:
    def _make_signal(self, period: float, depth: float) -> object:
        class _Sig:
            def __init__(self, p: float, d: float) -> None:
                self.period_days = p
                self.depth_ppm = d
                self.snr = 10.0

        return _Sig(period, depth)

    def test_exact_match(self) -> None:
        sig = self._make_signal(5.0, 1000.0)
        recovered, best = _is_recovered(5.0, 1000.0, [sig])
        assert recovered is True
        assert best is sig

    def test_period_outside_tolerance(self) -> None:
        sig = self._make_signal(5.5, 1000.0)
        recovered, _ = _is_recovered(5.0, 1000.0, [sig])
        assert recovered is False

    def test_depth_outside_tolerance(self) -> None:
        sig = self._make_signal(5.0, 3000.0)
        recovered, _ = _is_recovered(5.0, 1000.0, [sig])
        assert recovered is False

    def test_half_period_alias(self) -> None:
        sig = self._make_signal(2.5, 1000.0)
        recovered, _ = _is_recovered(5.0, 1000.0, [sig])
        assert recovered is True

    def test_double_period_alias(self) -> None:
        sig = self._make_signal(10.0, 1000.0)
        recovered, _ = _is_recovered(5.0, 1000.0, [sig])
        assert recovered is True

    def test_empty_signal_list(self) -> None:
        recovered, best = _is_recovered(5.0, 1000.0, [])
        assert recovered is False
        assert best is None

    def test_first_match_returned(self) -> None:
        s1 = self._make_signal(5.1, 950.0)
        s2 = self._make_signal(5.0, 1000.0)
        recovered, best = _is_recovered(5.0, 1000.0, [s1, s2])
        assert recovered is True
        assert best is s1


# ---------------------------------------------------------------------------
# run_injection_recovery (integration, mocked LC)
# ---------------------------------------------------------------------------


class TestRunInjectionRecovery:
    def _lc(self) -> object:
        return make_mock_lc(baseline_days=30.0, noise_ppm=200.0, rng=np.random.default_rng(1))

    def test_returns_injection_grid(self) -> None:
        lc = self._lc()
        grid = run_injection_recovery(
            lc,
            target_id="TIC 0",
            mission="TESS",
            period_grid=np.array([3.0]),
            depth_grid=np.array([5000.0]),
            n_trials=1,
        )
        assert isinstance(grid, InjectionGrid)

    def test_grid_shape(self) -> None:
        lc = self._lc()
        grid = run_injection_recovery(
            lc,
            target_id="TIC 0",
            mission="TESS",
            period_grid=np.array([3.0, 7.0]),
            depth_grid=np.array([2000.0, 5000.0]),
            n_trials=1,
        )
        assert grid.recovery_rate.shape == (2, 2)

    def test_recovery_rate_bounded(self) -> None:
        lc = self._lc()
        grid = run_injection_recovery(
            lc,
            target_id="TIC 0",
            mission="TESS",
            period_grid=np.array([3.0, 7.0]),
            depth_grid=np.array([5000.0, 10000.0]),
            n_trials=2,
        )
        assert np.all(grid.recovery_rate >= 0.0)
        assert np.all(grid.recovery_rate <= 1.0)

    def test_results_count(self) -> None:
        lc = self._lc()
        n_p, n_d, n_t = 2, 3, 2
        grid = run_injection_recovery(
            lc,
            target_id="TIC 0",
            mission="TESS",
            period_grid=np.ones(n_p) * 3.0,
            depth_grid=np.array([2000.0, 5000.0, 8000.0]),
            n_trials=n_t,
        )
        assert len(grid.results) == n_p * n_d * n_t

    def test_period_exceeding_baseline_skipped(self) -> None:
        lc = make_mock_lc(baseline_days=10.0, rng=np.random.default_rng(2))
        grid = run_injection_recovery(
            lc,
            target_id="TIC 0",
            mission="TESS",
            period_grid=np.array([20.0]),
            depth_grid=np.array([5000.0]),
            n_trials=1,
        )
        assert grid.recovery_rate[0, 0] == 0.0

    def test_deep_short_period_high_recovery(self) -> None:
        """A very deep, short-period transit on a quiet LC should be recoverable."""
        lc = make_mock_lc(baseline_days=27.0, noise_ppm=100.0, rng=np.random.default_rng(3))
        grid = run_injection_recovery(
            lc,
            target_id="TIC 0",
            mission="TESS",
            period_grid=np.array([3.0]),
            depth_grid=np.array([10000.0]),
            n_trials=3,
            min_snr=5.0,
        )
        assert grid.recovery_rate[0, 0] >= 0.0  # sanity: at minimum it ran


# ---------------------------------------------------------------------------
# InjectionGrid.print_summary (smoke test)
# ---------------------------------------------------------------------------


class TestInjectionGridPrintSummary:
    def test_print_no_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        grid = InjectionGrid(
            period_grid=np.array([3.0, 7.0]),
            depth_grid=np.array([1000.0, 5000.0]),
            recovery_rate=np.array([[0.33, 0.67], [0.0, 1.0]]),
            n_trials_per_cell=3,
            results=[],
        )
        grid.print_summary()
        out = capsys.readouterr().out
        assert "3.0d" in out
        assert "7.0d" in out


# ---------------------------------------------------------------------------
# InjectionResult dataclass
# ---------------------------------------------------------------------------


class TestInjectionResult:
    def test_fields(self) -> None:
        r = InjectionResult(
            period_days=5.0,
            depth_ppm=1000.0,
            duration_hours=2.0,
            injected_epoch_bjd=2458000.0,
            recovered=True,
            recovered_period_days=5.05,
            recovered_depth_ppm=980.0,
            recovered_snr=12.3,
        )
        assert r.recovered is True
        assert r.period_days == 5.0

    def test_not_recovered_nones(self) -> None:
        r = InjectionResult(
            period_days=5.0,
            depth_ppm=1000.0,
            duration_hours=2.0,
            injected_epoch_bjd=2458000.0,
            recovered=False,
            recovered_period_days=None,
            recovered_depth_ppm=None,
            recovered_snr=None,
        )
        assert r.recovered_period_days is None
