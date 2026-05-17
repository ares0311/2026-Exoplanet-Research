"""Tests for Skills.signal_statistics."""
from __future__ import annotations

import numpy as np
import pytest
from Skills.signal_statistics import SignalStats, compute_signal_stats, format_signal_stats


def _clean_lc_with_transit(
    n: int = 300,
    period: float = 5.0,
    epoch: float = 2458002.0,
    depth: float = 0.005,
    duration: float = 0.1,
) -> tuple[list[float], list[float]]:
    rng = np.random.default_rng(42)
    t = list(np.linspace(2458000.0, 2458027.0, n))
    f = []
    for ti in t:
        ph = (ti - epoch) % period
        if ph > period / 2:
            ph -= period
        f.append(1.0 - depth if abs(ph) <= duration / 2 else 1.0)
    noise = rng.normal(0, 1e-4, n)
    return t, [fi + ni for fi, ni in zip(f, noise, strict=False)]


class TestComputeSignalStats:
    def test_returns_signal_stats(self) -> None:
        t, f = _clean_lc_with_transit()
        stats = compute_signal_stats(t, f, 5.0, 2458002.0)
        assert isinstance(stats, SignalStats)

    def test_per_transit_snr_non_empty(self) -> None:
        t, f = _clean_lc_with_transit()
        stats = compute_signal_stats(t, f, 5.0, 2458002.0)
        assert len(stats.per_transit_snr) > 0

    def test_median_snr_positive(self) -> None:
        t, f = _clean_lc_with_transit()
        stats = compute_signal_stats(t, f, 5.0, 2458002.0, duration_days=0.1)
        assert stats.median_snr > 0.0

    def test_no_flags_for_clean_signal(self) -> None:
        t, f = _clean_lc_with_transit(depth=0.01)
        stats = compute_signal_stats(t, f, 5.0, 2458002.0, duration_days=0.1)
        assert "SECONDARY_DETECTED" not in stats.flags

    def test_secondary_detected_when_injected(self) -> None:
        # Inject a secondary at P/2
        rng = np.random.default_rng(7)
        period, epoch, depth = 10.0, 2458005.0, 0.01
        t = list(np.linspace(2458000.0, 2458060.0, 600))
        f = []
        for ti in t:
            ph = (ti - epoch) % period
            if ph > period / 2:
                ph -= period
            # primary
            in_p = abs(ph) <= 0.05
            # secondary at +5 days
            ph2 = (ti - (epoch + period / 2)) % period
            if ph2 > period / 2:
                ph2 -= period
            in_s = abs(ph2) <= 0.05
            f.append(1.0 - depth * (in_p or in_s))
        noise = rng.normal(0, 1e-5, 600)
        f = [fi + ni for fi, ni in zip(f, noise, strict=False)]
        stats = compute_signal_stats(t, f, period, epoch, duration_days=0.1,
                                      secondary_threshold_snr=1.0)
        assert "SECONDARY_DETECTED" in stats.flags

    def test_odd_even_depths_returned(self) -> None:
        t, f = _clean_lc_with_transit()
        stats = compute_signal_stats(t, f, 5.0, 2458002.0)
        # May be None if not enough transits on each parity
        if stats.odd_depth_ppm is not None and stats.even_depth_ppm is not None:
            assert isinstance(stats.odd_depth_ppm, float)

    def test_negative_period_raises(self) -> None:
        t, f = _clean_lc_with_transit()
        with pytest.raises(ValueError):
            compute_signal_stats(t, f, -1.0, 2458002.0)

    def test_flags_is_tuple(self) -> None:
        t, f = _clean_lc_with_transit()
        stats = compute_signal_stats(t, f, 5.0, 2458002.0)
        assert isinstance(stats.flags, tuple)

    def test_secondary_snr_non_negative(self) -> None:
        t, f = _clean_lc_with_transit()
        stats = compute_signal_stats(t, f, 5.0, 2458002.0)
        assert stats.secondary_snr >= 0.0


class TestFormatSignalStats:
    def test_format_contains_median_snr(self) -> None:
        t, f = _clean_lc_with_transit()
        stats = compute_signal_stats(t, f, 5.0, 2458002.0)
        text = format_signal_stats(stats)
        assert "SNR" in text

    def test_format_contains_no_flags_note(self) -> None:
        t, f = _clean_lc_with_transit()
        stats = compute_signal_stats(t, f, 5.0, 2458002.0)
        text = format_signal_stats(stats)
        assert "none" in text.lower() or len(stats.flags) > 0
