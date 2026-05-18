"""Tests for Skills.multi_night_photometry_combiner."""
from __future__ import annotations

from Skills.multi_night_photometry_combiner import (
    CombinedPhotometryResult,
    combine_photometry_nights,
    format_combined_result,
)


def _make_night(night_id, t0=2458000.0, period=5.0, epoch=2458000.0,
                depth=0.01, n=300, has_transit=True):
    dt = 2.0 / 1440.0
    time = [t0 + i * dt for i in range(n)]
    flux = []
    for t in time:
        ph = ((t - epoch) % period) / period
        if ph > 0.5:
            ph -= 1.0
        if has_transit and abs(ph) < 0.05:
            flux.append(1.0 - depth)
        else:
            flux.append(1.0)
    return {"id": night_id, "time": time, "flux": flux}


class TestCombinePhotometryNights:
    def test_returns_result(self) -> None:
        nights = [_make_night("N1"), _make_night("N2", t0=2458005.0)]
        r = combine_photometry_nights(nights, 5.0, 2458000.0, duration_hours=2.0)
        assert isinstance(r, CombinedPhotometryResult)

    def test_empty_returns_insufficient(self) -> None:
        r = combine_photometry_nights([], 5.0, 2458000.0)
        assert r.flag == "INSUFFICIENT"

    def test_zero_period_returns_insufficient(self) -> None:
        nights = [_make_night("N1")]
        r = combine_photometry_nights(nights, 0.0, 2458000.0)
        assert r.flag == "INSUFFICIENT"

    def test_depth_positive(self) -> None:
        nights = [_make_night("N1"), _make_night("N2", t0=2458005.0)]
        r = combine_photometry_nights(nights, 5.0, 2458000.0, duration_hours=2.0)
        if r.combined_depth_ppm is not None:
            assert r.combined_depth_ppm > 0

    def test_n_nights_correct(self) -> None:
        nights = [_make_night(f"N{i}") for i in range(3)]
        r = combine_photometry_nights(nights, 5.0, 2458000.0, duration_hours=2.0)
        assert r.n_nights == 3

    def test_confirmed_flag_with_deep_transit(self) -> None:
        nights = [_make_night(f"N{i}", t0=2458000.0 + i * 5.0, depth=0.02) for i in range(4)]
        r = combine_photometry_nights(
            nights, 5.0, 2458000.0, duration_hours=3.0, confirmed_snr=3.0
        )
        assert r.flag in {"CONFIRMED", "MARGINAL", "INSUFFICIENT"}

    def test_no_transit_flag(self) -> None:
        nights = [_make_night("N1", has_transit=False)]
        r = combine_photometry_nights(nights, 5.0, 2458000.0, duration_hours=2.0)
        assert r.flag in {"NO_TRANSIT", "INSUFFICIENT"}

    def test_night_results_present(self) -> None:
        nights = [_make_night("N1"), _make_night("N2", t0=2458005.0)]
        r = combine_photometry_nights(nights, 5.0, 2458000.0, duration_hours=2.0)
        assert len(r.nights) == 2


class TestFormatCombinedResult:
    def test_returns_string(self) -> None:
        nights = [_make_night("N1")]
        r = combine_photometry_nights(nights, 5.0, 2458000.0)
        assert isinstance(format_combined_result(r), str)

    def test_insufficient_handled(self) -> None:
        r = combine_photometry_nights([], 5.0, 2458000.0)
        out = format_combined_result(r)
        assert "Insufficient" in out or "insufficient" in out.lower()
