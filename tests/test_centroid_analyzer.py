"""Tests for Skills.centroid_analyzer."""
from __future__ import annotations

import numpy as np
import pytest
from Skills.centroid_analyzer import CentroidResult, analyze_centroid, format_centroid_result


def _flat_centroid(n: int = 200) -> tuple[list[float], list[float], list[float], list[float]]:
    t = list(np.linspace(2458000.0, 2458020.0, n))
    f = [1.0] * n
    ra  = [0.01 + 0.0001 * i for i in range(n)]   # tiny drift, no transit shift
    dec = [0.02 + 0.0001 * i for i in range(n)]
    return t, f, ra, dec


class TestAnalyzeCentroid:
    def test_returns_centroid_result(self) -> None:
        t, f, ra, dec = _flat_centroid()
        result = analyze_centroid(t, f, ra, dec, 5.0, 2458002.0)
        assert isinstance(result, CentroidResult)

    def test_no_shift_for_flat_centroid(self) -> None:
        t = list(np.linspace(2458000.0, 2458020.0, 200))
        f = [1.0] * 200
        ra  = [0.5] * 200
        dec = [0.5] * 200
        result = analyze_centroid(t, f, ra, dec, 5.0, 2458002.0, threshold_arcsec=2.0)
        assert not result.is_shifted

    def test_shift_detected_for_large_centroid_jump(self) -> None:
        t = list(np.linspace(2458000.0, 2458010.0, 100))
        f = [1.0] * 100
        # Centroid jumps 100 pixels during transit (in-transit frames)
        period, epoch, dur = 5.0, 2458002.0, 0.2
        ra, dec = [], []
        for ti in t:
            ph = (ti - epoch) % period
            if ph > period / 2:
                ph -= period
            if abs(ph) <= dur / 2:
                ra.append(100.0)
                dec.append(100.0)
            else:
                ra.append(0.0)
                dec.append(0.0)
        result = analyze_centroid(t, f, ra, dec, period, epoch,
                                   duration_days=dur, threshold_arcsec=0.5)
        assert result.is_shifted

    def test_delta_arcsec_non_negative(self) -> None:
        t, f, ra, dec = _flat_centroid()
        result = analyze_centroid(t, f, ra, dec, 5.0, 2458002.0)
        assert result.delta_arcsec >= 0.0

    def test_delta_significance_non_negative(self) -> None:
        t, f, ra, dec = _flat_centroid()
        result = analyze_centroid(t, f, ra, dec, 5.0, 2458002.0)
        assert result.delta_significance >= 0.0

    def test_note_empty_when_no_shift(self) -> None:
        t = list(np.linspace(2458000.0, 2458010.0, 100))
        f = [1.0] * 100
        ra = dec = [0.001] * 100
        result = analyze_centroid(t, f, ra, dec, 5.0, 2458002.0, threshold_arcsec=10.0)
        # No shift → note should not mention background source
        assert "background" not in result.note.lower()

    def test_insufficient_data_returns_zeros(self) -> None:
        result = analyze_centroid([1.0], [1.0], [0.1], [0.1], 5.0, 2458002.0)
        assert result.delta_arcsec == pytest.approx(0.0)
        assert not result.is_shifted

    def test_in_transit_centroid_stored(self) -> None:
        t, f, ra, dec = _flat_centroid()
        result = analyze_centroid(t, f, ra, dec, 5.0, 2458002.0)
        assert isinstance(result.in_transit_centroid_arcsec, float)


class TestFormatCentroidResult:
    def test_format_contains_delta(self) -> None:
        t, f, ra, dec = _flat_centroid()
        result = analyze_centroid(t, f, ra, dec, 5.0, 2458002.0)
        text = format_centroid_result(result)
        assert "Delta" in text

    def test_format_contains_ok_or_shifted(self) -> None:
        t, f, ra, dec = _flat_centroid()
        result = analyze_centroid(t, f, ra, dec, 5.0, 2458002.0, threshold_arcsec=100.0)
        text = format_centroid_result(result)
        assert "OK" in text or "SHIFTED" in text
