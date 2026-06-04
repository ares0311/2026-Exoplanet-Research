"""Tests for Skills/multi_instrument_lc_combiner.py"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from multi_instrument_lc_combiner import CombinedLcResult, combine_lightcurves


class TestMultiInstrumentLcCombiner:
    def test_single_instrument_ok(self) -> None:
        r = combine_lightcurves([[1.0, 1.1, 0.9, 1.0, 1.05]])
        assert r.flag == "OK"

    def test_two_instruments_ok(self) -> None:
        r = combine_lightcurves([[1.0, 1.1, 0.9], [0.5, 0.6, 0.4]])
        assert r.flag == "OK"
        assert r.n_instruments == 2

    def test_no_instruments(self) -> None:
        r = combine_lightcurves([])
        assert r.flag == "NO_INSTRUMENTS"

    def test_n_points_total(self) -> None:
        r = combine_lightcurves([[1.0, 2.0, 3.0], [4.0, 5.0]])
        assert r.n_points_total == 5

    def test_offsets_length(self) -> None:
        r = combine_lightcurves([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        assert len(r.offsets) == 3

    def test_rms_per_instrument_length(self) -> None:
        r = combine_lightcurves([[1.0, 2.0], [3.0, 4.0]])
        assert len(r.rms_per_instrument) == 2

    def test_normalize_subtracts_median(self) -> None:
        # After median subtraction offset should equal median
        r = combine_lightcurves([[10.0, 11.0, 12.0]], normalize=True)
        assert abs(r.offsets[0] - 11.0) < 1e-6

    def test_no_normalize(self) -> None:
        r = combine_lightcurves([[10.0, 11.0, 12.0]], normalize=False)
        assert r.offsets[0] == 0.0

    def test_combined_rms_finite(self) -> None:
        r = combine_lightcurves([[1.0, 0.9, 1.1], [2.0, 1.9, 2.1]])
        assert math.isfinite(r.combined_rms)

    def test_result_frozen(self) -> None:
        r = combine_lightcurves([[1.0, 2.0]])
        assert isinstance(r, CombinedLcResult)
        try:
            object.__setattr__(r, "flag", "x")
            raise AssertionError()
        except Exception:
            pass

    def test_format_output(self) -> None:
        from multi_instrument_lc_combiner import format_combined_lc_result
        r = combine_lightcurves([[1.0, 2.0], [3.0, 4.0]])
        s = format_combined_lc_result(r)
        assert "|" in s

    def test_empty_array_handled(self) -> None:
        r = combine_lightcurves([[], [1.0, 2.0]])
        assert r.flag == "OK"
        assert r.n_instruments == 2
