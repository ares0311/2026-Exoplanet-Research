"""Tests for transmission_window_predictor.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from transmission_window_predictor import (
    format_transmission_window_result,
    predict_transit_windows,
)

# Use arbitrary BJD values in the near future
BJD_START = 2460000.0
BJD_END = BJD_START + 30.0  # 30 days


class TestPredictTransitWindows:
    def test_basic_ok(self):
        r = predict_transit_windows(5.0, BJD_START, BJD_START, BJD_END)
        assert r.flag == "OK"
        assert r.n_windows > 0

    def test_correct_number_of_windows(self):
        r = predict_transit_windows(5.0, BJD_START, BJD_START, BJD_END)
        # epoch=BJD_START, n=0..6 all satisfy mid <= BJD_END → 7 windows
        assert r.n_windows == 7

    def test_window_bjd_in_range(self):
        r = predict_transit_windows(5.0, BJD_START, BJD_START, BJD_END)
        for w in r.windows:
            assert BJD_START <= w.bjd_mid <= BJD_END

    def test_transit_numbers_sequential(self):
        r = predict_transit_windows(5.0, BJD_START, BJD_START, BJD_END)
        numbers = [w.transit_number for w in r.windows]
        assert numbers == sorted(numbers)

    def test_no_windows_when_range_too_small(self):
        r = predict_transit_windows(10.0, BJD_START, BJD_START + 0.1, BJD_START + 0.5)
        assert r.flag in ("NO_WINDOWS", "OK")

    def test_invalid_zero_period(self):
        r = predict_transit_windows(0.0, BJD_START, BJD_START, BJD_END)
        assert r.flag == "INVALID"

    def test_invalid_end_before_start(self):
        r = predict_transit_windows(5.0, BJD_START, BJD_END, BJD_START)
        assert r.flag == "INVALID"

    def test_utc_mid_nonempty(self):
        r = predict_transit_windows(5.0, BJD_START, BJD_START, BJD_END)
        for w in r.windows:
            assert len(w.utc_mid) > 5

    def test_observability_without_airmass_fn(self):
        r = predict_transit_windows(5.0, BJD_START, BJD_START, BJD_END)
        # Without airmass/moon functions all windows default to observable
        for w in r.windows:
            assert w.is_observable

    def test_airmass_fn_injectable(self):
        class MockAirmassResult:
            airmass = 1.5
            is_observable = True

        def mock_airmass(ra, dec, lat, lon, bjd):
            return MockAirmassResult()

        r = predict_transit_windows(
            5.0, BJD_START, BJD_START, BJD_END,
            obs_lat_deg=-24.6, obs_lon_deg=-70.4,
            airmass_fn=mock_airmass,
        )
        for w in r.windows:
            assert w.airmass_mid == 1.5

    def test_moon_fn_injectable(self):
        class MockMoonResult:
            moon_separation_deg = 45.0
            moon_illumination_fraction = 0.3
            is_problematic = False

        def mock_moon(ra, dec, bjd):
            return MockMoonResult()

        r = predict_transit_windows(
            5.0, BJD_START, BJD_START, BJD_END,
            moon_fn=mock_moon,
        )
        for w in r.windows:
            assert w.moon_separation_deg == 45.0

    def test_moon_interference_marks_not_observable(self):
        class MockMoonResult:
            moon_separation_deg = 10.0
            moon_illumination_fraction = 0.9
            is_problematic = True

        r = predict_transit_windows(
            5.0, BJD_START, BJD_START, BJD_END,
            moon_fn=lambda ra, dec, bjd: MockMoonResult(),
        )
        for w in r.windows:
            assert not w.is_observable

    def test_n_observable_consistent(self):
        r = predict_transit_windows(5.0, BJD_START, BJD_START, BJD_END)
        assert r.n_observable == sum(1 for w in r.windows if w.is_observable)

    def test_result_frozen(self):
        r = predict_transit_windows(5.0, BJD_START, BJD_START, BJD_END)
        try:
            r.n_windows = 0  # type: ignore[misc]
            raise AssertionError()
        except (AttributeError, TypeError):
            pass


class TestFormatTransmissionWindowResult:
    def test_returns_string(self):
        r = predict_transit_windows(5.0, BJD_START, BJD_START, BJD_END)
        assert isinstance(format_transmission_window_result(r), str)

    def test_contains_flag(self):
        r = predict_transit_windows(5.0, BJD_START, BJD_START, BJD_END)
        s = format_transmission_window_result(r)
        assert r.flag in s

    def test_contains_n_windows(self):
        r = predict_transit_windows(5.0, BJD_START, BJD_START, BJD_END)
        s = format_transmission_window_result(r)
        assert str(r.n_windows) in s
