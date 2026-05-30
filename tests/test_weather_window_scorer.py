"""Tests for Skills/weather_window_scorer.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from weather_window_scorer import (  # noqa: E402
    WeatherWindowResult,
    format_weather_window,
    score_weather_windows,
)

GOOD_CONDITIONS = [
    {"seeing_arcsec": 1.5, "cloud_cover_pct": 0.0, "humidity_pct": 40.0, "wind_kph": 10.0}
] * 6

BAD_CONDITIONS = [
    {"seeing_arcsec": 5.0, "cloud_cover_pct": 100.0, "humidity_pct": 95.0, "wind_kph": 60.0}
] * 4


def test_returns_dataclass():
    r = score_weather_windows(GOOD_CONDITIONS)
    assert isinstance(r, WeatherWindowResult)


def test_empty_input():
    r = score_weather_windows([])
    assert r.flag == "POOR"
    assert r.n_hours == 0


def test_good_conditions_flag():
    r = score_weather_windows(GOOD_CONDITIONS)
    assert r.flag == "GOOD"


def test_poor_conditions_flag():
    r = score_weather_windows(BAD_CONDITIONS)
    assert r.flag == "POOR"


def test_n_hours_correct():
    conditions = GOOD_CONDITIONS[:4]
    r = score_weather_windows(conditions)
    assert r.n_hours == 4


def test_scores_length():
    r = score_weather_windows(GOOD_CONDITIONS)
    assert len(r.scores) == len(GOOD_CONDITIONS)


def test_scores_in_range():
    r = score_weather_windows(GOOD_CONDITIONS + BAD_CONDITIONS)
    for s in r.scores:
        assert 0.0 <= s <= 1.0


def test_mean_score_good_conditions():
    r = score_weather_windows(GOOD_CONDITIONS)
    assert r.mean_score > 0.65


def test_mean_score_bad_conditions():
    r = score_weather_windows(BAD_CONDITIONS)
    assert r.mean_score < 0.4


def test_best_start_idx_valid():
    conditions = [
        {"seeing_arcsec": 3.0, "cloud_cover_pct": 50.0, "humidity_pct": 70.0, "wind_kph": 20.0},
        {"seeing_arcsec": 1.5, "cloud_cover_pct": 0.0, "humidity_pct": 40.0, "wind_kph": 5.0},
        {"seeing_arcsec": 1.5, "cloud_cover_pct": 0.0, "humidity_pct": 40.0, "wind_kph": 5.0},
    ]
    r = score_weather_windows(conditions)
    if r.best_start_idx is not None:
        assert 0 <= r.best_start_idx < r.n_hours


def test_n_good_hours_correct():
    r = score_weather_windows(GOOD_CONDITIONS)
    assert r.n_good_hours == len(GOOD_CONDITIONS)


def test_format_returns_string():
    r = score_weather_windows(GOOD_CONDITIONS)
    s = format_weather_window(r)
    assert isinstance(s, str)


def test_format_contains_flag():
    r = score_weather_windows(GOOD_CONDITIONS)
    s = format_weather_window(r)
    assert "Flag" in s
