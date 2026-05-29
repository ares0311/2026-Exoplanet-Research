"""Tests for Skills/system_architecture_summarizer.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from system_architecture_summarizer import SystemArchResult, summarize_system_architecture, format_system_arch


class TestSystemArchResult:
    def test_dataclass_fields(self):
        r = SystemArchResult(
            n_planets=3, architecture_class="compact",
            period_ratio_min=1.5, period_ratio_max=2.0,
            n_rocky=2, n_giant=0, flag="COMPACT"
        )
        assert r.n_planets == 3
        assert r.architecture_class == "compact"

    def test_frozen(self):
        r = SystemArchResult(
            n_planets=1, architecture_class="SINGLE",
            period_ratio_min=None, period_ratio_max=None,
            n_rocky=1, n_giant=0, flag="SINGLE"
        )
        try:
            r.n_planets = 0
            assert False
        except Exception:
            pass


class TestSummarizeSystemArchitecture:
    def test_empty_list_error(self):
        r = summarize_system_architecture([])
        assert r.flag == "ERROR"
        assert r.n_planets == 0

    def test_single_planet(self):
        planets = [{"period_days": 10.0, "radius_rearth": 1.0}]
        r = summarize_system_architecture(planets)
        assert r.n_planets == 1
        assert r.architecture_class == "SINGLE"
        assert r.flag == "SINGLE"
        assert r.period_ratio_min is None

    def test_compact_system(self):
        planets = [
            {"period_days": 3.0, "radius_rearth": 1.5},
            {"period_days": 7.0, "radius_rearth": 2.0},
            {"period_days": 15.0, "radius_rearth": 1.0},
        ]
        r = summarize_system_architecture(planets)
        assert r.architecture_class == "compact"
        assert r.flag == "COMPACT"

    def test_spread_system(self):
        planets = [
            {"period_days": 10.0, "radius_rearth": 1.0},
            {"period_days": 365.0, "radius_rearth": 11.0},
        ]
        r = summarize_system_architecture(planets)
        assert r.architecture_class == "spread"

    def test_mixed_system(self):
        planets = [
            {"period_days": 10.0, "radius_rearth": 1.0},
            {"period_days": 100.0, "radius_rearth": 2.0},
        ]
        r = summarize_system_architecture(planets)
        assert r.architecture_class == "mixed"

    def test_period_ratios(self):
        planets = [
            {"period_days": 10.0, "radius_rearth": 1.0},
            {"period_days": 20.0, "radius_rearth": 1.0},
        ]
        r = summarize_system_architecture(planets)
        assert abs(r.period_ratio_min - 2.0) < 0.001
        assert abs(r.period_ratio_max - 2.0) < 0.001

    def test_sorted_by_period(self):
        # Provide out of order
        planets = [
            {"period_days": 30.0, "radius_rearth": 1.0},
            {"period_days": 10.0, "radius_rearth": 1.0},
        ]
        r = summarize_system_architecture(planets)
        assert r.period_ratio_min > 1.0  # sorted: 10, 30 → ratio=3.0

    def test_n_rocky_count(self):
        planets = [
            {"period_days": 5.0, "radius_rearth": 1.0},   # rocky
            {"period_days": 10.0, "radius_rearth": 1.8},  # rocky
            {"period_days": 20.0, "radius_rearth": 3.0},  # sub_neptune
        ]
        r = summarize_system_architecture(planets)
        assert r.n_rocky == 2

    def test_n_giant_count(self):
        planets = [
            {"period_days": 5.0, "radius_rearth": 1.0},
            {"period_days": 365.0, "radius_rearth": 11.0},  # giant
        ]
        r = summarize_system_architecture(planets)
        assert r.n_giant == 1

    def test_n_planets_correct(self):
        planets = [{"period_days": float(i), "radius_rearth": 1.0} for i in range(1, 6)]
        r = summarize_system_architecture(planets)
        assert r.n_planets == 5


class TestFormatSystemArch:
    def test_returns_string(self):
        planets = [{"period_days": 10.0, "radius_rearth": 1.0}]
        r = summarize_system_architecture(planets)
        s = format_system_arch(r)
        assert isinstance(s, str)

    def test_contains_architecture_class(self):
        planets = [
            {"period_days": 5.0, "radius_rearth": 1.0},
            {"period_days": 10.0, "radius_rearth": 1.0},
        ]
        r = summarize_system_architecture(planets)
        s = format_system_arch(r)
        assert r.architecture_class in s
