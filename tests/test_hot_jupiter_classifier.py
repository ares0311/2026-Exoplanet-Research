"""Tests for Skills/hot_jupiter_classifier.py."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from hot_jupiter_classifier import classify_planet_regime, format_regime_result


class TestClassifyPlanetRegime:
    def test_hot_jupiter(self) -> None:
        r = classify_planet_regime(period_days=3.5, radius_rearth=11.0)
        assert r.flag == "OK"
        assert r.period_class == "HOT"
        assert r.radius_class == "JOVIAN"
        assert r.regime == "HOT_JUPITER"

    def test_ultra_hot_jupiter(self) -> None:
        r = classify_planet_regime(period_days=0.8, radius_rearth=13.0)
        assert r.period_class == "ULTRA_HOT"
        assert r.radius_class == "JOVIAN"
        assert r.regime == "ULTRA_HOT_JUPITER"

    def test_warm_jupiter(self) -> None:
        r = classify_planet_regime(period_days=50.0, radius_rearth=10.0)
        assert r.period_class == "WARM"
        assert r.radius_class == "JOVIAN"
        assert r.regime == "WARM_JUPITER"

    def test_super_earth(self) -> None:
        r = classify_planet_regime(period_days=10.0, radius_rearth=1.2)
        assert r.radius_class == "SUPER_EARTH"

    def test_sub_neptune(self) -> None:
        r = classify_planet_regime(period_days=20.0, radius_rearth=2.5)
        assert r.radius_class == "SUB_NEPTUNE"

    def test_neptunian(self) -> None:
        r = classify_planet_regime(period_days=30.0, radius_rearth=4.5)
        assert r.radius_class == "NEPTUNIAN"

    def test_super_jovian(self) -> None:
        r = classify_planet_regime(period_days=5.0, radius_rearth=16.0)
        assert r.radius_class == "SUPER_JOVIAN"

    def test_long_period(self) -> None:
        r = classify_planet_regime(period_days=400.0, radius_rearth=11.0)
        assert r.period_class == "LONG_PERIOD"

    def test_cold_period(self) -> None:
        r = classify_planet_regime(period_days=200.0, radius_rearth=2.0)
        assert r.period_class == "COLD"

    def test_invalid_period(self) -> None:
        r = classify_planet_regime(period_days=-1.0, radius_rearth=5.0)
        assert r.flag == "INVALID_PERIOD"

    def test_invalid_radius(self) -> None:
        r = classify_planet_regime(period_days=5.0, radius_rearth=0.0)
        assert r.flag == "INVALID_RADIUS"

    def test_format_output(self) -> None:
        r = classify_planet_regime(period_days=3.5, radius_rearth=11.0)
        s = format_regime_result(r)
        assert "|" in s
        assert "HOT_JUPITER" in s
