"""Tests for Skills/planet_atmosphere_classifier.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from planet_atmosphere_classifier import AtmosphereClassResult, classify_atmosphere, format_atmosphere_class


class TestAtmosphereClassResult:
    def test_dataclass_fields(self):
        r = AtmosphereClassResult(class_label="rocky_no_atm", confidence="high",
                                  rationale="test", flag="OK")
        assert r.class_label == "rocky_no_atm"
        assert r.flag == "OK"

    def test_frozen(self):
        r = AtmosphereClassResult(class_label="rocky_no_atm", confidence="high",
                                  rationale="test", flag="OK")
        try:
            r.class_label = "gas_giant"
            assert False
        except Exception:
            pass


class TestClassifyAtmosphere:
    def test_rocky_no_atm_hot(self):
        # R < 1.5, S > 10 → rocky_no_atm
        r = classify_atmosphere(1.0, 15.0)
        assert r.class_label == "rocky_no_atm"
        assert r.confidence == "high"

    def test_rocky_thin_atm(self):
        # R < 1.5, S <= 10 → rocky_thin_atm
        r = classify_atmosphere(1.0, 1.0)
        assert r.class_label == "rocky_thin_atm"

    def test_water_world(self):
        # 1.5 <= R < 2.5, S < 4 → water_world
        r = classify_atmosphere(2.0, 1.0)
        assert r.class_label == "water_world"

    def test_sub_neptune_high_insolation(self):
        # 1.5 <= R < 2.5, S >= 4 → sub_neptune
        r = classify_atmosphere(2.0, 5.0)
        assert r.class_label == "sub_neptune"

    def test_sub_neptune_large(self):
        # 2.5 <= R < 4.0 → sub_neptune regardless of S
        r = classify_atmosphere(3.0, 1.0)
        assert r.class_label == "sub_neptune"

    def test_neptune_like(self):
        # 4.0 <= R < 8.0 → neptune_like
        r = classify_atmosphere(5.0, 1.0)
        assert r.class_label == "neptune_like"
        assert r.confidence == "high"

    def test_gas_giant(self):
        # R >= 8.0 → gas_giant
        r = classify_atmosphere(10.0, 1.0)
        assert r.class_label == "gas_giant"
        assert r.confidence == "high"

    def test_flag_always_ok(self):
        for r_val in [1.0, 2.0, 3.5, 6.0, 12.0]:
            r = classify_atmosphere(r_val, 1.0)
            assert r.flag == "OK"

    def test_rationale_non_empty(self):
        r = classify_atmosphere(1.0, 1.0)
        assert len(r.rationale) > 0

    def test_boundary_1_5(self):
        # Exactly 1.5 → should be water_world or sub_neptune (not rocky)
        r = classify_atmosphere(1.5, 1.0)
        assert r.class_label != "rocky_no_atm"
        assert r.class_label != "rocky_thin_atm"

    def test_boundary_4_0(self):
        r = classify_atmosphere(4.0, 1.0)
        assert r.class_label == "neptune_like"

    def test_boundary_8_0(self):
        r = classify_atmosphere(8.0, 1.0)
        assert r.class_label == "gas_giant"


class TestFormatAtmosphereClass:
    def test_returns_string(self):
        r = classify_atmosphere(1.0, 1.0)
        s = format_atmosphere_class(r)
        assert isinstance(s, str)

    def test_contains_class_label(self):
        r = classify_atmosphere(1.0, 1.0)
        s = format_atmosphere_class(r)
        assert r.class_label in s
