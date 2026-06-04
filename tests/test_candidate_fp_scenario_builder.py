"""Tests for Skills/candidate_fp_scenario_builder.py."""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from candidate_fp_scenario_builder import build_fp_scenarios, format_fp_scenarios


class TestBuildFpScenarios:
    def test_basic_call(self) -> None:
        r = build_fp_scenarios({})
        assert r.flag == "OK"
        assert r.n_scenarios == 5

    def test_weights_sum_to_one(self) -> None:
        r = build_fp_scenarios({})
        total = sum(s.probability_weight for s in r.scenarios)
        assert abs(total - 1.0) < 1e-3

    def test_centroid_shift_beb_dominant(self) -> None:
        r = build_fp_scenarios({"centroid_motion_arcsec": 2.0})
        assert r.dominant_scenario == "BACKGROUND_EB"

    def test_centroid_small_beb_ruled_out(self) -> None:
        r = build_fp_scenarios({"centroid_motion_arcsec": 0.1})
        beb = next(s for s in r.scenarios if s.name == "BACKGROUND_EB")
        assert beb.ruled_out

    def test_ruwe_increases_heb(self) -> None:
        r_low = build_fp_scenarios({"ruwe": 1.0})
        r_high = build_fp_scenarios({"ruwe": 2.0})
        heb_low = next(s for s in r_low.scenarios if s.name == "HIERARCHICAL_TRIPLE")
        heb_high = next(s for s in r_high.scenarios if s.name == "HIERARCHICAL_TRIPLE")
        assert heb_high.probability_weight >= heb_low.probability_weight

    def test_odd_even_sigma_grazing_eb(self) -> None:
        r = build_fp_scenarios({"odd_even_sigma": 5.0})
        grazing = next(s for s in r.scenarios if s.name == "GRAZING_EB")
        assert grazing.probability_weight > 0.1

    def test_high_snr_artefact_ruled_out(self) -> None:
        r = build_fp_scenarios({"snr": 20.0})
        artefact = next(s for s in r.scenarios if s.name == "INSTRUMENTAL_ARTEFACT")
        assert artefact.ruled_out

    def test_n_ruled_out_count(self) -> None:
        r = build_fp_scenarios({"centroid_motion_arcsec": 0.1, "snr": 20.0})
        assert r.n_ruled_out >= 1

    def test_dominant_scenario_in_list(self) -> None:
        r = build_fp_scenarios({})
        names = [s.name for s in r.scenarios]
        assert r.dominant_scenario in names

    def test_all_weights_finite(self) -> None:
        r = build_fp_scenarios({"depth_ppm": 5000.0, "secondary_snr": 2.0})
        for s in r.scenarios:
            assert math.isfinite(s.probability_weight)

    def test_format_output(self) -> None:
        r = build_fp_scenarios({"centroid_motion_arcsec": 0.5})
        s = format_fp_scenarios(r)
        assert "|" in s
        assert "scenario" in s.lower() or "BACKGROUND" in s

    def test_deep_transit_companion_increases(self) -> None:
        r_shallow = build_fp_scenarios({"depth_ppm": 500.0})
        r_deep = build_fp_scenarios({"depth_ppm": 15000.0})
        comp_shallow = next(
            s for s in r_shallow.scenarios if s.name == "STELLAR_COMPANION_TRANSIT"
        )
        comp_deep = next(
            s for s in r_deep.scenarios if s.name == "STELLAR_COMPANION_TRANSIT"
        )
        assert comp_deep.probability_weight >= comp_shallow.probability_weight
