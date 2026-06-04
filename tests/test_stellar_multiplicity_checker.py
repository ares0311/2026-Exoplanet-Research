"""Tests for Skills/stellar_multiplicity_checker.py."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from stellar_multiplicity_checker import (
    check_stellar_multiplicity,
    format_multiplicity_result,
)


class TestCheckStellarMultiplicity:
    def test_no_diagnostics(self) -> None:
        r = check_stellar_multiplicity()
        assert r.flag == "NO_DIAGNOSTICS"
        assert r.verdict == "SINGLE"

    def test_high_ruwe_likely_multiple(self) -> None:
        r = check_stellar_multiplicity(ruwe=3.0)
        assert r.ruwe_flag is True
        assert r.verdict in ("POSSIBLE_MULTIPLE", "LIKELY_MULTIPLE")

    def test_low_ruwe_single(self) -> None:
        r = check_stellar_multiplicity(ruwe=1.0)
        assert r.ruwe_flag is False

    def test_ruwe_at_threshold(self) -> None:
        r = check_stellar_multiplicity(ruwe=1.4)
        assert r.ruwe_flag is False

    def test_close_companion_sep_flag(self) -> None:
        r = check_stellar_multiplicity(companion_sep_arcsec=5.0)
        assert r.separation_flag is True

    def test_distant_companion_no_flag(self) -> None:
        r = check_stellar_multiplicity(companion_sep_arcsec=30.0)
        assert r.separation_flag is False

    def test_small_contrast_flag(self) -> None:
        r = check_stellar_multiplicity(contrast_delta_mag=2.0)
        assert r.contrast_flag is True

    def test_large_contrast_no_flag(self) -> None:
        r = check_stellar_multiplicity(contrast_delta_mag=7.0)
        assert r.contrast_flag is False

    def test_deep_transit_evidence(self) -> None:
        r = check_stellar_multiplicity(depth_ppm=50000.0)
        assert r.multiplicity_score > 0

    def test_multiple_evidence_higher_score(self) -> None:
        r_single = check_stellar_multiplicity(ruwe=1.1)
        r_multi = check_stellar_multiplicity(
            ruwe=3.0, companion_sep_arcsec=5.0, contrast_delta_mag=2.0
        )
        assert r_multi.multiplicity_score > r_single.multiplicity_score

    def test_verdict_values(self) -> None:
        r = check_stellar_multiplicity(ruwe=2.0)
        assert r.verdict in ("SINGLE", "POSSIBLE_MULTIPLE", "LIKELY_MULTIPLE")

    def test_format_output(self) -> None:
        r = check_stellar_multiplicity(ruwe=1.8, companion_sep_arcsec=10.0)
        s = format_multiplicity_result(r)
        assert "|" in s
        assert "RUWE" in s
