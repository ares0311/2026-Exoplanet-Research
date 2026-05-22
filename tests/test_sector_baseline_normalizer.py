"""Tests for sector_baseline_normalizer.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from sector_baseline_normalizer import (
    format_baseline_norm_result,
    normalize_sector_baselines,
)


def _two_sector_lc(n=100, offset=0.05):
    time = [i * 0.02 for i in range(n)]
    flux = [1.0 + offset if i < n // 2 else 1.0 for i in range(n)]
    sector_ids = [1 if i < n // 2 else 2 for i in range(n)]
    return time, flux, sector_ids


class TestNormalizeSectorBaselines:
    def test_result_frozen(self):
        t, f, s = _two_sector_lc()
        r = normalize_sector_baselines(t, f, s)
        try:
            r.n_sectors = 99  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except (AttributeError, TypeError):
            pass

    def test_sector_norm_frozen(self):
        t, f, s = _two_sector_lc()
        r = normalize_sector_baselines(t, f, s)
        if r.sector_results:
            sr = r.sector_results[0]
            try:
                sr.offset = 99.0  # type: ignore[misc]
                raise AssertionError("Should be frozen")
            except (AttributeError, TypeError):
                pass

    def test_empty_inputs_invalid(self):
        r = normalize_sector_baselines([], [], [])
        assert r.flag == "INVALID"

    def test_mismatched_lengths_invalid(self):
        r = normalize_sector_baselines([1.0, 2.0], [1.0], [1])
        assert r.flag == "INVALID"

    def test_invalid_method(self):
        t, f, s = _two_sector_lc()
        r = normalize_sector_baselines(t, f, s, method="invalid")
        assert r.flag == "INVALID"

    def test_additive_removes_offset(self):
        t, f, s = _two_sector_lc(offset=0.1)
        r = normalize_sector_baselines(t, f, s, method="additive")
        assert r.flag in ("OK", "SINGLE_SECTOR")
        # After normalisation, medians should be similar
        n = len(r.normalized_flux)
        s1 = sorted(r.normalized_flux[: n // 2])[n // 4]
        s2 = sorted(r.normalized_flux[n // 2 :])[n // 4]
        assert abs(s1 - s2) < 0.02

    def test_multiplicative_method(self):
        t, f, s = _two_sector_lc(offset=0.1)
        r = normalize_sector_baselines(t, f, s, method="multiplicative")
        assert r.flag in ("OK", "SINGLE_SECTOR")

    def test_n_sectors_matches_unique_ids(self):
        t, f, s = _two_sector_lc()
        r = normalize_sector_baselines(t, f, s)
        assert r.n_sectors == 2

    def test_single_sector_flag(self):
        t = [0.02 * i for i in range(50)]
        f = [1.0] * 50
        s = [1] * 50
        r = normalize_sector_baselines(t, f, s)
        assert r.flag == "SINGLE_SECTOR"

    def test_normalized_flux_same_length(self):
        t, f, s = _two_sector_lc()
        r = normalize_sector_baselines(t, f, s)
        assert len(r.normalized_flux) == len(f)

    def test_sector_results_length(self):
        t, f, s = _two_sector_lc()
        r = normalize_sector_baselines(t, f, s)
        assert len(r.sector_results) == 2

    def test_custom_reference(self):
        t, f, s = _two_sector_lc()
        r = normalize_sector_baselines(t, f, s, reference=2.0)
        assert r.flag in ("OK", "SINGLE_SECTOR")

    def test_format_returns_string(self):
        t, f, s = _two_sector_lc()
        r = normalize_sector_baselines(t, f, s)
        out = format_baseline_norm_result(r)
        assert isinstance(out, str)
        assert "Normalisation" in out or "Baseline" in out
