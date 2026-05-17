"""Tests for Skills.lightcurve_cache."""
from __future__ import annotations

from pathlib import Path

from Skills.lightcurve_cache import CachedLC, LightcurveCache, cache_key


class TestCacheKey:
    def test_includes_tic_id(self) -> None:
        assert "12345" in cache_key(12345, "TESS")

    def test_includes_mission_lower(self) -> None:
        assert "tess" in cache_key(1, "TESS")

    def test_all_sectors_marker(self) -> None:
        assert "all" in cache_key(1, "TESS", None)

    def test_sector_included(self) -> None:
        assert "7" in cache_key(1, "TESS", 7)

    def test_different_sectors_differ(self) -> None:
        assert cache_key(1, "TESS", 1) != cache_key(1, "TESS", 2)


class TestLightcurveCache:
    def test_save_creates_file(self, tmp_path: Path) -> None:
        cache = LightcurveCache(tmp_path / "lc")
        cache.save(1, "TESS", [1.0, 2.0], [1.0, 1.0])
        assert len(cache.list_entries()) == 1

    def test_load_returns_none_when_absent(self, tmp_path: Path) -> None:
        cache = LightcurveCache(tmp_path / "lc")
        assert cache.load(99, "TESS") is None

    def test_roundtrip_time_flux(self, tmp_path: Path) -> None:
        cache = LightcurveCache(tmp_path / "lc")
        cache.save(1, "TESS", [1.0, 2.0, 3.0], [0.9, 1.0, 1.1])
        lc = cache.load(1, "TESS")
        assert lc is not None
        assert lc.time == [1.0, 2.0, 3.0]
        assert lc.flux == [0.9, 1.0, 1.1]

    def test_flux_err_roundtrip(self, tmp_path: Path) -> None:
        cache = LightcurveCache(tmp_path / "lc")
        cache.save(1, "TESS", [1.0], [1.0], flux_err=[0.001])
        lc = cache.load(1, "TESS")
        assert lc is not None
        assert lc.flux_err == [0.001]

    def test_contains_after_save(self, tmp_path: Path) -> None:
        cache = LightcurveCache(tmp_path / "lc")
        assert not cache.contains(5, "TESS")
        cache.save(5, "TESS", [1.0], [1.0])
        assert cache.contains(5, "TESS")

    def test_delete_returns_true_when_present(self, tmp_path: Path) -> None:
        cache = LightcurveCache(tmp_path / "lc")
        cache.save(1, "TESS", [1.0], [1.0])
        assert cache.delete(1, "TESS") is True

    def test_delete_returns_false_when_absent(self, tmp_path: Path) -> None:
        cache = LightcurveCache(tmp_path / "lc")
        assert cache.delete(99, "TESS") is False

    def test_list_entries_empty_when_no_dir(self, tmp_path: Path) -> None:
        cache = LightcurveCache(tmp_path / "nonexistent")
        assert cache.list_entries() == []

    def test_clear_removes_all(self, tmp_path: Path) -> None:
        cache = LightcurveCache(tmp_path / "lc")
        cache.save(1, "TESS", [1.0], [1.0])
        cache.save(2, "TESS", [1.0], [1.0])
        n = cache.clear()
        assert n == 2
        assert cache.list_entries() == []

    def test_size_bytes_positive_after_save(self, tmp_path: Path) -> None:
        cache = LightcurveCache(tmp_path / "lc")
        cache.save(1, "TESS", list(range(100)), list(range(100)))
        assert cache.size_bytes() > 0

    def test_sector_scoped_entries_independent(self, tmp_path: Path) -> None:
        cache = LightcurveCache(tmp_path / "lc")
        cache.save(1, "TESS", [1.0], [1.0], sector=1)
        cache.save(1, "TESS", [2.0], [2.0], sector=2)
        assert len(cache.list_entries()) == 2

    def test_returns_cached_lc_instance(self, tmp_path: Path) -> None:
        cache = LightcurveCache(tmp_path / "lc")
        cache.save(1, "TESS", [1.0], [1.0])
        lc = cache.load(1, "TESS")
        assert isinstance(lc, CachedLC)
