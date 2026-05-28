"""Tests for Skills/snippet_cache_manager.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from snippet_cache_manager import SnippetCacheManager, SnippetCacheStats, format_cache_stats


def _write_snippet(cache_dir: Path, tic_id: int, period: float) -> Path:
    p = cache_dir / f"{tic_id}_{period:.6f}.json"
    p.write_text(json.dumps({"tic_id": tic_id, "period_days": period, "snippet": [1.0] * 8}))
    return p


def test_empty_cache_stats(tmp_path):
    mgr = SnippetCacheManager(tmp_path)
    s = mgr.stats()
    assert s.flag == "EMPTY"


def test_stats_after_write(tmp_path):
    _write_snippet(tmp_path, 100, 5.0)
    mgr = SnippetCacheManager(tmp_path)
    s = mgr.stats()
    assert s.flag == "OK"
    assert s.n_snippets == 1


def test_stats_n_tic_ids(tmp_path):
    _write_snippet(tmp_path, 100, 5.0)
    _write_snippet(tmp_path, 100, 10.0)
    _write_snippet(tmp_path, 200, 5.0)
    mgr = SnippetCacheManager(tmp_path)
    s = mgr.stats()
    assert s.n_tic_ids == 2


def test_contains_true(tmp_path):
    _write_snippet(tmp_path, 100, 5.0)
    mgr = SnippetCacheManager(tmp_path)
    assert mgr.contains(100, 5.0)


def test_contains_false(tmp_path):
    mgr = SnippetCacheManager(tmp_path)
    assert not mgr.contains(999, 5.0)


def test_contains_within_rtol(tmp_path):
    _write_snippet(tmp_path, 100, 5.0)
    mgr = SnippetCacheManager(tmp_path)
    assert mgr.contains(100, 5.005, period_rtol=0.01)


def test_contains_outside_rtol(tmp_path):
    _write_snippet(tmp_path, 100, 5.0)
    mgr = SnippetCacheManager(tmp_path)
    assert not mgr.contains(100, 6.0, period_rtol=0.01)


def test_prune_old_files(tmp_path):
    import os
    import time
    p = _write_snippet(tmp_path, 100, 5.0)
    old_mtime = time.time() - 40 * 86400  # 40 days ago
    os.utime(p, (old_mtime, old_mtime))
    mgr = SnippetCacheManager(tmp_path)
    n = mgr.prune(max_age_days=30.0)
    assert n == 1
    assert not p.exists()


def test_prune_keeps_recent(tmp_path):
    _write_snippet(tmp_path, 100, 5.0)
    mgr = SnippetCacheManager(tmp_path)
    n = mgr.prune(max_age_days=30.0)
    assert n == 0


def test_export_manifest(tmp_path):
    _write_snippet(tmp_path, 100, 5.0)
    _write_snippet(tmp_path, 200, 10.0)
    mgr = SnippetCacheManager(tmp_path)
    out = tmp_path / "manifest.json"
    n = mgr.export_manifest(out)
    assert n == 2
    entries = json.loads(out.read_text())
    assert len(entries) == 2


def test_format_stats_string(tmp_path):
    _write_snippet(tmp_path, 100, 5.0)
    mgr = SnippetCacheManager(tmp_path)
    s = mgr.stats()
    text = format_cache_stats(s)
    assert isinstance(text, str)
    assert "Cache" in text


def test_missing_dir_empty(tmp_path):
    mgr = SnippetCacheManager(tmp_path / "nonexistent")
    s = mgr.stats()
    assert s.flag == "EMPTY"
    assert s.n_snippets == 0
