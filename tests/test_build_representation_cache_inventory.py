from __future__ import annotations

import json
from pathlib import Path

from Skills.build_representation_cache_inventory import build_inventory


def _product(root: Path, tic: int, sector: int, size: int = 10) -> None:
    stem = f"tess2024000000000-s{sector:04d}-{tic:016d}-0001-s"
    directory = root / stem
    directory.mkdir(parents=True)
    (directory / f"{stem}_lc.fits").write_bytes(b"x" * size)


def test_inventory_excludes_labeled_and_live_targets(tmp_path: Path) -> None:
    cache = tmp_path / "TESS"
    _product(cache, 10, 1)
    _product(cache, 20, 2, size=20)
    _product(cache, 30, 3)
    labeled = tmp_path / "labeled.jsonl"
    labeled.write_text(json.dumps({"tic_id": 10, "label": 1}) + "\n")
    live = tmp_path / "live.json"
    live.write_text(json.dumps({"product_inventory": [{"target_id": "TIC 20"}]}))
    rows = tmp_path / "rows.jsonl"
    summary = tmp_path / "summary.json"
    reports = []

    result = build_inventory(
        cache,
        [labeled],
        live,
        rows,
        summary,
        report_fn=lambda report, path: reports.append((report, path)) or True,
    )

    inventory_rows = [json.loads(line) for line in rows.read_text().splitlines()]
    assert [row["target_id"] for row in inventory_rows] == ["TIC 30"]
    assert inventory_rows[0]["data_uri"].startswith("mast:TESS/product/")
    assert inventory_rows[0]["cache_relative_path"].endswith("_lc.fits")
    assert result["eligible_unique_tics"] == 1
    assert result["excluded_local_labeled_tics"] == 1
    assert result["excluded_live_search_tics"] == 1
    assert result["eligible_bytes"] == 10
    assert len(reports) == 1
    assert reports[0][0].items_written == 1


def test_inventory_rejects_missing_cache(tmp_path: Path) -> None:
    try:
        build_inventory(
            tmp_path / "missing",
            [],
            tmp_path / "live.json",
            tmp_path / "rows.jsonl",
            tmp_path / "summary.json",
            report_fn=lambda *_args: True,
        )
    except ValueError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("missing cache root should fail")
