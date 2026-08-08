"""Network-boundary fixture for the frozen Phase 3 installed-CLI gate.

The fixture replaces only the remote TIC and MAST product calls.  It does not
replace EXO-Hunter's selector, eligibility, ranking, persistence, or command
routing.  Python imports this module only when the gate supplies its private
bootstrap directory on ``PYTHONPATH``.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

if os.environ.get("EXO_HUNTER_PHASE3_FIXTURE") == "1":
    from Skills import star_scanner

    class FixtureTable(list[dict[str, Any]]):
        colnames = ("dataURI", "size", "sequence_number", "productFilename")

    rows = (
        (910_001, 60_759),
        (910_002, 61_099),
        (910_003, 3_419),
        (910_004, 88_888),
    )

    def fixture_catalog_page(
        page: int,
        pagesize: int,
        tmag_range: tuple[float, float],
        *,
        query_timeout_seconds: float,
    ) -> list[dict[str, Any]]:
        del pagesize, tmag_range, query_timeout_seconds
        if page != 1:
            return []
        return [
            {
                "ID": tic_id,
                "HIP": hip_id,
                "objType": "STAR",
                "Tmag": 12.5,
                "Teff": 4_500.0,
                "contratio": 0.0,
                "rad": 0.7,
                "ra": 10.0 + index / 100.0,
                "dec": -20.0,
                "d": 20.0 + index,
                "e_d": 0.1,
                "distFlag": "phase3-fixture-parallax",
                "version": "phase3-tic-v1",
            }
            for index, (tic_id, hip_id) in enumerate(rows, 1)
        ]

    def fixture_product_search(
        target_id: str,
        *,
        mission: str,
        author: str,
        exptime: str,
    ) -> SimpleNamespace:
        del mission, author, exptime
        tic_id = int(target_id.split()[-1])
        return SimpleNamespace(
            table=FixtureTable(
                [
                    {
                        "dataURI": f"mast:phase3/{tic_id}/sector-1.fits",
                        "size": 1_000_000,
                        "sequence_number": 1,
                        "productFilename": f"tic-{tic_id}-s01.fits",
                    }
                ]
            )
        )

    original_inspect = star_scanner.inspect_target_products

    def fixture_inspect_target_products(
        target: dict[str, Any],
        *,
        mission: str,
        pipeline: str,
        exptime: str,
    ) -> dict[str, Any]:
        return original_inspect(
            target,
            mission=mission,
            pipeline=pipeline,
            exptime=exptime,
            search_fn=fixture_product_search,
        )

    star_scanner._query_tic_criteria_page = fixture_catalog_page
    star_scanner._load_toi_tic_ids = lambda **_: set()
    star_scanner._load_ctoi_tic_ids = lambda **_: set()
    star_scanner._load_confirmed_host_tic_ids = lambda **_: frozenset()
    star_scanner._load_asassn_variable_tic_ids = lambda *_args, **_kwargs: set()
    star_scanner.inspect_target_products = fixture_inspect_target_products
