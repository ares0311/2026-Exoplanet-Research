"""Network-boundary fixtures for the deterministic Hunter production acceptance.

Python imports ``sitecustomize`` during installed-console-script startup. This
module is inert unless the acceptance runner supplies its private state path.
The production shell and business modules contain no fixture switch.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

STATE_ENV = "EXO_HUNTER_ACCEPTANCE_STATE"
state_path_text = os.environ.get(STATE_ENV)


if state_path_text:
    import lightkurve as lk
    import numpy as np
    from Skills import star_scanner

    from exo_toolkit import hunter_cli
    from exo_toolkit.cli import run_pipeline as production_run_pipeline
    from exo_toolkit.fetch import FetchProvenance, FetchResult

    state_path = Path(state_path_text)
    original_inspect_target_products = star_scanner.inspect_target_products

    class FixtureTable(list[dict[str, Any]]):
        """Minimal table shape consumed by the production product parser."""

        colnames = (
            "dataURI",
            "size",
            "sequence_number",
            "productFilename",
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
        rows = [
            {
                "ID": 800_000 + index,
                "HIP": 74_981 if index == 2 else None,
                "objType": "STAR",
                "Tmag": 12.5,
                "Teff": 4_500.0,
                "contratio": 0.0,
                "rad": 0.7,
                "ra": 10.0 + index / 100.0,
                "dec": -20.0,
                "d": 25.0 + index,
                "e_d": 0.2,
                "distFlag": "fixture-parallax",
                "version": "acceptance-tic-v1",
            }
            for index in range(1, 201)
        ]
        rows.append(
            {
                "ID": 999_999,
                "HIP": None,
                "objType": "STAR",
                "Tmag": 12.5,
                "Teff": 4_500.0,
                "contratio": 0.001,
                "rad": 0.7,
                "ra": 42.0,
                "dec": -17.0,
                "d": 19.0,
                "e_d": 0.1,
                "distFlag": "fixture-parallax",
                "version": "acceptance-tic-v1",
            }
        )
        return rows

    def fixture_product_search(
        target_id: str,
        *,
        mission: str,
        author: str,
        exptime: str,
    ) -> SimpleNamespace:
        del mission, author, exptime
        tic_id = int(target_id.split()[-1])
        sectors = range(1, 7) if tic_id == 999_999 else (1,)
        table = FixtureTable(
            {
                "dataURI": f"mast:fixture/{tic_id}/sector-{sector}.fits",
                "size": 1_000_000,
                "sequence_number": sector,
                "productFilename": f"tic-{tic_id}-s{sector:02d}.fits",
            }
            for sector in sectors
        )
        return SimpleNamespace(table=table)

    def fixture_inspect_target_products(
        target: dict[str, Any],
        *,
        mission: str,
        pipeline: str,
        exptime: str,
    ) -> dict[str, Any]:
        return original_inspect_target_products(
            target,
            mission=mission,
            pipeline=pipeline,
            exptime=exptime,
            search_fn=fixture_product_search,
        )

    def load_state() -> dict[str, Any]:
        if not state_path.is_file():
            return {"fetch_attempts": {}, "events": []}
        return json.loads(state_path.read_text(encoding="utf-8"))

    def save_state(state: dict[str, Any]) -> None:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(state, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(state_path)

    def fixture_fetch(
        target_id: str,
        mission: str,
        **_: Any,
    ) -> FetchResult:
        state = load_state()
        attempts = state.setdefault("fetch_attempts", {})
        attempt = int(attempts.get(target_id, 0)) + 1
        attempts[target_id] = attempt
        state.setdefault("events", []).append(
            {"event": "fetch", "target_id": target_id, "attempt": attempt}
        )
        save_state(state)
        if target_id == "TIC 999999" and attempt == 1:
            raise RuntimeError("acceptance fixture forced first-attempt acquisition failure")

        n_cadences = 1_000
        time = np.linspace(2_458_000.0, 2_458_027.0, n_cadences)
        period_days = 3.0
        epoch_bjd = 2_458_001.0
        duration_days = 2.0 / 24.0
        flux = 1.0 + 1e-4 * np.sin(np.arange(n_cadences, dtype=float) * 0.37)
        phase = ((time - epoch_bjd + 0.5 * period_days) % period_days) - (
            0.5 * period_days
        )
        flux[np.abs(phase) <= duration_days / 2.0] -= 0.003
        light_curve = lk.LightCurve(
            time=time,
            flux=flux,
            flux_err=np.full(n_cadences, 1e-4),
        )
        return FetchResult(
            light_curve=light_curve,
            provenance=FetchProvenance(
                target_id=target_id,
                mission=mission,
                sectors_or_quarters=(1,),
                cadence_seconds=1_800.0,
                pipeline="QLP",
                flux_column="kspsap_flux",
                n_cadences=n_cadences,
                time_baseline_days=27.0,
                fetched_at="2026-07-29T00:00:00+00:00",
                raw_uris=(f"mast:fixture/{target_id.replace(' ', '-')}.fits",),
            ),
        )

    def acceptance_run_pipeline(
        target_id: str,
        mission: str,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        return production_run_pipeline(
            target_id,
            mission,
            **kwargs,
            period_max=10.0,
            n_durations=8,
            max_period_grid_points=2_000,
            fetch_fn=fixture_fetch,
            stellar_params_fn=lambda _target_id: {
                "stellar_radius_rsun": 0.7,
                "stellar_mass_msun": 0.7,
                "stellar_teff_k": 4_500.0,
                "contamination_ratio": 0.0,
            },
        )

    star_scanner._query_tic_criteria_page = fixture_catalog_page
    star_scanner._load_toi_tic_ids = lambda **_: set()
    star_scanner._load_ctoi_tic_ids = lambda **_: set()
    star_scanner._load_confirmed_host_tic_ids = lambda **_: frozenset()
    star_scanner._load_asassn_variable_tic_ids = (
        lambda tic_ids, **_: {800_001} & set(tic_ids)
    )
    star_scanner.inspect_target_products = fixture_inspect_target_products
    hunter_cli.DEFAULT_POOL_SIZE = 1
    hunter_cli.run_pipeline = acceptance_run_pipeline
