"""Cross-match a candidate against TESS certified / confirmed planet ephemerides."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TcertMatch:
    planet_name: str
    catalog_period_days: float
    catalog_epoch_bjd: float
    period_diff_pct: float    # |P_cand - P_cat| / P_cat * 100
    epoch_offset_days: float  # minimum O-C after accounting for integer cycles
    is_alias: bool            # matched via P/2, 2P, etc.
    alias_ratio: str          # e.g. "1:1", "1:2", "2:1"


@dataclass(frozen=True)
class TcertCrossmatchResult:
    tic_id: str
    period_days: float
    n_catalog_checked: int
    n_matches: int
    best_match: TcertMatch | None
    known_planet: bool     # True if any 1:1 match within tolerances
    flag: str


# Small built-in reference table of well-characterised TESS planets
# (TIC_ID, planet_name, period_days, epoch_bjd)
_TCERT_TABLE: list[tuple[str, str, float, float]] = [
    ("150428135", "TOI-700 d", 37.4241, 2458867.390),
    ("150428135", "TOI-700 b", 9.9770, 2458873.220),
    ("150428135", "TOI-700 c", 16.0510, 2458861.530),
    ("261136679", "TOI-1338 b", 14.6108, 2458517.603),
    ("460205581", "TOI-2257 b", 35.1960, 2458850.370),
    ("272369ad", "TOI-125 b", 4.6547, 2458504.900),
    ("52368076", "TOI-402 b", 4.7566, 2458379.100),
    ("52368076", "TOI-402 c", 17.178, 2458369.900),
    ("55652896", "TOI-500 b", 13.6826, 2458740.390),
    ("441798995", "TOI-1452 b", 11.0628, 2459014.621),
]


def _min_oc(t_cand: float, t_cat: float, period: float) -> float:
    if period <= 0:
        return abs(t_cand - t_cat)
    diff = t_cand - t_cat
    n = round(diff / period)
    return abs(diff - n * period)


def crossmatch_tcert(
    tic_id: str,
    period_days: float,
    epoch_bjd: float | None = None,
    period_tol_pct: float = 1.0,
    epoch_tol_days: float = 0.5,
    check_aliases: bool = True,
    catalog: list[tuple[str, str, float, float]] | None = None,
) -> TcertCrossmatchResult:
    """
    Cross-match candidate period/epoch against certified TESS planet table.

    Parameters
    ----------
    tic_id:           Target TIC identifier.
    period_days:      Candidate orbital period in days.
    epoch_bjd:        Candidate transit epoch (BJD). If None, epoch match skipped.
    period_tol_pct:   Period match tolerance in percent (default 1%).
    epoch_tol_days:   Epoch match tolerance in days (default 0.5 d).
    check_aliases:    Also test P/2, 2P, P/3, 3P harmonics.
    catalog:          Override the built-in table (list of (tic_id, name, P, T0)).
    """
    if not math.isfinite(period_days) or period_days <= 0:
        return TcertCrossmatchResult(
            tic_id=str(tic_id), period_days=period_days,
            n_catalog_checked=0, n_matches=0, best_match=None,
            known_planet=False, flag="INVALID_PERIOD",
        )

    table = catalog if catalog is not None else _TCERT_TABLE
    tic_str = str(tic_id).strip()

    alias_ratios = [(1, 1)]
    if check_aliases:
        alias_ratios += [(1, 2), (2, 1), (1, 3), (3, 1)]

    matches: list[TcertMatch] = []
    n_checked = 0

    for row_tic, name, cat_period, cat_epoch in table:
        if str(row_tic).strip() != tic_str:
            continue
        n_checked += 1

        for p_num, p_den in alias_ratios:
            test_period = cat_period * p_num / p_den
            diff_pct = abs(period_days - test_period) / test_period * 100.0
            if diff_pct > period_tol_pct:
                continue

            ep_offset = 0.0
            if epoch_bjd is not None and math.isfinite(epoch_bjd):
                ep_offset = _min_oc(epoch_bjd, cat_epoch, test_period)
                if ep_offset > epoch_tol_days:
                    continue

            is_alias = (p_num, p_den) != (1, 1)
            ratio_str = f"{p_num}:{p_den}"
            matches.append(TcertMatch(
                planet_name=name,
                catalog_period_days=cat_period,
                catalog_epoch_bjd=cat_epoch,
                period_diff_pct=round(diff_pct, 4),
                epoch_offset_days=round(ep_offset, 6),
                is_alias=is_alias,
                alias_ratio=ratio_str,
            ))

    best: TcertMatch | None = None
    if matches:
        best = min(matches, key=lambda m: (m.period_diff_pct, m.epoch_offset_days))

    known = best is not None and not best.is_alias

    return TcertCrossmatchResult(
        tic_id=tic_str,
        period_days=period_days,
        n_catalog_checked=n_checked,
        n_matches=len(matches),
        best_match=best,
        known_planet=known,
        flag="OK",
    )


def format_tcert_result(r: TcertCrossmatchResult) -> str:
    if r.flag != "OK":
        return f"No result (flag: {r.flag}).\n"
    lines = [
        "| Parameter | Value |\n|---|---|",
        f"| TIC ID | {r.tic_id} |",
        f"| Candidate period (d) | {r.period_days:.6f} |",
        f"| Catalog entries checked | {r.n_catalog_checked} |",
        f"| Matches found | {r.n_matches} |",
        f"| Known planet | {r.known_planet} |",
    ]
    if r.best_match is not None:
        m = r.best_match
        lines += [
            f"| Best match | {m.planet_name} |",
            f"| Catalog period (d) | {m.catalog_period_days:.6f} |",
            f"| Period diff (%) | {m.period_diff_pct:.4f} |",
            f"| Epoch offset (d) | {m.epoch_offset_days:.6f} |",
            f"| Alias | {m.is_alias} ({m.alias_ratio}) |",
        ]
    return "\n".join(lines)


def _cli() -> int:
    p = argparse.ArgumentParser(description="Cross-match against TESS certified planets.")
    p.add_argument("tic_id")
    p.add_argument("period_days", type=float)
    p.add_argument("--epoch-bjd", type=float, default=None)
    p.add_argument("--period-tol-pct", type=float, default=1.0)
    args = p.parse_args()
    r = crossmatch_tcert(
        args.tic_id, args.period_days,
        epoch_bjd=args.epoch_bjd,
        period_tol_pct=args.period_tol_pct,
    )
    print(format_tcert_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
