"""Estimate total observable transits over a ground campaign.

Public API:
    YieldResult  -- frozen dataclass
    estimate_observation_yield(period_days, epoch_bjd, campaign_start_bjd, campaign_end_bjd,
                                nightly_window_hours) -> YieldResult
    format_yield_result(result) -> str
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class YieldResult:
    period_days: float
    n_transits_in_campaign: int
    n_observable: int
    campaign_days: float
    observable_fraction: float
    flag: str


def estimate_observation_yield(
    period_days: float,
    epoch_bjd: float,
    campaign_start_bjd: float,
    campaign_end_bjd: float,
    nightly_window_hours: float,
    transit_duration_hours: float = 2.0,
) -> YieldResult:
    if period_days <= 0:
        return YieldResult(
            period_days=period_days, n_transits_in_campaign=0, n_observable=0,
            campaign_days=0.0, observable_fraction=0.0, flag="INVALID_PERIOD",
        )
    if campaign_end_bjd <= campaign_start_bjd:
        return YieldResult(
            period_days=period_days, n_transits_in_campaign=0, n_observable=0,
            campaign_days=0.0, observable_fraction=0.0, flag="INVALID_CAMPAIGN_WINDOW",
        )
    if nightly_window_hours < 0 or nightly_window_hours > 24:
        return YieldResult(
            period_days=period_days, n_transits_in_campaign=0, n_observable=0,
            campaign_days=0.0, observable_fraction=0.0, flag="INVALID_NIGHTLY_WINDOW",
        )
    if transit_duration_hours <= 0:
        return YieldResult(
            period_days=period_days, n_transits_in_campaign=0, n_observable=0,
            campaign_days=0.0, observable_fraction=0.0, flag="INVALID_TRANSIT_DURATION",
        )
    campaign_days = campaign_end_bjd - campaign_start_bjd
    # Find first transit at or after campaign start
    if epoch_bjd <= campaign_start_bjd:
        n0 = math.ceil((campaign_start_bjd - epoch_bjd) / period_days)
    else:
        # Next transit after campaign start
        n0 = -math.floor((epoch_bjd - campaign_start_bjd) / period_days)
    transits = []
    n = n0
    while True:
        t_mid = epoch_bjd + n * period_days
        if t_mid > campaign_end_bjd:
            break
        if t_mid >= campaign_start_bjd:
            transits.append(t_mid)
        n += 1
    n_total = len(transits)
    # Observable: transit center falls within a night window
    # Approximate: nightly window fraction = nightly_window_hours / 24
    # A transit is fully observable if its midpoint falls within a nightly window
    # and transit fits: nightly_window_hours >= transit_duration_hours
    window_fraction = nightly_window_hours / 24.0
    if nightly_window_hours >= transit_duration_hours:
        n_observable = round(n_total * window_fraction)
    else:
        # Transit cannot fit in window
        n_observable = 0
    n_observable = min(n_observable, n_total)
    obs_fraction = n_observable / n_total if n_total > 0 else 0.0
    return YieldResult(
        period_days=period_days,
        n_transits_in_campaign=n_total,
        n_observable=n_observable,
        campaign_days=campaign_days,
        observable_fraction=obs_fraction,
        flag="OK",
    )


def format_yield_result(result: YieldResult) -> str:
    lines = [
        "## Observation Yield Estimate",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Period (days) | {result.period_days:.3f} |",
        f"| Campaign (days) | {result.campaign_days:.1f} |",
        f"| Transits in Campaign | {result.n_transits_in_campaign} |",
        f"| Observable Transits | {result.n_observable} |",
        f"| Observable Fraction | {result.observable_fraction:.3f} |",
        f"| Flag | {result.flag} |",
    ]
    return "\n".join(lines)


def _cli() -> int:
    parser = argparse.ArgumentParser(
        description="Estimate observable transit yield from ground campaign.",
    )
    parser.add_argument("period_days", type=float)
    parser.add_argument("epoch_bjd", type=float)
    parser.add_argument("campaign_start_bjd", type=float)
    parser.add_argument("campaign_end_bjd", type=float)
    parser.add_argument("nightly_window_hours", type=float)
    parser.add_argument("--transit-duration-hours", type=float, default=2.0)
    args = parser.parse_args()
    result = estimate_observation_yield(
        args.period_days, args.epoch_bjd, args.campaign_start_bjd,
        args.campaign_end_bjd, args.nightly_window_hours, args.transit_duration_hours,
    )
    print(format_yield_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
