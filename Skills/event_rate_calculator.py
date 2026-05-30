"""
Calculates the expected number of transit events per observing season.

Public API:
    EventRateResult       -- frozen dataclass holding event rate diagnostics
    calculate_event_rate(period_days, duration_hours, coverage_days) -> EventRateResult
    format_event_rate(result) -> str
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class EventRateResult:
    n_events: int
    coverage_days: float
    period_days: float
    duty_cycle: float
    flag: str


def calculate_event_rate(
    period_days: float,
    duration_hours: float,
    coverage_days: float,
) -> EventRateResult:
    if period_days <= 0.0:
        return EventRateResult(
            n_events=0,
            coverage_days=coverage_days,
            period_days=period_days,
            duty_cycle=0.0,
            flag="INVALID_PERIOD",
        )

    n_events = int(coverage_days / period_days)
    duty_cycle = duration_hours / 24.0 / period_days

    flag = "FEW_EVENTS" if n_events < 3 else "OK"

    return EventRateResult(
        n_events=n_events,
        coverage_days=coverage_days,
        period_days=period_days,
        duty_cycle=duty_cycle,
        flag=flag,
    )


def format_event_rate(result: EventRateResult) -> str:
    lines = [
        "## Transit Event Rate",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Period (days) | {result.period_days:.4f} |",
        f"| Coverage (days) | {result.coverage_days:.1f} |",
        f"| Expected events | {result.n_events} |",
        f"| Duty cycle | {result.duty_cycle:.5f} |",
        f"| Flag | {result.flag} |",
    ]
    return "\n".join(lines)


def _cli() -> int:
    parser = argparse.ArgumentParser(
        description="Calculate expected transit events per observing season."
    )
    parser.add_argument("period_days", type=float, help="Orbital period in days")
    parser.add_argument("duration_hours", type=float, help="Transit duration in hours")
    parser.add_argument("coverage_days", type=float, help="Total observing coverage in days")
    args = parser.parse_args()

    result = calculate_event_rate(args.period_days, args.duration_hours, args.coverage_days)
    print(format_event_rate(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
