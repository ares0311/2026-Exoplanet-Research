"""Aggregate injection-recovery completeness results across multiple targets."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CompletenessEntry:
    tic_id: str
    n_injected: int
    n_recovered: int
    recovery_rate: float
    mean_period_days: float | None
    mean_depth_ppm: float | None


@dataclass(frozen=True)
class CompletenessSummary:
    n_targets: int
    total_injected: int
    total_recovered: int
    mean_recovery_rate: float
    min_recovery_rate: float
    max_recovery_rate: float
    entries: tuple[CompletenessEntry, ...]
    flag: str


def build_completeness_summary(
    target_results: list[dict],
) -> CompletenessSummary:
    """
    Aggregate injection-recovery completeness across multiple targets.

    Each result dict should contain:
    - tic_id
    - n_injected (int)
    - n_recovered (int)
    - mean_period_days (float, optional)
    - mean_depth_ppm (float, optional)
    """
    if not target_results:
        return CompletenessSummary(
            n_targets=0, total_injected=0, total_recovered=0,
            mean_recovery_rate=float("nan"), min_recovery_rate=float("nan"),
            max_recovery_rate=float("nan"), entries=(), flag="NO_RESULTS",
        )

    entries: list[CompletenessEntry] = []
    for res in target_results:
        tic_id = str(res.get("tic_id", "unknown"))
        n_inj = int(res.get("n_injected", 0))
        n_rec = int(res.get("n_recovered", 0))
        rate = n_rec / n_inj if n_inj > 0 else float("nan")
        p = res.get("mean_period_days")
        d = res.get("mean_depth_ppm")
        entries.append(CompletenessEntry(
            tic_id=tic_id,
            n_injected=n_inj,
            n_recovered=n_rec,
            recovery_rate=round(rate, 4) if math.isfinite(rate) else float("nan"),
            mean_period_days=round(float(p), 3) if p is not None else None,
            mean_depth_ppm=round(float(d), 1) if d is not None else None,
        ))

    total_inj = sum(e.n_injected for e in entries)
    total_rec = sum(e.n_recovered for e in entries)
    rates = [e.recovery_rate for e in entries if math.isfinite(e.recovery_rate)]

    mean_rate = sum(rates) / len(rates) if rates else float("nan")
    min_rate = min(rates) if rates else float("nan")
    max_rate = max(rates) if rates else float("nan")

    return CompletenessSummary(
        n_targets=len(entries),
        total_injected=total_inj,
        total_recovered=total_rec,
        mean_recovery_rate=round(mean_rate, 4) if math.isfinite(mean_rate) else float("nan"),
        min_recovery_rate=round(min_rate, 4) if math.isfinite(min_rate) else float("nan"),
        max_recovery_rate=round(max_rate, 4) if math.isfinite(max_rate) else float("nan"),
        entries=tuple(entries),
        flag="OK",
    )


def format_completeness_summary(r: CompletenessSummary) -> str:
    if r.flag != "OK":
        return f"No results (flag: {r.flag}).\n"
    lines = [
        f"**Completeness Summary** — {r.n_targets} targets | "
        f"{r.total_recovered}/{r.total_injected} recovered "
        f"(mean {r.mean_recovery_rate:.1%})\n",
        "| TIC ID | Injected | Recovered | Rate | Period (d) | Depth (ppm) |",
        "|---|---|---|---|---|---|",
    ]
    for e in r.entries:
        rate_str = f"{e.recovery_rate:.1%}" if math.isfinite(e.recovery_rate) else "N/A"
        p_str = f"{e.mean_period_days:.3f}" if e.mean_period_days is not None else "—"
        d_str = f"{e.mean_depth_ppm:.0f}" if e.mean_depth_ppm is not None else "—"
        lines.append(
            f"| {e.tic_id} | {e.n_injected} | {e.n_recovered} | "
            f"{rate_str} | {p_str} | {d_str} |"
        )
    return "\n".join(lines)


def _cli() -> int:
    p = argparse.ArgumentParser(description="Aggregate injection-recovery completeness.")
    p.add_argument("results_json", help="JSON array of target result dicts or @file")
    args = p.parse_args()
    raw = args.results_json
    if raw.startswith("@"):
        with open(raw[1:]) as f:
            results = json.load(f)
    else:
        results = json.loads(raw)
    r = build_completeness_summary(results)
    print(format_completeness_summary(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
