"""Score hourly weather windows for ground-based transit observations.

Score per hour:
  score_i = (1 - cloud_cover_pct/100)*0.5
           + (1 - min(seeing/4.0, 1))*0.3
           + (1 - min(humidity_pct/90, 1))*0.1
           + (1 - min(wind_kph/50, 1))*0.1

A window is "good" if score > 0.65.

Public API
----------
WeatherWindowResult(n_hours, n_good_hours, best_start_idx, mean_score, scores, flag)
score_weather_windows(hourly_conditions) -> WeatherWindowResult
format_weather_window(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WeatherWindowResult:
    n_hours: int
    n_good_hours: int
    best_start_idx: int | None
    mean_score: float
    scores: tuple[float, ...]
    flag: str  # "GOOD", "MARGINAL", "POOR"


def _score_hour(cond: dict) -> float:
    """Compute weather score for a single hour."""
    cloud = float(cond.get("cloud_cover_pct", 0.0))
    seeing = float(cond.get("seeing_arcsec", 2.0))
    humidity = float(cond.get("humidity_pct", 50.0))
    wind = float(cond.get("wind_kph", 10.0))

    score = (
        (1.0 - cloud / 100.0) * 0.5
        + (1.0 - min(seeing / 4.0, 1.0)) * 0.3
        + (1.0 - min(humidity / 90.0, 1.0)) * 0.1
        + (1.0 - min(wind / 50.0, 1.0)) * 0.1
    )
    return max(0.0, min(1.0, score))


def score_weather_windows(
    hourly_conditions: list[dict],
) -> WeatherWindowResult:
    """Score hourly weather windows.

    Args:
        hourly_conditions: List of dicts per hour with keys:
            ``seeing_arcsec``, ``cloud_cover_pct``, ``humidity_pct``,
            ``wind_kph``.

    Returns:
        :class:`WeatherWindowResult`.
    """
    if not hourly_conditions:
        return WeatherWindowResult(
            n_hours=0,
            n_good_hours=0,
            best_start_idx=None,
            mean_score=0.0,
            scores=(),
            flag="POOR",
        )

    scores = tuple(_score_hour(c) for c in hourly_conditions)
    n_good = sum(1 for s in scores if s > 0.65)
    mean_score = sum(scores) / len(scores)

    # Best contiguous good window start
    best_start: int | None = None
    best_run = 0
    current_run = 0
    current_start = 0
    for i, s in enumerate(scores):
        if s > 0.65:
            if current_run == 0:
                current_start = i
            current_run += 1
            if current_run > best_run:
                best_run = current_run
                best_start = current_start
        else:
            current_run = 0

    if best_start is None and n_good > 0:
        # Non-contiguous: pick first good hour
        best_start = next(i for i, s in enumerate(scores) if s > 0.65)

    if mean_score > 0.65:
        flag = "GOOD"
    elif mean_score > 0.40:
        flag = "MARGINAL"
    else:
        flag = "POOR"

    return WeatherWindowResult(
        n_hours=len(scores),
        n_good_hours=n_good,
        best_start_idx=best_start,
        mean_score=round(mean_score, 4),
        scores=scores,
        flag=flag,
    )


def format_weather_window(result: WeatherWindowResult) -> str:
    """Format weather window result as Markdown."""
    lines = [
        "## Weather Window Scores",
        "",
        f"- Hours evaluated: {result.n_hours}",
        f"- Good hours (score > 0.65): {result.n_good_hours}",
        f"- Best window start index: {result.best_start_idx}",
        f"- Mean score: {result.mean_score:.4f}",
        f"- **Flag: {result.flag}**",
    ]
    if result.scores:
        lines += ["", "| Hour | Score |", "|------|-------|"]
        for i, s in enumerate(result.scores):
            lines.append(f"| {i:3d}  | {s:.4f} |")
    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("conditions_json", help="JSON file with list of hourly condition dicts")
    args = p.parse_args(argv)
    with open(args.conditions_json) as fh:
        conditions = json.load(fh)
    r = score_weather_windows(conditions)
    print(format_weather_window(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
