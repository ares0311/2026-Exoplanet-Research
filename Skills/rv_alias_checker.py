"""Check for common RV sampling aliases (1-day, sidereal, annual)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RVAliasResult:
    candidate_period_days: float
    alias_periods_days: tuple[float, ...]
    alias_labels: tuple[str, ...]
    closest_alias_days: float
    closest_alias_label: str
    alias_proximity_percent: float   # how close to nearest alias (%)
    is_likely_alias: bool
    flag: str


_SIDEREAL_DAY = 0.99727      # days
_SYNODIC_MONTH = 29.5306     # days
_TROPICAL_YEAR = 365.2422    # days
_ANOMALISTIC_YEAR = 365.2596 # days


def check_rv_aliases(
    candidate_period_days: float,
    sampling_aliases: list[float] | None = None,
    proximity_threshold_percent: float = 3.0,
) -> RVAliasResult:
    """Check if a candidate RV period is a common sampling alias.

    Common aliases arise from:
      - 1-day alias: f_alias = |f_cand ± 1 cycle/day|
      - Sidereal day: f_alias = |f_cand ± f_sidereal|
      - Monthly: gaps from lunar-phase scheduling
      - Annual: seasonal gaps

    For period P and alias frequency f_a:
      P_alias = 1 / |1/P ± f_a|

    Args:
        candidate_period_days: candidate orbital period (days)
        sampling_aliases: list of additional alias periods to check
        proximity_threshold_percent: threshold for flagging as likely alias (%)
    """
    if candidate_period_days <= 0.0:
        return RVAliasResult(candidate_period_days, (), (), float("nan"), "NONE",
                              float("nan"), False, "INVALID_PERIOD")

    f_cand = 1.0 / candidate_period_days

    # Build list of alias periods from standard cadences
    alias_freqs: list[tuple[float, str]] = [
        (1.0, "1-day"),
        (1.0 / _SIDEREAL_DAY, "sidereal-day"),
        (2.0, "2-day"),
        (1.0 / 7.0, "weekly"),
        (1.0 / _SYNODIC_MONTH, "monthly"),
        (1.0 / (_TROPICAL_YEAR / 2.0), "semi-annual"),
        (1.0 / _TROPICAL_YEAR, "annual"),
    ]

    alias_periods: list[float] = []
    alias_labels: list[str] = []

    for f_a, label in alias_freqs:
        for sign in (+1, -1):
            f_new = abs(f_cand + sign * f_a)
            if f_new > 0:
                p_new = 1.0 / f_new
                if 0.5 <= p_new <= 2.0 * _TROPICAL_YEAR:
                    alias_periods.append(p_new)
                    alias_labels.append(f"{label}({'+'if sign>0 else '-'})")

    # Also add user-supplied aliases
    if sampling_aliases:
        for p_a in sampling_aliases:
            if p_a > 0:
                for sign in (+1, -1):
                    f_new = abs(f_cand + sign / p_a)
                    if f_new > 0:
                        p_new = 1.0 / f_new
                        alias_periods.append(p_new)
                        alias_labels.append(f"custom({p_a:.1f}d,{'+' if sign>0 else '-'})")

    # Find closest alias to a set of "well-known" periods that are suspicious
    # Key periods: 1d, sidereal, 7d, 29.5d, 365d, 182d
    known_suspect = [1.0, _SIDEREAL_DAY, 7.0, _SYNODIC_MONTH, 182.6, _TROPICAL_YEAR]
    if sampling_aliases:
        known_suspect.extend(sampling_aliases)

    best_period = float("nan")
    best_label = "NONE"
    best_prox = float("inf")

    for p_suspect, label in zip(
        known_suspect,
        ["1-day", "sidereal-day", "7-day", "monthly", "semi-annual", "annual"] +
        (["custom" for _ in (sampling_aliases or [])]),
        strict=False,
    ):
        prox = abs(candidate_period_days - p_suspect) / p_suspect * 100.0
        if prox < best_prox:
            best_prox = prox
            best_period = p_suspect
            best_label = label

    is_alias = best_prox < proximity_threshold_percent

    return RVAliasResult(
        candidate_period_days=candidate_period_days,
        alias_periods_days=tuple(alias_periods[:10]),   # top 10
        alias_labels=tuple(alias_labels[:10]),
        closest_alias_days=best_period,
        closest_alias_label=best_label,
        alias_proximity_percent=best_prox,
        is_likely_alias=is_alias,
        flag="OK",
    )


def format_rv_alias_result(r: RVAliasResult) -> str:
    if r.flag != "OK":
        return f"RVAlias | flag={r.flag}"
    alias_str = "YES" if r.is_likely_alias else "NO"
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Candidate period | {r.candidate_period_days:.4f} d |\n"
        f"| Closest suspect period | {r.closest_alias_days:.4f} d ({r.closest_alias_label}) |\n"
        f"| Proximity | {r.alias_proximity_percent:.2f} % |\n"
        f"| Likely alias | {alias_str} |\n"
        f"| Flag | {r.flag} |"
    )


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="RV alias checker")
    p.add_argument("period_days", type=float)
    p.add_argument("--threshold", type=float, default=3.0)
    args = p.parse_args()
    r = check_rv_aliases(args.period_days, proximity_threshold_percent=args.threshold)
    print(format_rv_alias_result(r))


if __name__ == "__main__":
    _cli()
