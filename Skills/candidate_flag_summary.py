"""Consolidate multiple vetting flags into a single PASS / WARN / FAIL verdict.

Each vetting check contributes a named flag at one of three severity levels.
The summary verdict is the worst severity seen across all flags.  Unknown or
missing flags are treated as WARN.

Public API
----------
FlagLevel (enum-like str constants)
FlagEntry(name, level, message)
FlagSummaryResult(verdict, n_pass, n_warn, n_fail, flags, flag)
summarise_flags(flag_entries) -> FlagSummaryResult
format_flag_summary(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass

# Severity levels as string constants (avoid enum import for simplicity)
PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"

_SEVERITY = {PASS: 0, WARN: 1, FAIL: 2}
_VALID_LEVELS = frozenset(_SEVERITY)


@dataclass(frozen=True)
class FlagEntry:
    name: str
    level: str   # "PASS" | "WARN" | "FAIL"
    message: str = ""


@dataclass(frozen=True)
class FlagSummaryResult:
    verdict: str          # "PASS" | "WARN" | "FAIL"
    n_pass: int
    n_warn: int
    n_fail: int
    flags: tuple[FlagEntry, ...]
    flag: str             # "OK" | "EMPTY" | "INVALID"


def summarise_flags(flag_entries: list[FlagEntry]) -> FlagSummaryResult:
    """Consolidate a list of :class:`FlagEntry` into a verdict.

    Args:
        flag_entries: List of vetting flags from individual checks.

    Returns:
        :class:`FlagSummaryResult` with worst-case verdict.
    """
    if not flag_entries:
        return FlagSummaryResult(PASS, 0, 0, 0, (), "EMPTY")

    n_pass = 0
    n_warn = 0
    n_fail = 0
    worst = PASS

    sanitised: list[FlagEntry] = []
    for entry in flag_entries:
        lvl = entry.level if entry.level in _VALID_LEVELS else WARN
        sanitised.append(FlagEntry(entry.name, lvl, entry.message))
        if lvl == PASS:
            n_pass += 1
        elif lvl == WARN:
            n_warn += 1
        else:
            n_fail += 1
        if _SEVERITY[lvl] > _SEVERITY[worst]:
            worst = lvl

    return FlagSummaryResult(
        verdict=worst,
        n_pass=n_pass,
        n_warn=n_warn,
        n_fail=n_fail,
        flags=tuple(sanitised),
        flag="OK",
    )


def format_flag_summary(result: FlagSummaryResult) -> str:
    """Format flag summary as Markdown."""
    lines = [
        "## Candidate Flag Summary",
        "",
        f"- Verdict: **{result.verdict}**",
        f"- PASS: {result.n_pass}  WARN: {result.n_warn}  FAIL: {result.n_fail}",
        f"- Status: {result.flag}",
        "",
    ]
    if result.flags:
        lines += ["| Check | Level | Message |", "|---|---|---|"]
        for f in result.flags:
            lines.append(f"| {f.name} | {f.level} | {f.message} |")
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="candidate_flag_summary",
        description="Consolidate vetting flags into a single verdict.",
    )
    parser.parse_args(argv)

    result = summarise_flags([])
    print(format_flag_summary(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
