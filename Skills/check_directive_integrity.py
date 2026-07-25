"""Verify that binding agent directives are intact and reach both Claude Code
and OpenAI Codex, per AGENTS.md's "Multi-Agent Continuity" policy.

Exposure mechanism this repo relies on (verified empirically, not assumed):
  - Codex CLI natively auto-loads a repo-root `AGENTS.md` as its instruction
    file. AGENTS.md is therefore Codex's *entire* exposure path — nothing
    else is required for Codex to see these rules.
  - Claude Code natively auto-loads a repo-root `CLAUDE.md` (confirmed by
    this project's own session transcripts: CLAUDE.md's content is injected
    automatically, AGENTS.md's is not). Claude's exposure to AGENTS.md is
    therefore *indirect*: it depends entirely on CLAUDE.md continuing to
    instruct the reader to read AGENTS.md. If that pointer line is ever
    silently removed or reworded away, Claude Code sessions would stop
    reliably seeing the canonical rules while Codex sessions would not
    notice anything changed — a silent, one-sided drift.

This script cannot make Claude Code read a file it doesn't natively load;
it can only fail loudly the moment the pointer that currently makes that
happen goes missing, so the gap is caught in CI instead of discovered a
future frustrating session at a time.

Every check function takes an explicit `repo_root` so tests can exercise
known-good/known-bad/malformed fixtures without ever touching this
repository's real AGENTS.md/CLAUDE.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Conservative floor: AGENTS.md was ~68 KB when this check was written.
# This only needs to catch catastrophic truncation/deletion, not track size.
_AGENTS_MD_MIN_BYTES = 20_000

# The literal substring CLAUDE.md must contain, proving it still tells
# Claude Code to read AGENTS.md every session. Kept as one exact phrase
# (not a regex) so a drifted rewording is caught rather than pattern-matched
# away.
_CLAUDE_POINTER_PHRASE = "Read `AGENTS.md` first, every session"

# Section headers that must exist in AGENTS.md for the "fail loudly" /
# "no fake completion" / "no unsupported completion claims" directives
# required by the reliability-controls policy.
_REQUIRED_AGENTS_MD_SECTIONS = (
    "## Fail Loudly",
    "## No Fake Completion",
    "## No Unsupported Completion Claims",
)


def check_agents_md_present(repo_root: Path) -> list[str]:
    """AGENTS.md is Codex's entire exposure path — verify it is intact."""
    agents_md = repo_root / "AGENTS.md"
    problems: list[str] = []
    if not agents_md.is_file():
        problems.append(f"{agents_md} is missing — Codex has zero directive exposure.")
        return problems
    size = agents_md.stat().st_size
    if size < _AGENTS_MD_MIN_BYTES:
        problems.append(
            f"{agents_md} is only {size} bytes (expected at least "
            f"{_AGENTS_MD_MIN_BYTES}) — looks truncated or accidentally emptied."
        )
    return problems


def check_required_sections_present(repo_root: Path) -> list[str]:
    """Confirm the mandatory reliability-controls directives are present.

    Matched against actual markdown heading lines, not an unanchored
    substring search of the raw text — a substring match would report a
    demoted heading (e.g. "## Fail Loudly" demoted to "### Fail Loudly") or
    a mere prose/TOC mention of the phrase as if the real H2 heading were
    still present, exactly the silent-drift class this checker exists to
    catch.
    """
    agents_md = repo_root / "AGENTS.md"
    if not agents_md.is_file():
        return []  # already reported by check_agents_md_present
    headings = {
        stripped
        for line in agents_md.read_text(encoding="utf-8").splitlines()
        if (stripped := line.strip()).startswith("#")
    }
    return [
        f"{agents_md} is missing required section header: {section!r} "
        "(as an actual markdown heading, not merely mentioned in prose)"
        for section in _REQUIRED_AGENTS_MD_SECTIONS
        if not any(heading.startswith(section) for heading in headings)
    ]


def check_claude_pointer_present(repo_root: Path) -> list[str]:
    """Claude Code only sees AGENTS.md via this pointer — verify it survives."""
    claude_md = repo_root / "CLAUDE.md"
    problems: list[str] = []
    if not claude_md.is_file():
        problems.append(
            f"{claude_md} is missing — Claude Code has zero directive exposure "
            "(it natively auto-loads CLAUDE.md, not AGENTS.md)."
        )
        return problems
    text = claude_md.read_text(encoding="utf-8")
    if _CLAUDE_POINTER_PHRASE not in text:
        problems.append(
            f"{claude_md} no longer contains the required pointer phrase "
            f"{_CLAUDE_POINTER_PHRASE!r} — Claude Code sessions would silently "
            "stop being told to read AGENTS.md while Codex sessions are unaffected."
        )
    return problems


def check_codex_config_present(repo_root: Path) -> list[str]:
    """Sanity check that Codex's own repo wiring hasn't been deleted."""
    codex_config = repo_root / ".codex" / "config.toml"
    if not codex_config.is_file():
        return [f"{codex_config} is missing — Codex MCP/tool wiring may be broken."]
    return []


def run_all_checks(repo_root: Path) -> dict[str, list[str]]:
    return {
        "agents_md_present": check_agents_md_present(repo_root),
        "required_sections_present": check_required_sections_present(repo_root),
        "claude_pointer_present": check_claude_pointer_present(repo_root),
        "codex_config_present": check_codex_config_present(repo_root),
    }


def main(argv: list[str] | None = None) -> int:
    del argv
    results = run_all_checks(REPO_ROOT)
    problems = [problem for problems in results.values() for problem in problems]
    if not problems:
        print(
            "check_directive_integrity: AGENTS.md intact, required sections present, "
            "CLAUDE.md->AGENTS.md pointer intact, Codex config present.",
            flush=True,
        )
        return 0
    print(f"check_directive_integrity: {len(problems)} problem(s):", flush=True)
    for problem in problems:
        print(f"  {problem}", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
