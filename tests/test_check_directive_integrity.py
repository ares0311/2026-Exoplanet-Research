"""Tests for Skills/check_directive_integrity.py.

Every check function takes an explicit repo_root, so these tests build
isolated known-good/known-bad/malformed fixture trees under tmp_path and
never touch this repository's real AGENTS.md/CLAUDE.md/.codex — required by
AGENTS.md's "Verify The Verification System" policy (negative controls must
not leave the repository modified or broken).
"""
from __future__ import annotations

from pathlib import Path

from Skills.check_directive_integrity import (
    check_agents_md_present,
    check_claude_pointer_present,
    check_codex_config_present,
    check_required_sections_present,
    run_all_checks,
)

_REQUIRED_SECTIONS = (
    "## Fail Loudly",
    "## No Fake Completion",
    "## No Unsupported Completion Claims",
)


def _good_agents_md_text() -> str:
    # Padded well past the byte floor with realistic filler, matching how
    # the real AGENTS.md is a large multi-section policy document.
    body = "\n\n".join(f"{section}\n\nSome policy content here." for section in _REQUIRED_SECTIONS)
    padding = "Additional canonical policy text.\n" * 1500
    return f"# AGENTS.md\n\n{body}\n\n{padding}"


def _good_claude_md_text() -> str:
    return "# CLAUDE.md\n\nRead `AGENTS.md` first, every session — read both.\n"


def _write_good_tree(root: Path) -> None:
    (root / "AGENTS.md").write_text(_good_agents_md_text(), encoding="utf-8")
    (root / "CLAUDE.md").write_text(_good_claude_md_text(), encoding="utf-8")
    codex_dir = root / ".codex"
    codex_dir.mkdir(parents=True, exist_ok=True)
    (codex_dir / "config.toml").write_text("sandbox_mode = \"workspace-write\"\n", encoding="utf-8")


class TestKnownGood:
    def test_full_good_tree_has_no_problems(self, tmp_path: Path) -> None:
        _write_good_tree(tmp_path)
        results = run_all_checks(tmp_path)
        all_problems = [problem for problems in results.values() for problem in problems]
        assert all_problems == []

    def test_agents_md_present_passes(self, tmp_path: Path) -> None:
        _write_good_tree(tmp_path)
        assert check_agents_md_present(tmp_path) == []

    def test_required_sections_present_passes(self, tmp_path: Path) -> None:
        _write_good_tree(tmp_path)
        assert check_required_sections_present(tmp_path) == []

    def test_claude_pointer_present_passes(self, tmp_path: Path) -> None:
        _write_good_tree(tmp_path)
        assert check_claude_pointer_present(tmp_path) == []

    def test_codex_config_present_passes(self, tmp_path: Path) -> None:
        _write_good_tree(tmp_path)
        assert check_codex_config_present(tmp_path) == []


class TestKnownBadAgentsMdMissingOrTruncated:
    def test_missing_agents_md_fails(self, tmp_path: Path) -> None:
        _write_good_tree(tmp_path)
        (tmp_path / "AGENTS.md").unlink()
        problems = check_agents_md_present(tmp_path)
        assert len(problems) == 1
        assert "missing" in problems[0]

    def test_truncated_agents_md_fails(self, tmp_path: Path) -> None:
        _write_good_tree(tmp_path)
        (tmp_path / "AGENTS.md").write_text("# AGENTS.md\n\ntruncated\n", encoding="utf-8")
        problems = check_agents_md_present(tmp_path)
        assert len(problems) == 1
        assert "truncated" in problems[0] or "bytes" in problems[0]

    def test_empty_agents_md_fails_loudly(self, tmp_path: Path) -> None:
        _write_good_tree(tmp_path)
        (tmp_path / "AGENTS.md").write_text("", encoding="utf-8")
        problems = check_agents_md_present(tmp_path)
        assert len(problems) == 1


class TestKnownBadMissingSections:
    def test_missing_one_required_section_is_reported(self, tmp_path: Path) -> None:
        _write_good_tree(tmp_path)
        text = (tmp_path / "AGENTS.md").read_text(encoding="utf-8")
        mutated = text.replace("## Fail Loudly", "## Something Else")
        (tmp_path / "AGENTS.md").write_text(mutated, encoding="utf-8")
        problems = check_required_sections_present(tmp_path)
        assert len(problems) == 1
        assert "Fail Loudly" in problems[0]

    def test_missing_all_required_sections_reports_all(self, tmp_path: Path) -> None:
        filler = "# AGENTS.md\n\nno policy sections here\n" * 1000
        (tmp_path / "AGENTS.md").write_text(filler, encoding="utf-8")
        problems = check_required_sections_present(tmp_path)
        assert len(problems) == len(_REQUIRED_SECTIONS)

    def test_demoted_heading_level_is_caught(self, tmp_path: Path) -> None:
        # Regression: a substring search would find "## Fail Loudly" inside
        # "### Fail Loudly" and wrongly report the section present even
        # though the real H2 heading is gone.
        _write_good_tree(tmp_path)
        text = (tmp_path / "AGENTS.md").read_text(encoding="utf-8")
        mutated = text.replace("## Fail Loudly", "### Fail Loudly")
        (tmp_path / "AGENTS.md").write_text(mutated, encoding="utf-8")
        problems = check_required_sections_present(tmp_path)
        assert len(problems) == 1
        assert "Fail Loudly" in problems[0]

    def test_prose_mention_without_real_heading_is_caught(self, tmp_path: Path) -> None:
        # Regression: a substring search would find the phrase inside a TOC
        # entry or inline prose mention and wrongly report the section
        # present even though no real heading line exists.
        _write_good_tree(tmp_path)
        text = (tmp_path / "AGENTS.md").read_text(encoding="utf-8")
        mutated = text.replace(
            "## Fail Loudly", "See the ## Fail Loudly policy referenced elsewhere"
        )
        (tmp_path / "AGENTS.md").write_text(mutated, encoding="utf-8")
        problems = check_required_sections_present(tmp_path)
        assert len(problems) == 1
        assert "Fail Loudly" in problems[0]


class TestKnownBadClaudePointer:
    def test_missing_claude_md_fails(self, tmp_path: Path) -> None:
        _write_good_tree(tmp_path)
        (tmp_path / "CLAUDE.md").unlink()
        problems = check_claude_pointer_present(tmp_path)
        assert len(problems) == 1
        assert "missing" in problems[0]

    def test_reworded_pointer_is_caught(self, tmp_path: Path) -> None:
        # Simulates the exact silent-drift scenario this check exists to
        # catch: someone edits CLAUDE.md and the pointer phrasing drifts.
        _write_good_tree(tmp_path)
        (tmp_path / "CLAUDE.md").write_text(
            "# CLAUDE.md\n\nSee AGENTS.md for more info sometime.\n", encoding="utf-8"
        )
        problems = check_claude_pointer_present(tmp_path)
        assert len(problems) == 1
        assert "pointer phrase" in problems[0]

    def test_exact_pointer_phrase_is_required(self, tmp_path: Path) -> None:
        _write_good_tree(tmp_path)
        assert check_claude_pointer_present(tmp_path) == []


class TestKnownBadCodexConfig:
    def test_missing_codex_config_fails(self, tmp_path: Path) -> None:
        _write_good_tree(tmp_path)
        (tmp_path / ".codex" / "config.toml").unlink()
        problems = check_codex_config_present(tmp_path)
        assert len(problems) == 1
        assert "missing" in problems[0]


class TestMalformedTree:
    def test_completely_empty_root_fails_loudly_on_every_check(self, tmp_path: Path) -> None:
        results = run_all_checks(tmp_path)
        all_problems = [problem for problems in results.values() for problem in problems]
        # agents_md_present + claude_pointer_present + codex_config_present
        # each report on a completely absent tree; required_sections_present
        # defers to agents_md_present when AGENTS.md itself is absent.
        assert len(all_problems) == 3
        assert results["required_sections_present"] == []
