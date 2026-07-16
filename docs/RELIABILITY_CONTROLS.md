# Verifiable Agent Reliability Controls

Design and verification record for the reliability-control architecture
added in version 0.2.80. This document is the evidence trail required by
AGENTS.md's "No Unsupported Completion Claims": every claim below is backed
by a command that was actually run against this repository, not asserted.

## 1. Governing principle

Repository state and reproducible verification outrank agent claims. A
fresh developer or agent must be able to inspect the repository, run the
documented checks, and independently determine whether required behavior is
implemented correctly — without trusting any agent's self-report.

## 2. Architecture (why this is the smallest practical solution)

Before adding anything, the existing repository was inspected:

- **Directive files**: `AGENTS.md` (canonical rules, ~68 KB) and `CLAUDE.md`
  (architecture/state, defers rule authority to AGENTS.md). A scan for
  rule-indicator words (`MANDATORY`, `NEVER `, `ALWAYS `) found 10 hits in
  `AGENTS.md` and **0** in `CLAUDE.md` — confirming there was no duplicated
  or contradictory rule content to reconcile before adding new directives.
  No factoring of existing content was needed; only new sections were added.
- **Claude/Codex exposure mechanism** (verified empirically this session,
  not assumed): this project's own Claude Code session log shows
  `CLAUDE.md`'s content is auto-injected into context at session start;
  `AGENTS.md`'s is not — Claude only reads it because `CLAUDE.md`'s own
  opening lines instruct the reader to do so ("Read `AGENTS.md` first, every
  session"). `.codex/config.toml` (Codex's own MCP bootstrap config)
  contains no directive content of its own; Codex's documented convention is
  to auto-load a repo-root `AGENTS.md` directly. So: **Codex's exposure to
  the rules is direct and native; Claude's is indirect, mediated entirely by
  one sentence in `CLAUDE.md`.** That sentence going missing is the one
  concrete, plausible drift scenario worth guarding — everything else
  (`AGENTS.md` itself being deleted/truncated) affects both agents equally
  and was already somewhat implicitly assumed safe, but is now checked too.
- **Tests/CI/lint**: 2,828 pytest tests (0.2.79 baseline), `ruff`, `mypy
  --strict` on `src/`, GitHub Actions `ci.yml` running all three, and
  `Skills/run_quality_gates.py` as the existing canonical single-parent
  6×6 local gate supervisor. All already enforced with non-zero exit on
  failure. Nothing here needed replacing — only extending.
- **Existing traceability**: the repo already has a Run Report Policy
  (`Skills/run_report.py`) stamping structured completion records with
  timestamps and commit context for acquisition/processing scripts. That
  satisfies most of "make implementation claims traceable" already; the one
  gap was the quality-gate summary itself not recording *which* tree state
  it verified.

Given all of the above was already in good shape, the added surface is
deliberately small: two new single-purpose scripts (no new dependencies,
stdlib `ast`/`pathlib` only), two new gate entries in the existing
supervisor, one new field-pair in its existing summary JSON, and new
sections in the existing canonical rules file. No new task runner, no
provenance database, no directive-management framework.

## 3. Traceability chain

| Requirement (task spec) | Current authoritative location | Claude exposure | Codex exposure |
|---|---|---|---|
| Fail Loudly | `AGENTS.md` § "Fail Loudly" | Read via `CLAUDE.md` pointer (checked by `check_directive_integrity.py`) | Native auto-load of `AGENTS.md` |
| No Fake Completion | `AGENTS.md` § "No Fake Completion" | same | same |
| No Unsupported Completion Claims | `AGENTS.md` § "No Unsupported Completion Claims" | same | same |
| Incomplete-implementation detection | `Skills/check_incomplete_implementations.py`, run by `Skills/run_quality_gates.py` | n/a (tooling, not a text directive) | n/a |
| Directive integrity / Claude-Codex parity | `Skills/check_directive_integrity.py`, run by `Skills/run_quality_gates.py` | n/a | n/a |
| Verification freshness | `Skills/run_quality_gates.py`'s `_git_state()` → `git_head_sha`/`git_dirty` in `quality_gate_summary.json` | n/a | n/a |

## 4. Verification — commands actually executed and their actual results

All commands below were run via the project's `exo_guard` MCP server
(`Skills/run_quality_gates.py`), the repo's own canonical entry point, from
this working tree.

**Run 1** (baseline after adding the two new scripts, before line-length
fixes): `passed=9/10 failed=1` — `ruff` failed with 4 `E501` line-too-long
violations (three in the new files, one pre-existing style slip in a test).
Both new gates (`incomplete_implementations`, `directive_integrity`) already
passed on this first run.

**Run 2** (after fixing the 4 line-length violations): `passed=10/10
failed=0`, elapsed 25.2s, `test_files=151` (149 baseline + 2 new test
files). Summary JSON recorded `git_head_sha` matching the tree at that
commit and `git_dirty: true` (uncommitted work-in-progress at the time) —
demonstrating the freshness field actually reflects real state rather than
being a hardcoded stub.

Exact test count for the 0.2.80 release commit is recorded in the CLAUDE.md
changelog entry for this version, taken directly from the final pre-commit
gate run's own summary output rather than computed by hand.

## 5. Negative controls (adversarial verification)

Rather than a separate mutate-repo-then-restore script, negative controls
are implemented as ordinary pytest tests operating entirely on `tmp_path`
fixtures — they never read or write this repository's real `AGENTS.md`,
`CLAUDE.md`, `.codex/config.toml`, `src/`, or `Skills/`. This satisfies "must
not leave the repository modified or broken" by construction (nothing real
is ever touched) and, unlike a one-off manual adversarial script, these
controls run automatically on every normal `pytest`/quality-gate invocation
— they cannot be forgotten or silently skip being re-run.

`tests/test_check_incomplete_implementations.py`:
- Known-good: clean function, `except: pass` fallback (the exact pattern
  found live in `src/exo_toolkit/search.py`/`ml/cnn_scorer.py`),
  `@abstractmethod` bare-pass, `Protocol` `...` body, docstring-only body →
  all assert zero violations.
- Known-bad: bare-pass stub (top-level and nested-in-class), docstring +
  bare-pass, `raise NotImplementedError` (bare and call form), `TODO`
  marker, `FIXME` marker → each asserts exactly one violation of the
  expected `kind`.
- Allowlist: `# allow-stub:` marker suppresses each violation kind
  individually, and is scoped to only the marked line (a sibling unmarked
  stub in the same file is still caught).
- Malformed: unparseable source (`def broken(:`) asserts `scan_source`
  raises `SyntaxError` rather than silently returning no findings.
- Integration: a constructed `tmp_path` tree confirms only `src/`+`Skills/`
  are scanned (a stub under `tests/` is correctly ignored), and that an
  empty or scan-root-less tree returns no violations without raising.

`tests/test_check_directive_integrity.py`:
- Known-good: a full fixture tree (padded `AGENTS.md` with all three
  required sections, `CLAUDE.md` with the exact pointer phrase,
  `.codex/config.toml` present) asserts zero problems across every check.
- Known-bad: `AGENTS.md` missing, `AGENTS.md` truncated below the byte
  floor, `AGENTS.md` empty, one required section header renamed away, all
  three section headers absent, `CLAUDE.md` missing, `CLAUDE.md`'s pointer
  phrase reworded (the literal silent-drift scenario this checker exists to
  catch), `.codex/config.toml` missing — each asserts the corresponding
  specific problem is reported.
- Malformed: a completely empty root directory asserts every check that can
  independently fail does fail (3 of 4 — `required_sections_present`
  correctly defers to `agents_md_present` rather than double-reporting).

`tests/test_run_quality_gates.py` (`TestGitState`): a real temporary git
repository (via `git init`/`commit`) confirms a clean commit reports the
exact `HEAD` SHA with `git_dirty: false`; an uncommitted edit flips
`git_dirty` to `true`; a non-git directory reports `git_head_sha: None` with
an explicit `git_state_error` key rather than silently returning empty/zero
values.

All of the above ran as part of the passing `10/10` gate run in Section 4 —
the adversarial verification and the normal verification are the same run,
which is stronger evidence than a separately-invoked, easy-to-forget script.

## 6. Scope decisions and exceptions

- **No factoring of `AGENTS.md`/`CLAUDE.md` into smaller files.** The repo
  already has the established, lower-risk pattern for this exact problem —
  peeling historical/archival content into `docs/MILESTONE_HISTORY.md` —
  and the live policy content itself is not duplicated or contradictory
  (verified above), so splitting it further was judged to add restructuring
  risk without a demonstrated material problem. This should be revisited if
  `AGENTS.md`'s *core policy* section (not its changelog) grows
  materially — the changelog narrative is expected to keep growing and is
  not itself in scope for splitting.
- **No mypy coverage for the two new scripts.** `pyproject.toml`'s
  `[tool.mypy] files = ["src"]` already excludes all of `Skills/`; the new
  scripts are consistent with every existing `Skills/*.py` script in this
  regard, not a new gap.
- **No changes to `docs/PRODUCTION_READINESS.md`/`docs/ROADMAP.md`.** Those
  documents track the scientific production-readiness roadmap (Phase 1-4
  deliverables); this is orthogonal engineering-process tooling, not a
  Phase deliverable, so forcing it into that narrative would misrepresent
  it as scientific progress it is not.
- **`check_incomplete_implementations.py` is presence-only detection.** A
  clean scan is evidence of "not conspicuously incomplete," never evidence
  of correctness — this is stated in the script's own docstring and in
  AGENTS.md, to prevent exactly the "verification theater" the task warns
  against.
