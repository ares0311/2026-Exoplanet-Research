# MCP Bootstrapper — 2026 Exoplanet Research

> **Use:** Place this file at the repository root as `MCP_BOOTSTRAP.md` or keep this descriptive filename and instruct the coding agent to read it immediately after `AGENTS.md`.
>
> **Purpose:** Give Claude Code and Codex enough repo-local instructions to generate safe, project-scoped MCP configuration files without using manual application settings as the source of truth.

---

## 1. Objective

Bootstrap a conservative, project-scoped MCP setup for the **2026 Exoplanet Research** repository.

The rollout must generate, validate, and hand off these files:

```text
.mcp.json
.codex/config.toml
```

The generated MCP configuration must help coding agents inspect the repository, run fixed validation commands, and preserve scientific provenance while preventing unsafe behaviors such as arbitrary shell execution, live network tests by default, secret exposure, direct external submission, or unsupported discovery language.

This file is the source-of-truth rollout instruction. The generated config files are implementation artifacts derived from this policy.

---

## 2. Required Reading Before Any Change

Before creating or modifying MCP config, read these files in this order:

```text
AGENTS.md
CLAUDE.md
README.md
docs/PROJECT_STATUS.md
docs/PIPELINE_SPEC.md
docs/SCORING_MODEL.md
docs/DATA_SOURCES.md
docs/DECISIONS.md
CONTRIBUTING.md
pyproject.toml
```

If a listed file is missing, record the missing file in the rollout handoff and continue only if the remaining files are sufficient to preserve safety.

Do not rely on chat history, prior memory, or unstated assumptions as the source of truth.

---

## 3. Existing Project Constraints To Preserve

The MCP rollout must preserve the repository's existing rules:

- Python 3.11+.
- Package: `exo-toolkit`.
- CLI entry point: `exo`.
- Development quality tools: `ruff`, `mypy`, `pytest`.
- Default tests must not require live external services.
- Live external tests must remain explicitly marked `integration_live`.
- Do not claim an internally detected signal is a confirmed planet.
- Use conservative terms such as `candidate signal`, `possible transit-like event`, or `follow-up target`.
- Always expose false-positive evidence, missing diagnostics, thresholds, inputs, and provenance.
- Suppress formal submission pathways when key diagnostics are missing.
- External submission or contact requires explicit human approval.
- Background search runtime logs and generated reports are runtime artifacts and must not be committed unless a later explicit decision promotes a fixture.

---

## 4. Files To Generate Or Update

Generate or conservatively merge the following files:

```text
.mcp.json
.codex/config.toml
```

Create parent directories as needed.

If either file already exists:

1. Read it fully.
2. Preserve unrelated existing servers and comments where possible.
3. Do not overwrite existing user configuration.
4. Add only the Exoplanet Research MCP entries required by this bootstrap.
5. If merge safety is uncertain, stop and write a clear handoff instead of replacing the file.

Do not modify `AGENTS.md` unless the human explicitly asks.

---

## 5. MCP Server Design

Configure only a small, conservative MCP set.

### 5.1 Required Server: Project Files

Purpose:

- Read project files.
- Inspect docs, source, tests, schemas, and configs.
- Limit file access to this repository root.

Rules:

- Repository-root scope only.
- No access to parent directories.
- No access to global home directories.
- No access to `.venv/`, `data/`, `logs/`, `reports/`, or large caches unless the current task explicitly requires it.
- Do not expose secret files such as tokens, `.env`, Keychain exports, or credential dumps.

### 5.2 Required Server: Git Read / Limited Git

Purpose:

- Inspect `git status`, diffs, branches, and recent history.
- Help avoid overwriting unrelated user changes.

Allowed operations:

```text
git status --short --branch
git diff
git diff --staged
git log --oneline --decorate -n 20
git branch --show-current
```

Forbidden operations through MCP unless the human explicitly approves in the current task:

```text
git push
git push --force
git push --force-with-lease
git reset --hard
git clean -fd
git checkout -- .
git rebase
git merge
git tag
git remote set-url
```

### 5.3 Required Server: `exo_guard`

Create or configure a narrow local validation guard named:

```text
exo_guard
```

The guard must expose fixed commands only. It must not provide arbitrary shell access.

Allowed `exo_guard` commands:

```bash
ruff check .
python -m mypy src
PYTHONPATH=src python -m pytest
PYTHONPATH=src python -m pytest --cov=exo_toolkit --cov-report=term-missing
exo background-run-once --dry-run
exo run-summary
exo sqlite-integrity
```

If the CLI entry point `exo` is unavailable, use the repository's active Python environment and report the exact blocker. Do not install packages globally.

If the project uses `uv`, prefer the equivalent `uv run ...` form, but do not rewrite repository documentation merely to switch environment managers.

### 5.4 Optional Server: GitHub Read-Only

Configure GitHub MCP only if credentials are already available through the approved local mechanism and the human has authorized GitHub access for the task.

Allowed:

- Read issues.
- Read pull requests.
- Read workflow status.
- Read repository metadata.

Forbidden unless explicitly approved:

- Creating releases.
- Editing branch protections.
- Deleting branches.
- Force-pushing.
- Writing secrets.
- Opening external scientific submission records.

---

## 6. Forbidden MCP Capabilities

Do not configure MCP tools that allow:

- arbitrary shell execution;
- unrestricted filesystem access;
- package installation without approval;
- credential reading, printing, exporting, or modification;
- network access by default;
- live MAST, ExoFOP, NASA Exoplanet Archive, or catalog queries in default validation;
- external submission or contact;
- bypassing failing tests;
- editing science thresholds without documenting the decision;
- modifying `.gitignore` to permit runtime artifacts;
- committing `.venv/`, caches, raw data, SQLite runtime logs, generated reports, or API tokens.

---

## 7. Live Network Policy

Default MCP operation is offline.

Live network access is allowed only when all of the following are true:

1. The current human task explicitly asks for live data access.
2. The requested provider and scope are named.
3. Rate limits and cache behavior are documented.
4. Credentials are referenced by environment variable or approved local token file, never pasted into config.
5. Any live test is marked `integration_live` and is excluded from the default test suite.
6. No external submission or contact is attempted without explicit human approval.

Never turn a dry run into a live provider call by changing config silently.

---

## 8. Secrets And Credentials Policy

Never store secrets in:

```text
.mcp.json
.codex/config.toml
AGENTS.md
MCP_BOOTSTRAP.md
docs/
tests/
```

Allowed credential reference patterns:

```text
env:MAST_API_TOKEN
env:EXOFOP_TOKEN
~/.mast_api_token        # reference path only; do not read or print token contents
macOS Keychain           # reference service name only; do not export values
```

If a credential is required but absent, report the missing credential name and stop that live-network step. Do not prompt the agent to paste secrets into chat.

---

## 9. Generated Configuration Requirements

### 9.1 Claude Code: `.mcp.json`

Generate project-scoped Claude Code MCP configuration in:

```text
.mcp.json
```

Requirements:

- Keep server scope project-local.
- Use environment-variable references for secrets.
- Prefer stdio transports for local guard servers.
- Include only the approved servers from this file.
- Avoid global, user-home, or parent-directory file access.
- Include comments only if the target format supports them; JSON generally does not.

After writing `.mcp.json`, tell the human that Claude Code may require a one-time project trust / MCP approval prompt before using the servers.

### 9.2 Codex: `.codex/config.toml`

Generate project-scoped Codex MCP configuration in:

```text
.codex/config.toml
```

Requirements:

- Use project-local MCP server entries.
- Keep secrets out of TOML; use environment-variable references only.
- Do not modify global `~/.codex/config.toml`.
- Do not assume application UI settings are the source of truth.
- If Codex CLI syntax has changed, consult current local help or official docs and record any deviation in the handoff.

---

## 10. Validation Procedure

Run the safest available checks in this order:

```bash
git status --short --branch
python --version
ruff check .
python -m mypy src
PYTHONPATH=src python -m pytest
```

Then run the MCP-specific validation available locally, such as listing configured servers or starting each guard server in no-op mode.

Do not run live-network tests during bootstrap.

If any validation fails:

1. Do not hide the failure.
2. Record the exact command and failure summary.
3. Fix only failures directly caused by the MCP bootstrap changes.
4. If the failure predates the bootstrap or is unrelated, report it as an existing blocker.

---

## 11. Acceptance Criteria

The rollout is complete only when all of the following are true:

- `.mcp.json` exists or an existing file was safely merged.
- `.codex/config.toml` exists or an existing file was safely merged.
- Configured MCP servers are limited to project files, safe git inspection, and fixed validation commands.
- No arbitrary shell MCP is configured.
- No secrets are present in config files.
- No live network access is enabled by default.
- Default validation commands have been run or blockers are documented.
- The handoff states whether Claude Code and Codex require a one-time trust/approval action.
- No scientific claim, candidate status, or submission pathway has been changed by this rollout.

---

## 12. Handoff Format

At the end, report:

```text
MCP Bootstrap Handoff — Exoplanet Research

Files created/modified:
- ...

Servers configured:
- ...

Validation run:
- command: PASS/FAIL/SKIPPED — note

Live network status:
- disabled by default

Secrets status:
- no secrets stored in repo config

Human actions required:
- approve Claude Code project MCP config if prompted
- trust Codex project config if prompted
- provide any needed credentials through environment variables only

Known blockers:
- ...
```

Do not claim the repository is scientifically ready for live discovery or external submission merely because MCP bootstrap succeeded.
