# AGENTS.md (updated)

[See full content in chat — includes Environment Rules section at bottom]

## Multi-Agent Continuity

Multiple agents may work on this project across separate sessions, branches, and chat threads. Do not rely on chat context, memory, or prior conversation history as the source of truth.

Before making decisions or changes, recover project context from committed files, especially `AGENTS.md`, `README.md`, `CONTRIBUTING.md`, and the files in `docs/`. When new durable instructions, architectural decisions, operating rules, or scientific assumptions are established, record them in the appropriate repository document instead of leaving them only in chat.

If chat context conflicts with repository documentation, prefer the repository documentation unless the user explicitly instructs otherwise in the current task. Preserve enough rationale, provenance, and test evidence in commits, docs, and code comments for another agent to continue without needing this conversation.

## Local System Profile

Before making performance-sensitive changes or running large jobs, read `docs/SYSTEM_PROFILE.md`.

Optimize local defaults for the recorded MacBook Pro M4 Max profile while keeping scientific code portable and configurable. Do not hardcode local machine assumptions into candidate detection, scoring, or pathway logic.
