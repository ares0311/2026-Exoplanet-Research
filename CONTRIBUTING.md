# CONTRIBUTING.md (updated)

[See full content in chat — includes improved Local Setup + Environment Rules]

## Local Setup

Use Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

If your shell treats brackets specially, quote `.[dev]` as shown.

## Standard Validation

Run these before handing work to another agent or opening a pull request:

```bash
pytest --cov=exo_toolkit --cov-report=term-missing
ruff check .
mypy src
```

Default tests must not require live external services. Mark any live service test with `integration_live`.

## Background Automation Checks

The background search automation uses top-level configs and top-level SQLite logs:

```text
configs/background_search_v0.json
logs/background_search.sqlite3
```

Useful local commands:

```bash
PYTHONPATH=src python3 -m exo_toolkit.cli target-priority-summary
PYTHONPATH=src python3 -m exo_toolkit.cli background-run-once --dry-run
PYTHONPATH=src python3 -m exo_toolkit.cli background-run-once
PYTHONPATH=src python3 -m exo_toolkit.cli validation-summary
```

Generated SQLite databases and background report exports are runtime artifacts. Do not commit them unless a future decision explicitly promotes a fixture artifact.

## Local Performance Guidance

For local hardware details and recommended worker/thread defaults, see `docs/SYSTEM_PROFILE.md`.

Performance-sensitive code should use configurable resource limits. Do not require the local MacBook Pro profile for ordinary unit tests or lightweight development.
