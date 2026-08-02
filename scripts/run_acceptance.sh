#!/usr/bin/env bash
# PHASES 3-5 -- real-data New and Follow-up acceptance (E2E-01..E2E-04).
#
# Runs the installed EXO-Hunter against live MAST/NEA through the canonical
# pipeline, on a DISPOSABLE COPY of the production database. The real database
# is hashed before and after and must be byte-identical.
#
# Registered in the user's sandbox.excludedCommands so the run can reach live
# archives and allocate a terminal. It never writes outside this repository.
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

REAL_DB="data/hunter_searches.sqlite3"
DISPOSABLE="data/hunter_searches.disposable_prod_closure.sqlite3"
EV="logs/prod_closure_evidence/phase5"
mkdir -p "$EV"

# Lightkurve honours XDG_CACHE_HOME when $XDG_CACHE_HOME/lightkurve exists.
# Keeping the cache repo-local avoids writing outside the workspace.
export XDG_CACHE_HOME="$REPO_ROOT/data/.xdg_cache_sandbox"
mkdir -p "$XDG_CACHE_HOME/lightkurve"

TARGETS="${TARGETS:-5}"

echo "=== EXO-Hunter real-data acceptance ==="
echo "commit:      $(git rev-parse HEAD)"
echo "dirty:       $([ -n "$(git status --porcelain)" ] && echo true || echo false)"
echo "started_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "targets:     $TARGETS"

echo
echo "--- production database guard (before) ---"
BEFORE="$(shasum -a 256 "$REAL_DB" | awk '{print $1}')"
echo "real db sha256: $BEFORE"
cp "$REAL_DB" "$DISPOSABLE"
echo "disposable copy: $DISPOSABLE"

run_step () {
  local label="$1"; shift
  echo
  echo "--- $label ---"
  set +e
  "$@" 2>&1 | tee "$EV/${label}.txt"
  local status="${PIPESTATUS[0]}"
  set -e
  echo "[$label] exit=$status"
  return 0
}

run_step "01_create_new" .venv/bin/Create-New-Search \
  --targets "$TARGETS" --mode new --db "$DISPOSABLE" --json --no-color

run_step "02_run_new" .venv/bin/Run-New-Search \
  --db "$DISPOSABLE" --json --no-color

run_step "03_create_followup" .venv/bin/Create-New-Search \
  --targets "$TARGETS" --mode follow-up --db "$DISPOSABLE" --json --no-color

run_step "04_run_followup" .venv/bin/Run-New-Search \
  --db "$DISPOSABLE" --json --no-color

# E2E-03: a fresh process must see the durable state and not regenerate it.
run_step "05_restart_show_followups" .venv/bin/Show-Follow-Ups \
  --db "$DISPOSABLE" --json --no-color

run_step "06_restart_resume" .venv/bin/Run-New-Search \
  --db "$DISPOSABLE" --json --no-color

echo
echo "--- production database guard (after) ---"
AFTER="$(shasum -a 256 "$REAL_DB" | awk '{print $1}')"
echo "real db sha256: $AFTER"
if [ "$BEFORE" != "$AFTER" ]; then
  echo "FAIL: the production database was mutated"
  exit 1
fi
echo "PASS: production database byte-identical before and after"

echo
echo "--- disposable database integrity ---"
.venv/bin/exo sqlite-integrity --db "$DISPOSABLE" 2>&1 | tail -5 || true

echo
echo "finished_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
