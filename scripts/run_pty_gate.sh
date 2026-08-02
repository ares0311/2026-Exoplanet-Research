#!/usr/bin/env bash
# PHASE 2 PRIMARY GATE runner -- installed interactive PTY operator experience.
#
# This script exists because the acceptance gate must allocate a real
# pseudo-terminal (/dev/ptmx) and spawn the installed console script as a
# separate operating-system process. It is registered in the user's
# sandbox.excludedCommands so it can obtain that device.
#
# It runs the gate only. It performs no network access, no installation, no
# git mutation, and no writes outside this repository.
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
EVIDENCE_DIR="logs/prod_closure_evidence/phase2"
mkdir -p "$EVIDENCE_DIR"

echo "=== EXO-Hunter PTY acceptance gate ==="
echo "repo:       $REPO_ROOT"
echo "commit:     $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "dirty:      $([ -n "$(git status --porcelain)" ] && echo true || echo false)"
echo "executable: $REPO_ROOT/.venv/bin/EXO-Hunter"
echo

echo "--- PTY availability probe ---"
.venv/bin/pytest logs/prod_closure_evidence/phase0/test_ptmx_probe.py \
  -s -n0 -q -p no:cacheprovider 2>&1 | sed -n '1,6p'
echo

echo "--- primary gate ---"
set +e
.venv/bin/pytest tests/test_pty_operator_acceptance.py \
  -v -n0 -p no:cacheprovider -rs --tb=short "$@" \
  2>&1 | tee "$EVIDENCE_DIR/pty_gate.txt"
STATUS="${PIPESTATUS[0]}"
set -e

echo
echo "--- evidence bundle ---"
if [ -f artifacts/manifests/exohunter_pty_acceptance.json ]; then
  echo "written: artifacts/manifests/exohunter_pty_acceptance.json"
else
  echo "NOT WRITTEN -- the gate did not execute against a real PTY (NOT EXECUTED)"
fi

echo
echo "gate exit status: $STATUS"
exit "$STATUS"
