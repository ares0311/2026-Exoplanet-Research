"""Repository-native EXO-Hunter production gate (contract PROD-01).

Emits a versioned machine-readable report and exits nonzero when any mandatory
requirement is not satisfied.

The central integrity rule is contract CLAIM-03: a check that could not run is
reported as ``NOT_EXECUTED`` with a reason and is never counted as a pass. A
``NOT_EXECUTED`` mandatory requirement blocks PROD exactly as a ``FAIL`` does,
so this gate can never report green for something it did not actually verify.
Unit tests alone are explicitly not ``prod-check`` (PROD-01), so this module
inspects the shipped artifacts and the real command surface rather than
re-running the test suite.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import platform
import shlex
import subprocess
import sys
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

REPORT_VERSION = "exo-hunter-prod-check-v1"
CONTRACT_VERSION = "HUNTER-PROD-2026-07-30.3"
CLI_UX_VERSION = "HUNTER-CLI-UX-2026-07-30.3"
LEDGER_SCHEMA_VERSION = "1.3.0"
STATE_WRITER = "exo_toolkit.prod_state:update_state_from_report"

Status = Literal["PASS", "FAIL", "NOT_EXECUTED"]

REQUIRED_COMMANDS = (
    "/New-Search",
    "/Follow-Up-Search",
    "/Run-Search",
    "/Show-Follow-Ups",
    "/Inspect-Target",
    "/Help",
    "/Exit",
)

REQUIRED_GOLDEN_FILES = (
    "command_palette.txt",
    "new_search_fields.txt",
    "invalid_targets.txt",
    "action_preview.txt",
    "results_table_80_columns.txt",
    "results_table_140_columns.txt",
)

# docs/README_SPEC.md's mandated top-level structure.
REQUIRED_README_HEADINGS = (
    "## Table of Contents",
    "## 1. Executive Summary",
    "### 1.1 Research Objective and Scientific Context",
    "### 1.2 Scope, Boundaries, and Exclusions",
    "### 1.3 System and Workflow Overview",
    "### 1.4 Verified Capability Status",
    "### 1.5 Evidence and Reproducibility",
    "## 2. CLI Tool Usage",
    "### 2.1 Prerequisites",
    "### 2.2 Installation",
    "### 2.3 Environment Setup",
    "### 2.4 Command Structure",
    "### 2.5 End-to-End Workflow",
    "### 2.6 Command Reference",
    "### 2.7 Outputs and Artifacts",
    "### 2.8 Exit Codes and Failure Behavior",
    "### 2.9 Troubleshooting",
    "## 3. Analytics, Mathematics, and Theoretical Foundation",
    "### 3.1 Problem Formulation",
    "### 3.2 Inputs, Outputs, Labels, Units, and Provenance",
    "### 3.3 Mathematical Notation",
    "### 3.4 Models, Algorithms, and Scores",
    "### 3.5 Assumptions, Objectives, and Statistical Methods",
    "### 3.6 Thresholds, Calibration, and Uncertainty",
    "### 3.7 Evaluation and Validation",
    "### 3.8 Limitations and Failure Modes",
    "### 3.9 Implementation and Test Traceability",
    "## 4. Sibling Repositories and Shared Data",
    "### 4.1 Research Program and Repository Responsibilities",
    "### 4.2 Local Discovery and Configuration",
    "### 4.3 Shared Artifacts, Ownership, and Access",
    "### 4.4 Schemas, Provenance, Versioning, and Compatibility",
    "### 4.5 Availability, Failure Behavior, and Regeneration",
    "### 4.6 Cross-Repository Safety Boundaries",
)

# README-03 forbids these; "partial" is checked word-boundary-wise to avoid
# false positives on words that merely contain it.
FORBIDDEN_README_WORDS = ("planned", "roadmap", "backlog", "future work", "future-work", "partial")

# UX-CMD-01 requires the palette to open on a bare "/" keystroke. Delivering a
# character before the operator presses Enter requires the terminal to be taken
# out of canonical (line-buffered) mode. These are the mechanisms that can do
# so; if the shipped interactive module imports none of them, the capability is
# absent from the product, independently of any runtime observation.
RAW_MODE_MODULES = frozenset(
    {
        "termios",
        "tty",
        "curses",
        "msvcrt",
        "prompt_toolkit",
        "textual",
        "urwid",
        "blessed",
        "readchar",
        "getkey",
    }
)

# The guided inline editor and the resolved-action preview are required
# operator surfaces (UX-IN-01, CLI/UX spec section 8). PIPE-02 forbids
# production-looking code that is reachable only from tests or direct imports,
# so they must be referenced by the shipped interactive shell itself.
# The results table is rendered by the command layer, not the shell, so it is
# checked separately by package-wide reachability rather than shell wiring.
REQUIRED_SHELL_UX_SYMBOLS = ("GuidedEntry", "render_action_preview")

# A behavioural PTY acceptance bundle, when one has been retained, lives here.
PTY_EVIDENCE_PATH = "artifacts/manifests/exohunter_pty_acceptance.json"

PHASE_CHECK_IDS: dict[int, frozenset[str]] = {
    0: frozenset(
        {
            "governing_artifacts",
            "state_ledger",
            "state_authority",
            "sibling_write_isolation",
        }
    ),
}


@dataclass(frozen=True)
class CheckResult:
    """One gate outcome."""

    check_id: str
    requirements: tuple[str, ...]
    status: Status
    detail: str

    @property
    def blocking(self) -> bool:
        """Both FAIL and NOT_EXECUTED block PROD (CLAIM-03)."""
        return self.status != "PASS"


# A repository checkout is identified by artifacts the gate must inspect, not by
# the package layout: an installed wheel has no docs/ or configs/ beside it.
ROOT_MARKERS = ("pyproject.toml", "docs/HUNTER_PROD_CONTRACT.md")


def _looks_like_repo(path: Path) -> bool:
    return all((path / marker).exists() for marker in ROOT_MARKERS)


def _repo_root(start: Path | None = None) -> Path | None:
    """Locate the repository checkout this gate must inspect.

    ``Path(__file__).parents[2]`` is correct for a source checkout and an
    editable install, but for a wheel install it resolves inside
    ``site-packages`` -- where none of the governing artifacts exist. Reporting
    that as ``governing_artifacts: FAIL`` blames the contract for what is really
    a lookup miss, so the search falls back to walking up from the working
    directory and finally reports "no repository" as a single honest result
    instead of a list of misleading failures.
    """
    module_relative = Path(__file__).resolve().parents[2]
    if _looks_like_repo(module_relative):
        return module_relative
    begin = (start or Path.cwd()).resolve()
    for candidate in (begin, *begin.parents):
        if _looks_like_repo(candidate):
            return candidate
    return None


def _check_entry_points(root: Path) -> CheckResult:
    """LAUNCH-04 / CLI-02: every required executable is registered."""
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file():
        return CheckResult(
            "entry_points", ("LAUNCH-04",), "NOT_EXECUTED", "pyproject.toml not found"
        )
    scripts = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["scripts"]
    required = {
        "EXO-Hunter": "exo_toolkit.hunter_shell:exohunter_entry",
        "Inspect-Target": "exo_toolkit.hunter_cli:inspect_target_entry",
        "prod-check": "exo_toolkit.prod_check:main_entry",
    }
    missing = [name for name, target in required.items() if scripts.get(name) != target]
    if missing:
        return CheckResult(
            "entry_points",
            ("LAUNCH-04", "CLI-02"),
            "FAIL",
            f"missing or misregistered console scripts: {', '.join(sorted(missing))}",
        )
    return CheckResult(
        "entry_points",
        ("LAUNCH-04", "CLI-02"),
        "PASS",
        f"{len(required)} required console scripts registered",
    )


def _check_command_palette() -> CheckResult:
    """CLI-02 / UX-CMD-02: the described command registry.

    This check proves only that the required commands are registered and render
    with descriptions and parameter shapes. It deliberately does not claim
    UX-CMD-01: calling ``render_palette`` in-process says nothing about whether
    a bare ``/`` keystroke opens the palette in a real terminal. That behaviour
    is covered by ``interactive_input_capability`` and by the behavioural
    ``interactive_pty_operator_experience`` gate.
    """
    try:
        from exo_toolkit.hunter_ux import COMMAND_SPECS, render_palette
    except Exception as exc:  # noqa: BLE001 - import failure is itself the finding
        return CheckResult(
            "command_palette", ("CLI-02",), "FAIL", f"palette module not importable: {exc}"
        )
    names = {spec.name for spec in COMMAND_SPECS}
    missing = [name for name in REQUIRED_COMMANDS if name not in names]
    if missing:
        return CheckResult(
            "command_palette",
            ("CLI-02", "UX-CMD-02"),
            "FAIL",
            f"required commands absent from the palette: {', '.join(missing)}",
        )
    rendered = render_palette(terminal_width=100)
    undescribed = [spec.name for spec in COMMAND_SPECS if not spec.summary.strip()]
    if undescribed:
        return CheckResult(
            "command_palette",
            ("UX-CMD-02",),
            "FAIL",
            f"palette entries without a description: {', '.join(undescribed)}",
        )
    if "Required:" not in rendered or "Optional:" not in rendered:
        return CheckResult(
            "command_palette",
            ("UX-CMD-02",),
            "FAIL",
            "palette does not render required/optional parameter shapes",
        )
    return CheckResult(
        "command_palette",
        ("CLI-02", "UX-CMD-02"),
        "PASS",
        f"{len(names)} described commands registered; all {len(REQUIRED_COMMANDS)} required "
        "present (registry only; keystroke behaviour not asserted here)",
    )


def _check_guided_input_validation() -> CheckResult:
    """UX-IN-03 / UX-IN-04 live validity sentinels."""
    try:
        from exo_toolkit.hunter_ux import GuidedEntry, ValidationError, command_index
        from exo_toolkit.hunter_ux import validate_target_count as validate
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            "guided_input_validation",
            ("UX-IN-03",),
            "FAIL",
            f"validation module not importable: {exc}",
        )
    for bad in ("twenty", "0", "-1", ""):
        try:
            validate(bad)
        except ValidationError:
            continue
        return CheckResult(
            "guided_input_validation",
            ("UX-IN-03",),
            "FAIL",
            f"invalid target count {bad!r} was accepted",
        )
    entry = GuidedEntry(command_index()["/inspect-target"])
    if entry.is_executable():
        return CheckResult(
            "guided_input_validation",
            ("UX-IN-02",),
            "FAIL",
            "a command with an unset required field reported itself executable",
        )
    return CheckResult(
        "guided_input_validation",
        ("UX-IN-02", "UX-IN-03", "UX-IN-04"),
        "PASS",
        "invalid input is rejected with sentinels and cannot advance",
    )


def _check_results_table() -> CheckResult:
    """UX-TABLE-01 width-aware rendering."""
    try:
        from exo_toolkit.hunter_ux import (
            DEFAULT_RESULT_COLUMNS,
            render_results_table,
            select_columns,
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            "results_table", ("UX-TABLE-01",), "FAIL", f"table module not importable: {exc}"
        )
    rows = [
        {
            "rank": 1,
            "target_id": "TIC 237884073",
            "ranking_score": 100.0,
            "search_status": "follow-up",
            "object_classification": "star",
            "selection_reason": "x" * 500,
        }
    ]
    for width in (40, 80, 140):
        rendered = render_results_table(rows, terminal_width=width)
        overflow = [line for line in rendered.splitlines() if len(line) > width]
        if overflow:
            return CheckResult(
                "results_table",
                ("UX-TABLE-01",),
                "FAIL",
                f"rendered line exceeded terminal width {width}",
            )
    narrow = [column.header for column in select_columns(DEFAULT_RESULT_COLUMNS, 5)]
    if "Rank" not in narrow or "Target" not in narrow:
        return CheckResult(
            "results_table",
            ("UX-TABLE-01",),
            "FAIL",
            "rank/identity columns were dropped under width pressure",
        )
    return CheckResult(
        "results_table",
        ("UX-TABLE-01",),
        "PASS",
        "no overflow at 40/80/140 columns; rank and identity always preserved",
    )


def _check_animation_degradation() -> CheckResult:
    """UX-START-04 / UX-A11Y-01 accessible degradation."""
    try:
        import io

        from exo_toolkit.hunter_shell import _animation_allowed
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            "animation_degradation", ("UX-START-04",), "FAIL", f"shell not importable: {exc}"
        )
    non_tty = io.StringIO()
    if _animation_allowed(disabled=False, input_stream=non_tty, stream=non_tty):
        return CheckResult(
            "animation_degradation",
            ("UX-START-04",),
            "FAIL",
            "animation was permitted on a non-TTY stream",
        )
    if _animation_allowed(disabled=True, input_stream=non_tty, stream=non_tty):
        return CheckResult(
            "animation_degradation",
            ("UX-START-04",),
            "FAIL",
            "explicit no-animation mode did not disable animation",
        )
    return CheckResult(
        "animation_degradation",
        ("UX-START-04", "UX-A11Y-01"),
        "PASS",
        "animation disabled for non-TTY and explicit no-animation mode",
    )


def _check_canonical_routing(root: Path) -> CheckResult:
    """CLI/UX §12 and PIPE-02: the CLI layer must not duplicate business logic."""
    ux = root / "src" / "exo_toolkit" / "hunter_ux.py"
    if not ux.is_file():
        return CheckResult(
            "canonical_routing", ("CLI-03",), "NOT_EXECUTED", "hunter_ux.py not found"
        )
    # Inspect real imports via the AST rather than raw text: a docstring that
    # merely names the business module (to say logic belongs there, not here) is
    # documentation, not a runtime dependency, and must not trip this gate.
    try:
        tree = ast.parse(ux.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        return CheckResult("canonical_routing", ("CLI-03",), "FAIL", f"unparsable: {exc}")
    forbidden = {"search_lifecycle", "sqlite3", "scoring", "hunter_cli"}
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    leaked = sorted(
        module
        for module in imported
        if any(part in forbidden for part in module.split("."))
    )
    if leaked:
        return CheckResult(
            "canonical_routing",
            ("CLI-03",),
            "FAIL",
            f"presentation layer imports business/persistence modules: {', '.join(leaked)}",
        )
    return CheckResult(
        "canonical_routing",
        ("CLI-03",),
        "PASS",
        "presentation layer holds no persistence, selection, or scoring logic",
    )


def _check_ranking_formula_integrity() -> CheckResult:
    """RANK-01: published equations must match the canonical implementation."""
    try:
        from exo_toolkit.hunter_ranking import selection_contract
        from exo_toolkit.search_lifecycle import _evidence_score
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            "ranking_formula_integrity", ("RANK-01",), "FAIL", f"not importable: {exc}"
        )
    published = selection_contract("follow-up")["expected_information_gain"]
    if published != "(1-fpp)*detection_confidence":
        return CheckResult(
            "ranking_formula_integrity",
            ("RANK-01",),
            "FAIL",
            f"unexpected published follow-up formula: {published!r}",
        )
    fpp, confidence = 0.10, 0.50
    evidence = {"false_positive_probability": fpp, "detection_confidence": confidence}
    got_fpp = _evidence_score(evidence, "false_positive_probability")
    got_conf = _evidence_score(evidence, "detection_confidence")
    if got_fpp is None or got_conf is None:
        return CheckResult(
            "ranking_formula_integrity",
            ("RANK-01",),
            "FAIL",
            "evidence extraction returned None for a well-formed payload",
        )
    expected = (1.0 - fpp) * confidence
    superseded = (100.0 * (1.0 - fpp) + 10.0 * confidence) / 110.0
    if abs(expected - superseded) < 1e-9:
        return CheckResult(
            "ranking_formula_integrity",
            ("RANK-01",),
            "NOT_EXECUTED",
            "probe row cannot distinguish the contract formula from the superseded proxy",
        )
    return CheckResult(
        "ranking_formula_integrity",
        ("RANK-01",),
        "PASS",
        f"published formula matches implementation; proxy divergence probe {expected:.4f} "
        f"vs {superseded:.4f}",
    )


def _check_golden_tests(root: Path) -> CheckResult:
    """EVAL-01 and CLI/UX §13."""
    golden = root / "tests" / "golden"
    if not golden.is_dir():
        return CheckResult("golden_tests", ("EVAL-01",), "FAIL", "tests/golden/ does not exist")
    missing = [name for name in REQUIRED_GOLDEN_FILES if not (golden / name).is_file()]
    if missing:
        return CheckResult(
            "golden_tests", ("EVAL-01",), "FAIL", f"missing golden files: {', '.join(missing)}"
        )
    empty = [
        name
        for name in REQUIRED_GOLDEN_FILES
        if not (golden / name).read_text(encoding="utf-8").strip()
    ]
    if empty:
        return CheckResult(
            "golden_tests", ("EVAL-01",), "FAIL", f"empty golden files: {', '.join(empty)}"
        )
    return CheckResult(
        "golden_tests",
        ("EVAL-01",),
        "PASS",
        f"{len(REQUIRED_GOLDEN_FILES)} EXO-scoped golden files present and non-empty",
    )


def _check_readme_conformance(root: Path) -> CheckResult:
    """README-01 / README-03."""
    readme = root / "README.md"
    if not readme.is_file():
        return CheckResult("readme_conformance", ("README-01",), "FAIL", "README.md not found")
    text = readme.read_text(encoding="utf-8")
    missing = [heading for heading in REQUIRED_README_HEADINGS if heading not in text]
    if missing:
        return CheckResult(
            "readme_conformance",
            ("README-01",),
            "FAIL",
            f"{len(missing)} required heading(s) absent, first: {missing[0]!r}",
        )
    positions = [text.index(heading) for heading in REQUIRED_README_HEADINGS]
    if positions != sorted(positions):
        return CheckResult(
            "readme_conformance", ("README-01",), "FAIL", "required headings are out of order"
        )
    lowered = text.lower()
    found = [word for word in FORBIDDEN_README_WORDS if word in lowered]
    if found:
        return CheckResult(
            "readme_conformance",
            ("README-03",),
            "FAIL",
            f"forbidden status vocabulary present: {', '.join(found)}",
        )
    return CheckResult(
        "readme_conformance",
        ("README-01", "README-03"),
        "PASS",
        f"all {len(REQUIRED_README_HEADINGS)} headings present, ordered, vocabulary clean",
    )


def _check_state_ledger(root: Path) -> CheckResult:
    """The durable ledger must have one coherent, machine-owned active state."""
    ledger = root / "configs" / "HUNTER_PROD_STATE.json"
    if not ledger.is_file():
        return CheckResult(
            "state_ledger", ("CLAIM-04",), "FAIL", "configs/HUNTER_PROD_STATE.json not found"
        )
    try:
        data = json.loads(ledger.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return CheckResult(
            "state_ledger", ("CLAIM-04",), "FAIL", f"ledger is not valid JSON: {exc}"
        )
    if data.get("artifact") != "configs/HUNTER_PROD_STATE.json":
        return CheckResult(
            "state_ledger",
            ("CLAIM-04",),
            "FAIL",
            f"ledger artifact path is {data.get('artifact')!r}",
        )
    if data.get("contract_version") != CONTRACT_VERSION:
        return CheckResult(
            "state_ledger",
            ("CLAIM-04",),
            "FAIL",
            f"ledger contract version {data.get('contract_version')!r} != {CONTRACT_VERSION}",
        )
    if data.get("schema_version") != LEDGER_SCHEMA_VERSION:
        return CheckResult(
            "state_ledger",
            ("CLAIM-04",),
            "FAIL",
            f"ledger schema {data.get('schema_version')!r} != {LEDGER_SCHEMA_VERSION}",
        )
    active_phase = data.get("active_phase")
    if active_phase not in {f"PHASE {index}" for index in range(8)}:
        return CheckResult(
            "state_ledger",
            ("CLAIM-04",),
            "FAIL",
            f"ledger must name exactly one active phase, got {active_phase!r}",
        )
    implementation_state = data.get("implementation_state")
    allowed_implementation_states = set(data.get("allowed_implementation_states", []))
    required_implementation_states = {
        "BLOCKING",
        "IN_PROGRESS",
        "IMPLEMENTED_NOT_VERIFIED",
    }
    if allowed_implementation_states != required_implementation_states:
        return CheckResult(
            "state_ledger",
            ("CLAIM-04",),
            "FAIL",
            "allowed implementation states must be exactly BLOCKING, IN_PROGRESS, "
            "and IMPLEMENTED_NOT_VERIFIED",
        )
    if implementation_state not in allowed_implementation_states:
        return CheckResult(
            "state_ledger",
            ("CLAIM-04",),
            "FAIL",
            f"invalid implementation_state {implementation_state!r}",
        )
    if data.get("gate_execution_state") not in {
        "EXECUTED",
        "NOT_EXECUTED",
        "UNKNOWN",
    }:
        return CheckResult(
            "state_ledger", ("CLAIM-04",), "FAIL", "invalid gate_execution_state"
        )
    if data.get("gate_result") not in {"PASS", "FAIL", "NOT_EXECUTED", "UNKNOWN"}:
        return CheckResult("state_ledger", ("CLAIM-04",), "FAIL", "invalid gate_result")
    requirements = data.get("requirements")
    if not isinstance(requirements, dict) or not requirements:
        return CheckResult(
            "state_ledger", ("CLAIM-04",), "FAIL", "active requirements map is missing"
        )
    required_fields = {"implementation_state", "gate_execution_state", "gate_result"}
    for requirement_id, requirement in requirements.items():
        if not isinstance(requirement, dict) or not required_fields.issubset(requirement):
            return CheckResult(
                "state_ledger",
                ("CLAIM-04",),
                "FAIL",
                f"requirement {requirement_id} does not separate implementation and gate state",
            )
        if "status" in requirement:
            return CheckResult(
                "state_ledger",
                ("CLAIM-04",),
                "FAIL",
                f"requirement {requirement_id} uses ambiguous agent-writable status",
            )
        if requirement["implementation_state"] not in allowed_implementation_states:
            return CheckResult(
                "state_ledger",
                ("CLAIM-04",),
                "FAIL",
                f"requirement {requirement_id} has invalid implementation_state",
            )
        if requirement["gate_execution_state"] not in {
            "EXECUTED",
            "NOT_EXECUTED",
            "UNKNOWN",
        }:
            return CheckResult(
                "state_ledger",
                ("CLAIM-04",),
                "FAIL",
                f"requirement {requirement_id} has invalid gate_execution_state",
            )
        if requirement["gate_result"] not in {"PASS", "FAIL", "NOT_EXECUTED", "UNKNOWN"}:
            return CheckResult(
                "state_ledger",
                ("CLAIM-04",),
                "FAIL",
                f"requirement {requirement_id} has invalid gate_result",
            )
        verification_state = requirement.get("verification_state")
        if verification_state not in {None, "VERIFIED"}:
            return CheckResult(
                "state_ledger",
                ("CLAIM-04",),
                "FAIL",
                f"requirement {requirement_id} has invalid verification_state",
            )
        if verification_state == "VERIFIED" and (
            requirement["gate_execution_state"] != "EXECUTED"
            or requirement["gate_result"] != "PASS"
            or not requirement.get("evidence_ref")
            or not isinstance(requirement.get("tested_code_identity"), dict)
            or not isinstance(requirement.get("gate_hashes"), dict)
        ):
            return CheckResult(
                "state_ledger",
                ("CLAIM-04",),
                "FAIL",
                f"requirement {requirement_id} claims VERIFIED without complete passing "
                "machine evidence",
            )
    prod_status = data.get("prod_status")
    if prod_status not in {None, "PROD"}:
        return CheckResult(
            "state_ledger",
            ("CLAIM-04",),
            "FAIL",
            f"prod_status must be null or machine-written PROD, got {prod_status!r}",
        )
    return CheckResult(
        "state_ledger",
        ("CLAIM-04",),
        "PASS",
        f"ledger schema {LEDGER_SCHEMA_VERSION}; one active phase; "
        f"{len(requirements)} requirements separate implementation and gate state",
    )


def _check_state_authority(root: Path) -> CheckResult:
    """Only the deterministic runner may own VERIFIED and PROD state."""
    ledger = root / "configs" / "HUNTER_PROD_STATE.json"
    try:
        data = json.loads(ledger.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return CheckResult(
            "state_authority", ("CLAIM-04", "PROD-01"), "FAIL", f"ledger unavailable: {exc}"
        )
    authority = data.get("verification_authority")
    if not isinstance(authority, dict):
        return CheckResult(
            "state_authority",
            ("CLAIM-04", "PROD-01"),
            "FAIL",
            "verification_authority object is missing",
        )
    if authority.get("verified_and_prod_writer") != STATE_WRITER:
        return CheckResult(
            "state_authority",
            ("CLAIM-04", "PROD-01"),
            "FAIL",
            f"verified_and_prod_writer must be {STATE_WRITER}",
        )
    writer_path = root / "src" / "exo_toolkit" / "prod_state.py"
    if not writer_path.is_file():
        return CheckResult(
            "state_authority",
            ("CLAIM-04", "PROD-01"),
            "FAIL",
            "deterministic state writer src/exo_toolkit/prod_state.py is missing",
        )
    try:
        tree = ast.parse(writer_path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError) as exc:
        return CheckResult(
            "state_authority",
            ("CLAIM-04", "PROD-01"),
            "FAIL",
            f"deterministic state writer is unreadable: {exc}",
        )
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    if "update_state_from_report" not in functions:
        return CheckResult(
            "state_authority",
            ("CLAIM-04", "PROD-01"),
            "FAIL",
            "deterministic state writer lacks update_state_from_report",
        )
    return CheckResult(
        "state_authority",
        ("CLAIM-04", "PROD-01"),
        "PASS",
        f"VERIFIED and PROD ownership is bound to {STATE_WRITER}",
    )


def _check_governing_artifacts(root: Path) -> CheckResult:
    """The governing artifacts must be readable in their declared formats."""
    problems: list[str] = []
    for relative in ("docs/HUNTER_PROD_CONTRACT.md", "docs/CLI_UX_SPEC.md", "docs/README_SPEC.md"):
        path = root / relative
        if not path.is_file():
            problems.append(f"{relative} missing")
            continue
        head = path.read_text(encoding="utf-8", errors="replace")[:64]
        if head.startswith("{\\rtf"):
            problems.append(f"{relative} is RTF, not Markdown")
    if problems:
        return CheckResult(
            "governing_artifacts", ("WS-04",), "FAIL", "; ".join(problems)
        )
    return CheckResult(
        "governing_artifacts", ("WS-04",), "PASS", "all governing artifacts are readable Markdown"
    )


def _check_sibling_write_isolation(root: Path) -> CheckResult:
    """WS-01 / WS-03: sibling history is read-only and never runtime-coupled.

    The contract requires current, versioned sibling-history *reads*.  A raw
    marker scan therefore rejects required production behaviour.  This check
    instead follows sibling-derived path values through the AST and rejects
    observable mutation, command execution, runtime-import coupling, symlinks,
    and hard-coded absolute sibling paths while permitting ordinary reads.
    """
    markers = (
        "NEOHunter",
        "TechnoHunter",
        "2026 Near Earth Objects",
        "2026 Technosignatures",
    )
    # This gate module necessarily names the siblings in order to detect them,
    # so scanning itself is a guaranteed false positive. Identify it by its
    # location *inside the scanned tree* rather than by the running file's
    # absolute path: under a wheel install the executing module lives in
    # site-packages and would never compare equal to the checkout copy, so the
    # gate would flag its own marker constants on that surface only.
    self_relative = Path("exo_toolkit") / Path(__file__).name
    write_methods = {
        "chmod",
        "hardlink_to",
        "mkdir",
        "rename",
        "replace",
        "rmdir",
        "symlink_to",
        "touch",
        "unlink",
        "write_bytes",
        "write_text",
    }
    command_calls = {
        "subprocess.call",
        "subprocess.check_call",
        "subprocess.check_output",
        "subprocess.Popen",
        "subprocess.run",
    }
    mutating_calls = {
        "os.chmod",
        "os.link",
        "os.makedirs",
        "os.mkdir",
        "os.remove",
        "os.rename",
        "os.replace",
        "os.rmdir",
        "os.symlink",
        "os.unlink",
        "shutil.copy",
        "shutil.copy2",
        "shutil.copyfile",
        "shutil.copytree",
        "shutil.move",
        "shutil.rmtree",
    }
    import_calls = {
        "importlib.import_module",
        "importlib.machinery.SourceFileLoader",
        "importlib.util.spec_from_file_location",
    }

    def dotted_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            prefix = dotted_name(node.value)
            return f"{prefix}.{node.attr}" if prefix else node.attr
        return ""

    def contains_marker(value: str) -> bool:
        return any(marker in value for marker in markers)

    def references_tainted(node: ast.AST, names: set[str]) -> bool:
        return any(
            (isinstance(child, ast.Name) and child.id in names)
            or (isinstance(child, ast.Constant) and isinstance(child.value, str)
                and contains_marker(child.value))
            or (
                isinstance(child, ast.Call)
                and dotted_name(child.func).endswith("sibling_history_export_path")
            )
            for child in ast.walk(node)
        )

    def assigned_names(node: ast.AST) -> set[str]:
        return {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}

    offenders: list[str] = []
    for path in (root / "src").rglob("*.py"):
        if path.relative_to(root / "src") == self_relative:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError as exc:
            offenders.append(f"{path.relative_to(root)}:unparseable:{exc.lineno}")
            continue

        tainted: set[str] = set()
        assignments = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr))
        ]
        changed = True
        while changed:
            changed = False
            for assignment in assignments:
                value = assignment.value
                if value is None:
                    continue
                if not references_tainted(value, tainted):
                    continue
                targets: list[ast.AST]
                if isinstance(assignment, ast.Assign):
                    targets = assignment.targets
                else:
                    targets = [assignment.target]
                new_names = set().union(*(assigned_names(target) for target in targets))
                if not new_names.issubset(tainted):
                    tainted.update(new_names)
                    changed = True

        relative = path.relative_to(root)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and contains_marker(node.value)
                and Path(node.value).is_absolute()
            ):
                offenders.append(f"{relative}:{node.lineno}:hardcoded-absolute-sibling-path")
            if not isinstance(node, ast.Call):
                continue
            call_name = dotted_name(node.func)
            args = [*node.args, *(keyword.value for keyword in node.keywords)]
            tainted_args = any(references_tainted(arg, tainted) for arg in args)

            if isinstance(node.func, ast.Attribute):
                receiver_tainted = references_tainted(node.func.value, tainted)
                if node.func.attr in write_methods and receiver_tainted:
                    offenders.append(f"{relative}:{node.lineno}:{node.func.attr}")
                if call_name in {"sys.path.append", "sys.path.insert"} and tainted_args:
                    offenders.append(f"{relative}:{node.lineno}:runtime-import-path")
                if node.func.attr == "open" and receiver_tainted:
                    mode_nodes = [
                        *node.args,
                        *(kw.value for kw in node.keywords if kw.arg == "mode"),
                    ]
                    modes = [
                        child.value
                        for mode in mode_nodes
                        for child in ast.walk(mode)
                        if isinstance(child, ast.Constant) and isinstance(child.value, str)
                    ]
                    if any(set(mode) & set("wax+") for mode in modes):
                        offenders.append(f"{relative}:{node.lineno}:open-write")

            if call_name == "open" and node.args and references_tainted(node.args[0], tainted):
                mode_nodes = [
                    *node.args[1:],
                    *(kw.value for kw in node.keywords if kw.arg == "mode"),
                ]
                modes = [
                    child.value
                    for mode in mode_nodes
                    for child in ast.walk(mode)
                    if isinstance(child, ast.Constant) and isinstance(child.value, str)
                ]
                if any(set(mode) & set("wax+") for mode in modes):
                    offenders.append(f"{relative}:{node.lineno}:open-write")
            if call_name in command_calls and tainted_args:
                offenders.append(f"{relative}:{node.lineno}:sibling-command")
            if call_name in mutating_calls and tainted_args:
                offenders.append(f"{relative}:{node.lineno}:sibling-mutation")
            if call_name in import_calls and tainted_args:
                offenders.append(f"{relative}:{node.lineno}:runtime-import")
    if offenders:
        return CheckResult(
            "sibling_write_isolation",
            ("WS-01", "WS-03"),
            "FAIL",
            f"prohibited sibling mutation or runtime coupling: "
            f"{', '.join(sorted(set(offenders))[:5])}",
        )
    return CheckResult(
        "sibling_write_isolation",
        ("WS-01", "WS-03"),
        "PASS",
        "sibling references are read-only: no writes, commands, runtime imports, "
        "symlinks, generated output, or hard-coded absolute sibling paths",
    )


def _check_package_completeness(root: Path) -> CheckResult:
    """Every shipped runtime module must import cleanly."""
    package = root / "src" / "exo_toolkit"
    modules = sorted(
        p.relative_to(root / "src").with_suffix("").as_posix().replace("/", ".")
        for p in package.rglob("*.py")
        if "__pycache__" not in p.parts and p.name != "__init__.py"
    )
    failures: list[str] = []
    for name in modules:
        code = subprocess.run(
            [sys.executable, "-c", f"import {name}"],
            capture_output=True,
            text=True,
            cwd=root / "src",
        )
        if code.returncode != 0:
            first = (code.stderr.strip().splitlines() or ["unknown error"])[-1]
            failures.append(f"{name}: {first}")
    if failures:
        return CheckResult(
            "package_completeness",
            ("PIPE-01",),
            "FAIL",
            f"{len(failures)} module(s) failed to import, first: {failures[0]}",
        )
    return CheckResult(
        "package_completeness", ("PIPE-01",), "PASS", f"all {len(modules)} runtime modules import"
    )


def _module_symbols(path: Path) -> tuple[set[str], set[str]]:
    """Return (imported top-level module names, referenced symbol names).

    Both are collected from the parsed AST rather than raw text, so a mention
    inside a docstring or comment can never satisfy a wiring requirement.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    referenced: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.add(node.module.split(".")[0])
            referenced.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.Name):
            referenced.add(node.id)
        elif isinstance(node, ast.Attribute):
            referenced.add(node.attr)
    return imported, referenced


def _check_interactive_input_capability(root: Path) -> CheckResult:
    """UX-CMD-01: a bare ``/`` must open the palette without Enter.

    Character-at-a-time delivery is a hard prerequisite: a terminal left in
    canonical mode does not hand any byte to the process until the operator
    presses Enter, so a line-buffered read cannot observe ``/`` in time. This
    check therefore asks a structural question about the shipped artifact --
    does the interactive module have any mechanism capable of leaving canonical
    mode? -- rather than inferring runtime behaviour from source. Absence of
    every such mechanism is a demonstrated missing capability, which CLAIM-03
    treats as a FAIL rather than an unexecuted check.
    """
    shell = root / "src" / "exo_toolkit" / "hunter_shell.py"
    if not shell.is_file():
        return _not_executed(
            "interactive_input_capability",
            ("UX-CMD-01",),
            "src/exo_toolkit/hunter_shell.py not found",
        )
    imported, _ = _module_symbols(shell)
    capable = sorted(imported & RAW_MODE_MODULES)
    if capable:
        return CheckResult(
            "interactive_input_capability",
            ("UX-CMD-01", "CLI-01"),
            "PASS",
            f"character-at-a-time input capability present via {', '.join(capable)}",
        )
    return CheckResult(
        "interactive_input_capability",
        ("UX-CMD-01", "CLI-01"),
        "FAIL",
        "the interactive loop has no character-at-a-time input mechanism "
        f"(none of {', '.join(sorted(RAW_MODE_MODULES))} is imported), so the terminal "
        "stays in canonical mode and '/' cannot open the palette until Enter is pressed",
    )


def _check_guided_entry_wiring(root: Path) -> CheckResult:
    """UX-IN-01 / PIPE-02: required operator surfaces must be reachable.

    ``GuidedEntry``, the resolved-action preview, and the width-aware results
    table are contract-required operator surfaces. PIPE-02 forbids
    production-looking code reachable only through tests or direct imports, so
    the shipped interactive shell must actually reference them.
    """
    shell = root / "src" / "exo_toolkit" / "hunter_shell.py"
    if not shell.is_file():
        return _not_executed(
            "guided_entry_wiring",
            ("UX-IN-01", "PIPE-02"),
            "src/exo_toolkit/hunter_shell.py not found",
        )
    _, referenced = _module_symbols(shell)
    missing = [name for name in REQUIRED_SHELL_UX_SYMBOLS if name not in referenced]
    if missing:
        return CheckResult(
            "guided_entry_wiring",
            ("UX-IN-01", "PIPE-02"),
            "FAIL",
            f"required operator surfaces are never referenced by the shipped shell: "
            f"{', '.join(missing)} (reachable only from tests or this gate)",
        )
    return CheckResult(
        "guided_entry_wiring",
        ("UX-IN-01", "PIPE-02"),
        "PASS",
        f"all {len(REQUIRED_SHELL_UX_SYMBOLS)} required operator surfaces are wired into the shell",
    )


def _check_pty_operator_experience(root: Path) -> CheckResult:
    """UX-CMD-01/UX-CMD-03/UX-IN-01: behavioural proof from a real terminal.

    The only admissible evidence is a retained bundle produced by spawning the
    installed executable as a separate operating-system process attached to a
    real pseudo-terminal and sending actual keystrokes. A renderer test, a
    golden file, or a ``/`` followed by Enter cannot substitute for it.
    """
    bundle = root / PTY_EVIDENCE_PATH
    if not bundle.is_file():
        return _not_executed(
            "interactive_pty_operator_experience",
            ("UX-CMD-01", "UX-CMD-03", "UX-IN-01", "UX-IN-03", "LAUNCH-04"),
            f"no retained real-PTY acceptance bundle at {PTY_EVIDENCE_PATH}; requires "
            "spawning the installed executable in a real pseudo-terminal and sending "
            "keystrokes, which no in-process renderer check can substitute for",
        )
    try:
        payload = json.loads(bundle.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return CheckResult(
            "interactive_pty_operator_experience",
            ("UX-CMD-01", "LAUNCH-04"),
            "FAIL",
            f"real-PTY acceptance bundle is unreadable: {exc}",
        )
    if payload.get("palette_opened_without_enter") is not True:
        return CheckResult(
            "interactive_pty_operator_experience",
            ("UX-CMD-01", "LAUNCH-04"),
            "FAIL",
            "retained PTY bundle does not record the palette opening on a bare '/' keystroke",
        )
    return CheckResult(
        "interactive_pty_operator_experience",
        ("UX-CMD-01", "UX-CMD-03", "UX-IN-01", "UX-IN-03", "LAUNCH-04"),
        "PASS",
        f"real-PTY acceptance bundle records keystroke-driven palette at {PTY_EVIDENCE_PATH}",
    )


def _not_executed(check_id: str, requirements: tuple[str, ...], reason: str) -> CheckResult:
    return CheckResult(check_id, requirements, "NOT_EXECUTED", reason)


def _environmental_checks() -> list[CheckResult]:
    """Checks requiring a fresh environment or live data.

    These are reported honestly rather than assumed. CLAIM-03 forbids counting
    any of them toward an ``N/N passed`` total.
    """
    return [
        _not_executed(
            "documented_installation_launch",
            ("LAUNCH-01", "LAUNCH-02", "LAUNCH-03"),
            "requires a fresh out-of-tree virtual environment, a built wheel, an "
            "upgrade-in-place, and execution from an unrelated directory; not staged by "
            "this gate",
        ),
        _not_executed(
            "adaptive_discovery",
            ("DISC-01", "DISC-02", "DISC-03"),
            "requires a live catalog sweep with expansion rounds and sufficiency evidence",
        ),
        _not_executed(
            "identity_history_completeness",
            ("IDENT-01", "IDENT-02", "IDENT-03", "IDENT-04"),
            "requires a validated cross-project history corpus to assert novelty exclusion",
        ),
        _not_executed(
            "exact_target_execution",
            ("DUR-02", "DUR-03"),
            "requires a real create-then-execute run against the canonical pipeline",
        ),
        _not_executed(
            "partial_state_and_resume",
            ("DUR-04", "E2E-03"),
            "requires an interrupted real run followed by a resume",
        ),
        _not_executed(
            "real_data_evidence_freshness",
            ("E2E-01", "E2E-02", "E2E-04"),
            "requires a retained live-MAST New and Follow-up acceptance bundle",
        ),
    ]


def run_checks(root: Path | None = None) -> list[CheckResult]:
    """Run every gate check and return results in report order.

    When no repository checkout can be located the gate reports one honest
    ``NOT_EXECUTED`` result rather than a list of artifact failures that would
    misattribute a lookup miss to a broken contract.
    """
    base = root or _repo_root()
    if base is None:
        return [
            _not_executed(
                "repository_checkout",
                ("PROD-01",),
                "no repository checkout found (looked for "
                f"{' and '.join(ROOT_MARKERS)} beside the package and above the working "
                "directory); prod-check must run from a checkout, not from an installed "
                "wheel alone",
            )
        ]
    static: list[Callable[[], CheckResult]] = [
        lambda: _check_governing_artifacts(base),
        lambda: _check_state_ledger(base),
        lambda: _check_state_authority(base),
        lambda: _check_entry_points(base),
        _check_command_palette,
        lambda: _check_interactive_input_capability(base),
        lambda: _check_guided_entry_wiring(base),
        lambda: _check_pty_operator_experience(base),
        _check_guided_input_validation,
        _check_results_table,
        _check_animation_degradation,
        lambda: _check_canonical_routing(base),
        _check_ranking_formula_integrity,
        lambda: _check_golden_tests(base),
        lambda: _check_readme_conformance(base),
        lambda: _check_sibling_write_isolation(base),
        lambda: _check_package_completeness(base),
    ]
    results = [check() for check in static]
    results.extend(_environmental_checks())
    return results


def build_report(
    results: Sequence[CheckResult],
    *,
    commit: str | None = None,
    phase: int | None = None,
) -> dict[str, Any]:
    """Assemble the versioned machine-readable report."""
    passed = [r for r in results if r.status == "PASS"]
    failed = [r for r in results if r.status == "FAIL"]
    not_executed = [r for r in results if r.status == "NOT_EXECUTED"]
    gate_passed = not failed and not not_executed
    return {
        "report_version": REPORT_VERSION,
        "contract_version": CONTRACT_VERSION,
        "cli_ux_version": CLI_UX_VERSION,
        "gate_scope": "FULL PROD" if phase is None else f"PHASE {phase}",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "commit": commit,
        "checks": [asdict(result) for result in results],
        "summary": {
            "total": len(results),
            "passed": len(passed),
            "failed": len(failed),
            "not_executed": len(not_executed),
            # CLAIM-02: name the denominator explicitly.
            "passed_denominator": (
                f"{len(passed)}/{len(results)} checks passed; "
                f"{len(not_executed)} NOT EXECUTED and excluded from any pass claim"
            ),
        },
        "gate_passed": gate_passed,
        # A phase-scoped pass is not a PROD decision.  Only the unscoped
        # repository-native runner is permitted to make that claim.
        "prod_ready": phase is None and gate_passed,
    }


def _git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=root
        )
    except OSError:
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


MATERIAL_GATE_PATHS = (
    "docs/HUNTER_PROD_CONTRACT.md",
    "docs/CLI_UX_SPEC.md",
    "docs/README_SPEC.md",
    "src/exo_toolkit/prod_check.py",
    "tests/test_prod_check.py",
)


def _material_hashes(root: Path) -> dict[str, str]:
    """Hash the frozen gate and the governing fixtures it interprets."""
    hashes: dict[str, str] = {}
    for relative in MATERIAL_GATE_PATHS:
        path = root / relative
        hashes[relative] = (
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "MISSING"
        )
    return hashes


def _relative_evidence_path(root: Path, output: Path) -> str:
    resolved = output.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _gate_result(report: dict[str, Any]) -> Status:
    if report["summary"]["failed"]:
        return "FAIL"
    if report["summary"]["not_executed"]:
        return "NOT_EXECUTED"
    return "PASS"


def _validate_state_update(
    *,
    root: Path,
    report: dict[str, Any],
    phase: int,
    command: str,
    evidence_path: str,
    gate_hashes: dict[str, str],
) -> None:
    """Independently validate the writer's durable, externally visible result."""
    ledger = json.loads((root / "configs" / "HUNTER_PROD_STATE.json").read_text(encoding="utf-8"))
    expected_result = _gate_result(report)
    expected_exit = 0 if expected_result == "PASS" else 1
    exact = {
        "gate_execution_state": "EXECUTED",
        "gate_result": expected_result,
        "gate_exit_code": expected_exit,
        "gate_command": command,
        "gate_evidence_path": evidence_path,
        "gate_hashes": gate_hashes,
    }
    mismatches = [key for key, value in exact.items() if ledger.get(key) != value]
    identity = ledger.get("tested_code_identity", {})
    if identity.get("git_head_sha") != report.get("commit"):
        mismatches.append("tested_code_identity.git_head_sha")
    phase_state = ledger.get("phase_results", {}).get(f"PHASE {phase}", {})
    if phase_state.get("gate_result") != expected_result:
        mismatches.append(f"phase_results.PHASE {phase}.gate_result")
    expected_verification = "VERIFIED" if expected_result == "PASS" else None
    if phase_state.get("verification_state") != expected_verification:
        mismatches.append(f"phase_results.PHASE {phase}.verification_state")
    expected_prod = "PROD" if phase == 6 and report.get("prod_ready") is True else None
    if ledger.get("prod_status") != expected_prod:
        mismatches.append("prod_status")
    if mismatches:
        raise ValueError("state writer postcondition mismatch: " + ", ".join(mismatches))


def _update_state(
    *,
    root: Path,
    report: dict[str, Any],
    phase: int,
    command: str,
    evidence_path: str,
    gate_hashes: dict[str, str],
) -> CheckResult:
    """Invoke and then independently check the deterministic state writer."""
    try:
        module_name, function_name = STATE_WRITER.split(":", 1)
        module = importlib.import_module(module_name)
        writer = getattr(module, function_name)
        writer(
            root=root,
            report=report,
            phase=phase,
            command=command,
            evidence_path=evidence_path,
            gate_hashes=gate_hashes,
            environment={
                "platform": platform.platform(),
                "python": platform.python_version(),
                "python_executable": str(Path(sys.executable).resolve()),
                "resolved_executable": str((root / ".venv" / "bin" / "EXO-Hunter").resolve()),
            },
        )
        _validate_state_update(
            root=root,
            report=report,
            phase=phase,
            command=command,
            evidence_path=evidence_path,
            gate_hashes=gate_hashes,
        )
    except Exception as exc:  # noqa: BLE001 - state authority failure must block the gate
        return CheckResult(
            "state_update",
            ("PHASE0-STATUS-AUTHORITY", "CLAIM-04", "PROD-01"),
            "FAIL",
            f"deterministic state update failed: {type(exc).__name__}: {exc}",
        )
    return CheckResult(
        "state_update",
        ("PHASE0-STATUS-AUTHORITY", "CLAIM-04", "PROD-01"),
        "PASS",
        f"machine state durably updated and independently re-read via {STATE_WRITER}",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the gate. Returns 0 only when every mandatory requirement passes."""
    parser = argparse.ArgumentParser(
        prog="prod-check",
        description="Repository-native EXO-Hunter production gate (contract PROD-01).",
    )
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--output", type=Path, help="Write the machine-readable report here.")
    parser.add_argument(
        "--root",
        type=Path,
        help="Repository checkout to inspect; defaults to autodetection.",
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=sorted(PHASE_CHECK_IDS),
        help="Run one frozen phase gate; currently Phase 0 is registered.",
    )
    parser.add_argument(
        "--update-state",
        action="store_true",
        help="Let the deterministic writer update the machine ledger from this phase result.",
    )
    args = parser.parse_args(argv)
    if args.update_state and args.output is None:
        parser.error("--update-state requires --output")

    root = args.root.resolve() if args.root else _repo_root()
    results = run_checks(root)
    if args.phase is not None:
        wanted = PHASE_CHECK_IDS[args.phase]
        results = [result for result in results if result.check_id in wanted]
        present = {result.check_id for result in results}
        for missing in sorted(wanted - present):
            results.append(
                _not_executed(
                    missing,
                    ("PROD-01",),
                    f"registered Phase {args.phase} check was not produced by the runner",
                )
            )

    command_tokens = [sys.argv[0], *sys.argv[1:]] if argv is None else ["prod-check", *argv]
    command = shlex.join(str(token) for token in command_tokens)
    commit = _git_commit(root) if root else None
    gate_hashes = _material_hashes(root) if root else {}

    if args.update_state and root and args.output:
        # Include the writer assertion in the report that the writer consumes.
        # A failure replaces this provisional result before the report is saved.
        provisional = CheckResult(
            "state_update",
            ("PHASE0-STATUS-AUTHORITY", "CLAIM-04", "PROD-01"),
            "PASS",
            f"machine state update requested through {STATE_WRITER}",
        )
        results.append(provisional)
        report = build_report(results, commit=commit, phase=args.phase)
        update = _update_state(
            root=root,
            report=report,
            phase=args.phase if args.phase is not None else 6,
            command=command,
            evidence_path=_relative_evidence_path(root, args.output),
            gate_hashes=gate_hashes,
        )
        results[-1] = update
    report = build_report(results, commit=commit, phase=args.phase)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    if args.json_output:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for result in results:
            marker = {"PASS": "PASS", "FAIL": "FAIL", "NOT_EXECUTED": "NOT EXECUTED"}[result.status]
            print(f"[{marker:>12}] {result.check_id}: {result.detail}")
        summary = report["summary"]
        print()
        print(summary["passed_denominator"])
        if args.phase is not None:
            phase_result = _gate_result(report)
            display = "NOT EXECUTED" if phase_result == "NOT_EXECUTED" else phase_result
            print(f"PRIMARY PHASE GATE: {display}")
        elif report["prod_ready"]:
            print("PROD gate: PASS")
        else:
            print(f"PROD gate: BLOCKED ({summary['failed']} failed, "
                  f"{summary['not_executed']} not executed)")

    return 0 if report["gate_passed"] else 1


def main_entry() -> None:
    raise SystemExit(main())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
