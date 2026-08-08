"""Tests for the repository-native production gate (contract PROD-01)."""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from exo_toolkit.prod_check import (
    LEDGER_SCHEMA_VERSION,
    PTY_EVIDENCE_PATH,
    REPORT_VERSION,
    STATE_WRITER,
    CheckResult,
    _check_guided_entry_wiring,
    _check_interactive_input_capability,
    _check_pty_operator_experience,
    _check_sibling_write_isolation,
    _check_state_authority,
    _check_state_ledger,
    _looks_like_repo,
    _repo_root,
    build_report,
    main,
    run_checks,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestReportShape:
    def test_report_is_versioned_and_machine_readable(self) -> None:
        report = build_report([CheckResult("x", ("WS-01",), "PASS", "ok")], commit="abc123")
        assert report["report_version"] == REPORT_VERSION
        assert report["commit"] == "abc123"
        json.dumps(report)  # must be serializable

    def test_not_executed_never_counts_as_passed(self) -> None:
        """CLAIM-03: a skipped stage cannot appear in an N/N passed total."""
        results = [
            CheckResult("a", ("WS-01",), "PASS", "ok"),
            CheckResult("b", ("E2E-01",), "NOT_EXECUTED", "needs live data"),
        ]
        report = build_report(results)
        assert report["summary"]["passed"] == 1
        assert report["summary"]["not_executed"] == 1
        assert report["summary"]["total"] == 2
        assert "1/2 checks passed" in report["summary"]["passed_denominator"]
        assert "NOT EXECUTED" in report["summary"]["passed_denominator"]

    def test_not_executed_blocks_prod_ready(self) -> None:
        report = build_report([CheckResult("b", ("E2E-01",), "NOT_EXECUTED", "needs live data")])
        assert report["prod_ready"] is False

    def test_failure_blocks_prod_ready(self) -> None:
        report = build_report([CheckResult("b", ("CLI-02",), "FAIL", "missing command")])
        assert report["prod_ready"] is False

    def test_all_pass_is_prod_ready(self) -> None:
        report = build_report([CheckResult("a", ("WS-01",), "PASS", "ok")])
        assert report["prod_ready"] is True

    def test_phase_pass_is_not_a_prod_claim(self) -> None:
        report = build_report(
            [CheckResult("a", ("WS-01",), "PASS", "ok")], phase=0
        )
        assert report["gate_passed"] is True
        assert report["prod_ready"] is False
        assert report["gate_scope"] == "PHASE 0"

    def test_denominator_is_always_named(self) -> None:
        """CLAIM-02: coverage-style claims must name their denominator."""
        report = build_report([CheckResult("a", ("WS-01",), "PASS", "ok")])
        assert "passed_denominator" in report["summary"]


@pytest.fixture(scope="module")
def results() -> list[CheckResult]:
    """Run the real gate once for the whole module."""
    return run_checks(REPO_ROOT)


class TestRealRepositoryGate:
    """Run the gate against this actual repository."""

    def test_every_check_has_a_requirement_and_detail(
        self, results: list[CheckResult]
    ) -> None:
        for result in results:
            assert result.requirements, f"{result.check_id} cites no requirement"
            assert result.detail.strip(), f"{result.check_id} has no detail"

    def test_environmental_checks_are_labelled_not_executed(
        self, results: list[CheckResult]
    ) -> None:
        by_id = {result.check_id: result for result in results}
        for check_id in (
            "documented_installation_launch",
            "adaptive_discovery",
            "identity_history_completeness",
            "exact_target_execution",
            "partial_state_and_resume",
            "real_data_evidence_freshness",
        ):
            assert by_id[check_id].status == "NOT_EXECUTED", (
                f"{check_id} must be honestly reported as NOT_EXECUTED, "
                "never assumed to pass"
            )

    def test_governing_artifacts_are_readable_markdown(
        self, results: list[CheckResult]
    ) -> None:
        by_id = {result.check_id: result for result in results}
        assert by_id["governing_artifacts"].status == "PASS", by_id[
            "governing_artifacts"
        ].detail

    def test_state_ledger_parses(self, results: list[CheckResult]) -> None:
        by_id = {result.check_id: result for result in results}
        assert by_id["state_ledger"].status == "PASS", by_id["state_ledger"].detail

    def test_required_commands_are_present(self, results: list[CheckResult]) -> None:
        by_id = {result.check_id: result for result in results}
        assert by_id["command_palette"].status == "PASS", by_id["command_palette"].detail

    def test_entry_points_registered(self, results: list[CheckResult]) -> None:
        by_id = {result.check_id: result for result in results}
        assert by_id["entry_points"].status == "PASS", by_id["entry_points"].detail

    def test_ranking_formula_integrity_holds(self, results: list[CheckResult]) -> None:
        by_id = {result.check_id: result for result in results}
        assert by_id["ranking_formula_integrity"].status == "PASS", by_id[
            "ranking_formula_integrity"
        ].detail

    def test_presentation_layer_holds_no_business_logic(
        self, results: list[CheckResult]
    ) -> None:
        by_id = {result.check_id: result for result in results}
        assert by_id["canonical_routing"].status == "PASS", by_id["canonical_routing"].detail

    def test_golden_files_present(self, results: list[CheckResult]) -> None:
        by_id = {result.check_id: result for result in results}
        assert by_id["golden_tests"].status == "PASS", by_id["golden_tests"].detail

    def test_gate_exit_code_matches_report(self, results: list[CheckResult]) -> None:
        report = build_report(results)
        assert report["prod_ready"] is False, (
            "this repository still has open blockers; a green gate here would "
            "indicate the check is not actually inspecting anything"
        )


class TestCliInvocation:
    def test_json_mode_emits_a_parsable_report(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        exit_code = main(["--json", "--output", str(tmp_path / "report.json")])
        captured = capsys.readouterr().out
        report = json.loads(captured)
        assert report["report_version"] == REPORT_VERSION
        assert exit_code == 1, "gate must exit nonzero while blockers remain"
        written = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
        assert written["report_version"] == REPORT_VERSION

    def test_human_mode_labels_skipped_stages_explicitly(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main([])
        out = capsys.readouterr().out
        assert "NOT EXECUTED" in out
        assert "PROD gate: BLOCKED" in out


def _synthetic_root(tmp_path: Path, shell_source: str) -> Path:
    """Build a throwaway repository tree containing only a shell module.

    These are fixture-based negative controls: they never touch the real
    repository files, so a known-good and a known-bad case can both be proven.
    """
    shell = tmp_path / "src" / "exo_toolkit" / "hunter_shell.py"
    shell.parent.mkdir(parents=True)
    shell.write_text(shell_source, encoding="utf-8")
    return tmp_path


class TestInteractiveInputCapability:
    """UX-CMD-01 status authority: a renderer test may not stand in for a keystroke."""

    LINE_BUFFERED = (
        "def run_interactive(input_fn=input):\n"
        "    while True:\n"
        "        line = input_fn('EXO-Hunter> ')\n"
    )
    RAW_CAPABLE = (
        "import termios\n"
        "import tty\n"
        "def run_interactive():\n"
        "    tty.setcbreak(0)\n"
    )

    def test_line_buffered_loop_fails(self, tmp_path: Path) -> None:
        root = _synthetic_root(tmp_path, self.LINE_BUFFERED)
        result = _check_interactive_input_capability(root)
        assert result.status == "FAIL"
        assert "canonical mode" in result.detail

    def test_raw_mode_capability_passes(self, tmp_path: Path) -> None:
        root = _synthetic_root(tmp_path, self.RAW_CAPABLE)
        result = _check_interactive_input_capability(root)
        assert result.status == "PASS"
        assert "termios" in result.detail

    def test_missing_module_is_not_executed_not_passed(self, tmp_path: Path) -> None:
        assert _check_interactive_input_capability(tmp_path).status == "NOT_EXECUTED"

    def test_real_repository_shell_has_raw_mode_capability(self) -> None:
        """The shipped shell can take the terminal out of canonical mode.

        This asserted FAIL until the Phase 2 repair: the loop was builtins.input(),
        so no byte reached the process before Enter. It now guards against a
        regression back to a line-buffered loop.
        """
        result = _check_interactive_input_capability(REPO_ROOT)
        assert result.status == "PASS", result.detail

    def test_command_palette_check_no_longer_claims_ux_cmd_01(self) -> None:
        """A registry check must not carry the keystroke requirement."""
        from exo_toolkit.prod_check import _check_command_palette

        assert "UX-CMD-01" not in _check_command_palette().requirements


class TestGuidedEntryWiring:
    """PIPE-02: contract-required surfaces must not be reachable only from tests."""

    def test_unreferenced_surfaces_fail(self, tmp_path: Path) -> None:
        root = _synthetic_root(tmp_path, "def run_interactive():\n    return 0\n")
        result = _check_guided_entry_wiring(root)
        assert result.status == "FAIL"
        assert "GuidedEntry" in result.detail

    def test_docstring_mention_does_not_satisfy_wiring(self, tmp_path: Path) -> None:
        """A raw-text scan would wrongly pass this; the AST scan must not."""
        source = (
            '"""This shell uses GuidedEntry, render_action_preview, '
            'render_results_table."""\n'
            "def run_interactive():\n"
            "    return 0\n"
        )
        result = _check_guided_entry_wiring(_synthetic_root(tmp_path, source))
        assert result.status == "FAIL"

    def test_referenced_surfaces_pass(self, tmp_path: Path) -> None:
        source = (
            "from exo_toolkit.hunter_ux import (\n"
            "    GuidedEntry,\n"
            "    render_action_preview,\n"
            "    render_results_table,\n"
            ")\n"
            "def run_interactive():\n"
            "    return GuidedEntry, render_action_preview, render_results_table\n"
        )
        assert _check_guided_entry_wiring(_synthetic_root(tmp_path, source)).status == "PASS"

    def test_real_repository_shell_wires_the_operator_surfaces(self) -> None:
        """GuidedEntry and the action preview are reachable from the shipped shell.

        This asserted FAIL until the Phase 2 repair, when both were reachable
        only from tests and the gate itself (PIPE-02).
        """
        result = _check_guided_entry_wiring(REPO_ROOT)
        assert result.status == "PASS", result.detail


class TestPtyOperatorExperienceGate:
    """The behavioural gate must never be satisfiable without real PTY evidence."""

    def test_absent_bundle_is_not_executed(self, tmp_path: Path) -> None:
        result = _check_pty_operator_experience(tmp_path)
        assert result.status == "NOT_EXECUTED"
        assert "pseudo-terminal" in result.detail

    def test_bundle_recording_failure_fails(self, tmp_path: Path) -> None:
        bundle = tmp_path / PTY_EVIDENCE_PATH
        bundle.parent.mkdir(parents=True)
        bundle.write_text(
            json.dumps({"palette_opened_without_enter": False}), encoding="utf-8"
        )
        assert _check_pty_operator_experience(tmp_path).status == "FAIL"

    def test_bundle_recording_keystroke_palette_passes(self, tmp_path: Path) -> None:
        bundle = tmp_path / PTY_EVIDENCE_PATH
        bundle.parent.mkdir(parents=True)
        bundle.write_text(
            json.dumps({"palette_opened_without_enter": True}), encoding="utf-8"
        )
        assert _check_pty_operator_experience(tmp_path).status == "PASS"

    def test_malformed_bundle_fails_rather_than_passing(self, tmp_path: Path) -> None:
        bundle = tmp_path / PTY_EVIDENCE_PATH
        bundle.parent.mkdir(parents=True)
        bundle.write_text("{not json", encoding="utf-8")
        assert _check_pty_operator_experience(tmp_path).status == "FAIL"


def _fake_checkout(tmp_path: Path) -> Path:
    """Create a directory carrying both repository root markers."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    (tmp_path / "docs").mkdir(exist_ok=True)
    (tmp_path / "docs" / "HUNTER_PROD_CONTRACT.md").write_text("", encoding="utf-8")
    return tmp_path


class TestRepositoryRootResolution:
    """LAUNCH-02: a wheel install must not misreport a lookup miss as a defect."""

    def test_real_repository_is_found_from_the_source_checkout(self) -> None:
        assert _repo_root() == REPO_ROOT

    def test_walks_up_from_working_directory_when_package_is_elsewhere(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Simulates the wheel surface: the package sits outside any checkout,
        so the root must be recovered by walking up from the working directory."""
        checkout = _fake_checkout(tmp_path / "checkout")
        nested = checkout / "a" / "b"
        nested.mkdir(parents=True)
        # Force the module-relative probe to miss, as it does from site-packages.
        real = _looks_like_repo
        monkeypatch.setattr(
            "exo_toolkit.prod_check._looks_like_repo",
            lambda path: False if path == REPO_ROOT else real(path),
        )
        assert _repo_root(start=nested) == checkout.resolve()

    def test_start_directory_outside_any_checkout_returns_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real = _looks_like_repo
        monkeypatch.setattr(
            "exo_toolkit.prod_check._looks_like_repo",
            lambda path: False if path == REPO_ROOT else real(path),
        )
        bare = tmp_path / "bare"
        bare.mkdir()
        assert _repo_root(start=bare) is None

    def test_no_checkout_reports_one_honest_not_executed_result(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A wheel-only environment must not emit misleading artifact FAILs."""
        monkeypatch.setattr("exo_toolkit.prod_check._repo_root", lambda: None)
        results = run_checks(None)
        assert len(results) == 1
        assert results[0].check_id == "repository_checkout"
        assert results[0].status == "NOT_EXECUTED"
        assert not build_report(results)["prod_ready"]


class TestSiblingIsolationSelfExclusion:
    """The gate must exclude itself by module location, not by running path."""

    MARKER_SOURCE = "MARKERS = ('NEOHunter', 'TechnoHunter')\n"

    def test_gate_copy_at_a_different_path_is_still_self_excluded(
        self, tmp_path: Path
    ) -> None:
        """Reproduces the wheel-install false positive: the scanned copy of
        prod_check.py lives at a different absolute path than the running one."""
        pkg = tmp_path / "src" / "exo_toolkit"
        pkg.mkdir(parents=True)
        (pkg / "prod_check.py").write_text(self.MARKER_SOURCE, encoding="utf-8")
        assert _check_sibling_write_isolation(tmp_path).status == "PASS"

    def _write_runtime(self, tmp_path: Path, source: str) -> Path:
        pkg = tmp_path / "src" / "exo_toolkit"
        pkg.mkdir(parents=True)
        (pkg / "prod_check.py").write_text(self.MARKER_SOURCE, encoding="utf-8")
        (pkg / "runtime.py").write_text(source, encoding="utf-8")
        return tmp_path

    def test_read_only_sibling_reference_is_permitted(self, tmp_path: Path) -> None:
        root = self._write_runtime(
            tmp_path,
            "from pathlib import Path\n"
            "PATH = Path('../2026 Technosignatures/history.json')\n"
            "PAYLOAD = PATH.read_text(encoding='utf-8')\n",
        )
        result = _check_sibling_write_isolation(root)
        assert result.status == "PASS", result.detail

    @pytest.mark.parametrize(
        "source, operation",
        [
            (
                "from pathlib import Path\n"
                "PATH = Path('../2026 Technosignatures/history.json')\n"
                "PATH.write_text('{}')\n",
                "write_text",
            ),
            (
                "from pathlib import Path\n"
                "PATH = Path('../2026 Near Earth Objects/history.json')\n"
                "open(PATH, 'w').write('{}')\n",
                "open-write",
            ),
            (
                "import subprocess\n"
                "from pathlib import Path\n"
                "ROOT = Path('../2026 Technosignatures')\n"
                "subprocess.run(['git', '-C', ROOT, 'status'])\n",
                "sibling-command",
            ),
            (
                "import sys\n"
                "from pathlib import Path\n"
                "ROOT = Path('../2026 Near Earth Objects')\n"
                "sys.path.insert(0, ROOT)\n",
                "runtime-import-path",
            ),
            (
                "from exo_toolkit.hunter_cross_project_history import "
                "sibling_history_export_path\n"
                "sibling_history_export_path('techno_hunter').unlink()\n",
                "unlink",
            ),
        ],
    )
    def test_sibling_mutation_or_runtime_coupling_fails(
        self, tmp_path: Path, source: str, operation: str
    ) -> None:
        self._write_runtime(tmp_path, source)
        result = _check_sibling_write_isolation(tmp_path)
        assert result.status == "FAIL"
        assert operation in result.detail

    def test_hardcoded_absolute_sibling_path_fails_even_for_a_read(
        self, tmp_path: Path
    ) -> None:
        root = self._write_runtime(
            tmp_path,
            "from pathlib import Path\n"
            "Path('/Users/operator/2026 Technosignatures/history.json').read_text()\n",
        )
        result = _check_sibling_write_isolation(root)
        assert result.status == "FAIL"
        assert "hardcoded-absolute-sibling-path" in result.detail


def _write_ledger(tmp_path: Path, *, overrides: dict[str, object] | None = None) -> Path:
    root = tmp_path
    ledger = {
        "artifact": "configs/HUNTER_PROD_STATE.json",
        "schema_version": LEDGER_SCHEMA_VERSION,
        "contract_version": "HUNTER-PROD-2026-07-30.3",
        "active_phase": "PHASE 0",
        "implementation_state": "BLOCKING",
        "allowed_implementation_states": [
            "BLOCKING",
            "IN_PROGRESS",
            "IMPLEMENTED_NOT_VERIFIED",
        ],
        "gate_execution_state": "EXECUTED",
        "gate_result": "FAIL",
        "prod_status": None,
        "verification_authority": {"verified_and_prod_writer": STATE_WRITER},
        "requirements": {
            "CLAIM-04": {
                "implementation_state": "BLOCKING",
                "gate_execution_state": "EXECUTED",
                "gate_result": "FAIL",
            }
        },
    }
    ledger.update(overrides or {})
    path = root / "configs" / "HUNTER_PROD_STATE.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(ledger), encoding="utf-8")
    return root


class TestStateAuthority:
    """CLAIM-04: prose or an agent-authored string cannot verify a requirement."""

    def test_coherent_separated_ledger_passes(self, tmp_path: Path) -> None:
        root = _write_ledger(tmp_path)
        assert _check_state_ledger(root).status == "PASS"

    def test_compound_active_phase_fails(self, tmp_path: Path) -> None:
        root = _write_ledger(tmp_path, overrides={"active_phase": "PHASE 0; PHASE 2"})
        result = _check_state_ledger(root)
        assert result.status == "FAIL"
        assert "exactly one active phase" in result.detail

    def test_ambiguous_requirement_status_fails(self, tmp_path: Path) -> None:
        root = _write_ledger(
            tmp_path,
            overrides={
                "requirements": {
                    "CLAIM-04": {
                        "status": "VERIFIED",
                        "implementation_state": "IMPLEMENTED_NOT_VERIFIED",
                        "gate_execution_state": "EXECUTED",
                        "gate_result": "PASS",
                    }
                }
            },
        )
        result = _check_state_ledger(root)
        assert result.status == "FAIL"
        assert "ambiguous agent-writable status" in result.detail

    def test_agent_authored_prod_alias_fails(self, tmp_path: Path) -> None:
        root = _write_ledger(tmp_path, overrides={"prod_status": "PROD_ACCEPTED"})
        assert _check_state_ledger(root).status == "FAIL"

    def test_verified_without_machine_evidence_fails(self, tmp_path: Path) -> None:
        root = _write_ledger(
            tmp_path,
            overrides={
                "requirements": {
                    "CLAIM-04": {
                        "implementation_state": "IMPLEMENTED_NOT_VERIFIED",
                        "gate_execution_state": "EXECUTED",
                        "gate_result": "PASS",
                        "verification_state": "VERIFIED",
                    }
                }
            },
        )
        result = _check_state_ledger(root)
        assert result.status == "FAIL"
        assert "complete passing machine evidence" in result.detail

    def test_missing_writer_fails_authority(self, tmp_path: Path) -> None:
        root = _write_ledger(tmp_path)
        result = _check_state_authority(root)
        assert result.status == "FAIL"
        assert "prod_state.py is missing" in result.detail

    def test_unbound_writer_fails_authority(self, tmp_path: Path) -> None:
        root = _write_ledger(
            tmp_path,
            overrides={"verification_authority": {"verified_and_prod_writer": None}},
        )
        writer = root / "src" / "exo_toolkit" / "prod_state.py"
        writer.parent.mkdir(parents=True)
        writer.write_text("def update_state_from_report():\n    pass\n", encoding="utf-8")
        assert _check_state_authority(root).status == "FAIL"

    def test_bound_concrete_writer_passes_structural_authority(self, tmp_path: Path) -> None:
        root = _write_ledger(tmp_path)
        writer = root / "src" / "exo_toolkit" / "prod_state.py"
        writer.parent.mkdir(parents=True)
        writer.write_text(
            "def update_state_from_report(*, root, report, phase, **kwargs):\n"
            "    return None\n",
            encoding="utf-8",
        )
        assert _check_state_authority(root).status == "PASS"


class TestDeterministicStateWriter:
    """The bound writer derives state from results and fails closed."""

    ENVIRONMENT = {
        "platform": "test-platform",
        "python": "3.14.3",
        "python_executable": "/test/python",
        "resolved_executable": "/test/EXO-Hunter",
    }
    HASHES = {"src/exo_toolkit/prod_check.py": "a" * 64}

    @staticmethod
    def _writer():
        module = importlib.import_module("exo_toolkit.prod_state")
        return module.update_state_from_report

    def _run(
        self,
        root: Path,
        result: CheckResult,
        *,
        phase: int = 0,
        full_prod: bool = False,
    ) -> dict[str, object]:
        report = build_report(
            [result],
            commit="abc123",
            phase=None if full_prod else phase,
        )
        self._writer()(
            root=root,
            report=report,
            phase=phase,
            command="prod-check --phase 0 --update-state --output evidence.json",
            evidence_path="evidence.json",
            gate_hashes=self.HASHES,
            environment=self.ENVIRONMENT,
        )
        return json.loads(
            (root / "configs" / "HUNTER_PROD_STATE.json").read_text(encoding="utf-8")
        )

    def test_pass_is_verified_and_advances_without_claiming_prod(self, tmp_path: Path) -> None:
        root = _write_ledger(tmp_path)
        state = self._run(root, CheckResult("state_ledger", ("CLAIM-04",), "PASS", "ok"))
        assert state["gate_result"] == "PASS"
        assert state["gate_exit_code"] == 0
        assert state["active_phase"] == "PHASE 1"
        assert state["prod_status"] is None
        assert state["phase_results"]["PHASE 0"]["verification_state"] == "VERIFIED"
        requirement = state["requirements"]["CLAIM-04"]
        assert requirement["verification_state"] == "VERIFIED"
        assert requirement["tested_code_identity"]["git_head_sha"] == "abc123"
        assert requirement["gate_hashes"] == self.HASHES

    @pytest.mark.parametrize("status", ["FAIL", "NOT_EXECUTED"])
    def test_nonpass_is_blocking_nonzero_and_never_verified(
        self, tmp_path: Path, status: str
    ) -> None:
        root = _write_ledger(tmp_path)
        state = self._run(
            root,
            CheckResult("state_ledger", ("CLAIM-04",), status, "blocked"),  # type: ignore[arg-type]
        )
        assert state["gate_result"] == status
        assert state["gate_exit_code"] == 1
        assert state["active_phase"] == "PHASE 0"
        assert state["prod_status"] is None
        assert state["phase_results"]["PHASE 0"]["verification_state"] is None
        assert state["requirements"]["CLAIM-04"]["verification_state"] is None

    def test_only_full_phase_six_pass_may_write_prod(self, tmp_path: Path) -> None:
        root = _write_ledger(tmp_path, overrides={"active_phase": "PHASE 6"})
        state = self._run(
            root,
            CheckResult("full_prod", ("CLAIM-04",), "PASS", "ok"),
            phase=6,
            full_prod=True,
        )
        assert state["prod_status"] == "PROD"
        assert state["phase_results"]["PHASE 6"]["verification_state"] == "VERIFIED"
        assert state["active_phase"] == "PHASE 7"

    def test_out_of_order_phase_write_is_rejected(self, tmp_path: Path) -> None:
        root = _write_ledger(tmp_path)
        before = (root / "configs" / "HUNTER_PROD_STATE.json").read_bytes()
        with pytest.raises(ValueError, match="active phase"):
            self._run(
                root,
                CheckResult("phase_two", ("CLAIM-04",), "PASS", "ok"),
                phase=2,
            )
        assert (root / "configs" / "HUNTER_PROD_STATE.json").read_bytes() == before
