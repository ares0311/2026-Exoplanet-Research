"""Tests for Skills/check_incomplete_implementations.py.

Includes the adversarial negative controls required by AGENTS.md's
"Verify The Verification System" policy: a known-good fixture must pass
cleanly, known-bad fixtures (one per detected pattern) must each be caught,
and a malformed (unparseable) fixture must fail loudly rather than being
silently skipped. Every fixture lives in an isolated tmp_path tree — this
suite never touches the real repository's src/ or Skills/.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from Skills.check_incomplete_implementations import (
    ALLOWLIST_MARKER,
    find_violations,
    scan_source,
)


class TestScanSourceKnownGood:
    def test_clean_function_has_no_violations(self) -> None:
        source = """
def add(a: int, b: int) -> int:
    '''Add two numbers.'''
    return a + b
"""
        assert scan_source(source, rel_path="clean.py") == []

    def test_except_pass_is_not_flagged(self) -> None:
        # Regression fixture: mirrors the real, legitimate pattern found in
        # src/exo_toolkit/search.py and src/exo_toolkit/ml/cnn_scorer.py —
        # `except: pass` is a fallback block, not a stub function body.
        source = """
def load(path: str) -> str | None:
    try:
        return open(path).read()
    except (OSError, ValueError):
        pass
    return None
"""
        assert scan_source(source, rel_path="fallback.py") == []

    def test_abstractmethod_bare_pass_is_exempt(self) -> None:
        source = """
from abc import ABC, abstractmethod

class Base(ABC):
    @abstractmethod
    def compute(self) -> float:
        pass
"""
        assert scan_source(source, rel_path="abstract.py") == []

    def test_protocol_ellipsis_body_is_exempt(self) -> None:
        source = """
from typing import Protocol

class Scorer(Protocol):
    def score(self, x: float) -> float: ...
"""
        assert scan_source(source, rel_path="protocol.py") == []

    def test_docstring_only_body_is_not_bare_pass(self) -> None:
        source = """
def placeholder() -> None:
    '''Intentionally documented no-op, but not a bare pass.'''
    return None
"""
        assert scan_source(source, rel_path="documented.py") == []

    def test_identifier_containing_marker_substring_is_not_flagged(self) -> None:
        # Word-boundary regression: "AUTODOC" and "TODOIST_TOKEN" contain the
        # substring "TODO" but are not TODO markers.
        source = """
AUTODOC_ENABLED = True
TODOIST_TOKEN = "unused"
"""
        assert scan_source(source, rel_path="not_a_marker.py") == []


class TestScanSourceKnownBad:
    def test_bare_pass_stub_is_flagged(self) -> None:
        source = """
def compute_score(x: float) -> float:
    pass
"""
        violations = scan_source(source, rel_path="stub.py")
        assert len(violations) == 1
        assert violations[0].kind == "bare_pass_stub"
        assert violations[0].path == "stub.py"

    def test_docstring_plus_bare_pass_is_still_flagged(self) -> None:
        source = """
def compute_score(x: float) -> float:
    '''Docstring does not excuse an empty body.'''
    pass
"""
        violations = scan_source(source, rel_path="stub_doc.py")
        assert len(violations) == 1
        assert violations[0].kind == "bare_pass_stub"

    def test_not_implemented_error_is_flagged(self) -> None:
        source = """
def compute_score(x: float) -> float:
    raise NotImplementedError
"""
        violations = scan_source(source, rel_path="not_impl.py")
        assert len(violations) == 1
        assert violations[0].kind == "not_implemented"

    def test_not_implemented_error_call_form_is_flagged(self) -> None:
        source = """
def compute_score(x: float) -> float:
    raise NotImplementedError("fill this in")
"""
        violations = scan_source(source, rel_path="not_impl_call.py")
        assert len(violations) == 1
        assert violations[0].kind == "not_implemented"

    def test_todo_marker_is_flagged(self) -> None:
        source = """
def compute_score(x: float) -> float:
    # TODO: handle negative depths
    return x
"""
        violations = scan_source(source, rel_path="todo.py")
        assert len(violations) == 1
        assert violations[0].kind == "todo_marker"

    def test_fixme_marker_is_flagged(self) -> None:
        source = "x = 1  # FIXME this is wrong\n"
        violations = scan_source(source, rel_path="fixme.py")
        assert len(violations) == 1
        assert violations[0].kind == "todo_marker"

    def test_nested_method_stub_is_flagged(self) -> None:
        source = """
class Scorer:
    def compute(self, x: float) -> float:
        pass
"""
        violations = scan_source(source, rel_path="nested.py")
        assert len(violations) == 1
        assert violations[0].kind == "bare_pass_stub"

    def test_lowercase_todo_marker_is_flagged(self) -> None:
        source = """
def compute_score(x: float) -> float:
    # todo: handle negative depths
    return x
"""
        violations = scan_source(source, rel_path="lower_todo.py")
        assert len(violations) == 1
        assert violations[0].kind == "todo_marker"

    def test_mixed_case_fixme_marker_is_flagged(self) -> None:
        source = "x = 1  # Fixme later\n"
        violations = scan_source(source, rel_path="mixed_fixme.py")
        assert len(violations) == 1
        assert violations[0].kind == "todo_marker"

    def test_qualified_not_implemented_error_is_flagged(self) -> None:
        source = """
def compute_score(x: float) -> float:
    raise builtins.NotImplementedError("fill this in")
"""
        violations = scan_source(source, rel_path="qualified_raise.py")
        assert len(violations) == 1
        assert violations[0].kind == "not_implemented"

    def test_indirect_not_implemented_error_via_variable_is_flagged(self) -> None:
        source = """
def compute_score(x: float) -> float:
    err = NotImplementedError("fill this in")
    raise err
"""
        violations = scan_source(source, rel_path="indirect_raise.py")
        assert len(violations) == 1
        assert violations[0].kind == "not_implemented"


class TestAllowlistMarker:
    def test_allowlist_marker_suppresses_bare_pass(self) -> None:
        source = f"""
def deliberately_unfinished() -> None:  {ALLOWLIST_MARKER} tracked in ISSUE-42
    pass
"""
        assert scan_source(source, rel_path="allowed.py") == []

    def test_allowlist_marker_suppresses_not_implemented(self) -> None:
        source = f"""
def deliberately_unfinished() -> None:
    raise NotImplementedError  {ALLOWLIST_MARKER} tracked in ISSUE-42
"""
        assert scan_source(source, rel_path="allowed_raise.py") == []

    def test_allowlist_marker_suppresses_todo(self) -> None:
        source = f"x = 1  # TODO revisit {ALLOWLIST_MARKER} tracked in ISSUE-42\n"
        assert scan_source(source, rel_path="allowed_todo.py") == []

    def test_marker_only_suppresses_its_own_line(self) -> None:
        source = f"""
def deliberately_unfinished() -> None:  {ALLOWLIST_MARKER} tracked in ISSUE-42
    pass

def still_a_stub() -> None:
    pass
"""
        violations = scan_source(source, rel_path="partial_allow.py")
        assert len(violations) == 1
        assert "still_a_stub" in violations[0].detail


class TestMalformedInput:
    def test_unparseable_source_fails_loudly(self) -> None:
        with pytest.raises(SyntaxError):
            scan_source("def broken(:\n    pass\n", rel_path="broken.py")


class TestFindViolationsIntegration:
    def test_scans_src_and_skills_only(self, tmp_path: Path) -> None:
        (tmp_path / "src").mkdir()
        (tmp_path / "Skills").mkdir()
        (tmp_path / "tests").mkdir()
        (tmp_path / "src" / "bad.py").write_text("def f():\n    pass\n")
        (tmp_path / "Skills" / "bad.py").write_text("def g():\n    pass\n")
        # A stub inside tests/ must NOT be flagged — this checker targets
        # production paths only, matching its own SCAN_ROOTS configuration.
        (tmp_path / "tests" / "test_bad.py").write_text("def test_f():\n    pass\n")

        violations = find_violations(tmp_path)
        paths = {violation.path for violation in violations}
        assert paths == {"src/bad.py", "Skills/bad.py"}

    def test_empty_tree_has_no_violations(self, tmp_path: Path) -> None:
        assert find_violations(tmp_path) == []

    def test_missing_scan_roots_do_not_error(self, tmp_path: Path) -> None:
        # Neither src/ nor Skills/ exists under this root — must not raise.
        assert find_violations(tmp_path) == []
