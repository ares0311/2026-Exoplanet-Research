"""Detect suspicious incomplete production code in src/ and Skills/.

Flags, per AGENTS.md's "No Fake Completion" directive:
  - functions/methods whose entire body is a bare `pass` (docstring-only
    bodies are exempt; `except: pass` fallback blocks are not function
    bodies and are never flagged)
  - `raise NotImplementedError(...)`
  - `TODO` / `FIXME` comment markers

Legitimate abstract interfaces are exempt automatically: a function/method
decorated with `@abstractmethod`/`@abstractproperty`, or one whose body is
literally `...` (the conventional `Protocol`/stub-file placeholder), is not
flagged. Anything else that is intentionally incomplete (a deliberate
extension point, a test double, generated code) must carry an inline
`# allow-stub: <reason>` comment on the `def`/`raise`/`TODO` line — a narrow,
documented, per-line exception rather than a broad ignore.

This is presence-only detection. It supplements, but can never replace,
behavioral testing (mypy/ruff/pytest) — a scan finding nothing "suspicious"
is not evidence that the code is correct, only that it is not conspicuously
incomplete. See AGENTS.md "No Fake Completion" and "Verify Behavior, Not
Presence".
"""
from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCAN_ROOTS = ("src", "Skills")
EXCLUDE_DIR_NAMES = {"__pycache__"}
# This checker's own source intentionally documents the marker words above;
# exclude it from self-scanning rather than obscuring the words in prose.
EXCLUDE_FILES = {Path(__file__).resolve()}
ALLOWLIST_MARKER = "# allow-stub:"
_ABSTRACT_DECORATOR_NAMES = {"abstractmethod", "abstractproperty"}
_TODO_MARKERS = ("TODO", "FIXME")


@dataclass(frozen=True)
class Violation:
    """One detected incomplete-implementation finding."""

    path: str
    line: int
    kind: str  # "bare_pass_stub" | "not_implemented" | "todo_marker"
    detail: str

    def format(self) -> str:
        return f"{self.path}:{self.line}: [{self.kind}] {self.detail}"


def _decorator_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    return None


def _is_abstract(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return any(
        _decorator_name(decorator) in _ABSTRACT_DECORATOR_NAMES
        for decorator in node.decorator_list
    )


def _is_docstring_expr(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def _non_docstring_body(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.stmt]:
    body = node.body
    if body and _is_docstring_expr(body[0]):
        return body[1:]
    return list(body)


def _line_has_allowlist_marker(source_lines: list[str], line_no: int) -> bool:
    if 1 <= line_no <= len(source_lines):
        return ALLOWLIST_MARKER in source_lines[line_no - 1]
    return False


def _scan_function(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    rel_path: str,
    source_lines: list[str],
) -> Violation | None:
    if _is_abstract(node):
        return None
    body = _non_docstring_body(node)
    if len(body) != 1:
        return None
    stmt = body[0]
    is_ellipsis = (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and stmt.value.value is Ellipsis
    )
    if is_ellipsis:
        return None  # Protocol/stub-file convention: `def foo(...): ...`
    if not isinstance(stmt, ast.Pass):
        return None
    if _line_has_allowlist_marker(source_lines, node.lineno):
        return None
    return Violation(
        path=rel_path,
        line=node.lineno,
        kind="bare_pass_stub",
        detail=f"'{node.name}' has no body other than 'pass'",
    )


def _scan_raise(
    node: ast.Raise, *, rel_path: str, source_lines: list[str]
) -> Violation | None:
    exc = node.exc
    name = None
    if isinstance(exc, ast.Name):
        name = exc.id
    elif isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name):
        name = exc.func.id
    if name != "NotImplementedError":
        return None
    if _line_has_allowlist_marker(source_lines, node.lineno):
        return None
    return Violation(
        path=rel_path, line=node.lineno, kind="not_implemented", detail="raises NotImplementedError"
    )


def _scan_todo_markers(*, rel_path: str, source_lines: list[str]) -> list[Violation]:
    violations = []
    for index, line in enumerate(source_lines, 1):
        if ALLOWLIST_MARKER in line:
            continue
        for marker in _TODO_MARKERS:
            if marker in line:
                violations.append(
                    Violation(
                        path=rel_path,
                        line=index,
                        kind="todo_marker",
                        detail=f"contains unresolved '{marker}' marker",
                    )
                )
                break
    return violations


def scan_source(source: str, *, rel_path: str) -> list[Violation]:
    """Scan one file's source text and return every finding.

    Raises SyntaxError-derived exceptions unmodified on unparseable source
    (fail loudly rather than silently skipping a broken file).
    """
    tree = ast.parse(source, filename=rel_path)
    source_lines = source.splitlines()
    violations: list[Violation] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            found = _scan_function(node, rel_path=rel_path, source_lines=source_lines)
            if found is not None:
                violations.append(found)
        elif isinstance(node, ast.Raise):
            found = _scan_raise(node, rel_path=rel_path, source_lines=source_lines)
            if found is not None:
                violations.append(found)
    violations.extend(_scan_todo_markers(rel_path=rel_path, source_lines=source_lines))
    return sorted(violations, key=lambda v: (v.path, v.line, v.kind))


def _iter_python_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for scan_root_name in SCAN_ROOTS:
        scan_root = root / scan_root_name
        if not scan_root.is_dir():
            continue
        for path in sorted(scan_root.rglob("*.py")):
            if any(part in EXCLUDE_DIR_NAMES for part in path.parts):
                continue
            if path.resolve() in EXCLUDE_FILES:
                continue
            files.append(path)
    return files


def find_violations(root: Path) -> list[Violation]:
    """Scan every production Python file under root's configured SCAN_ROOTS."""
    all_violations: list[Violation] = []
    for path in _iter_python_files(root):
        rel_path = str(path.relative_to(root))
        source = path.read_text(encoding="utf-8")
        all_violations.extend(scan_source(source, rel_path=rel_path))
    return sorted(all_violations, key=lambda v: (v.path, v.line, v.kind))


def main(argv: list[str] | None = None) -> int:
    del argv
    violations = find_violations(REPO_ROOT)
    if not violations:
        print("check_incomplete_implementations: 0 findings across src/ and Skills/.", flush=True)
        return 0
    print(
        f"check_incomplete_implementations: {len(violations)} suspicious finding(s):",
        flush=True,
    )
    for violation in violations:
        print(f"  {violation.format()}", flush=True)
    print(
        f"  -> resolve the required work, or mark a deliberate exception with "
        f"'{ALLOWLIST_MARKER} <reason>' on the offending line.",
        flush=True,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
