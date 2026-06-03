"""Project-scoped MCP servers for the Exoplanet Research bootstrap.

The server intentionally exposes only narrow, fixed operations:

- project file inspection inside this repository
- read-only git inspection commands
- fixed validation commands from the bootstrap policy

It does not expose arbitrary shell execution or live-network commands.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

ServerMode = Literal["project_files", "git_read", "exo_guard"]

_EXCLUDED_PARTS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "data",
    "logs",
    "reports",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".claude",
}
_SECRET_NAMES = {
    ".env",
    ".netrc",
    "id_rsa",
    "id_ed25519",
    "credentials",
    "credentials.json",
    "token",
    "tokens",
}
_SECRET_SUFFIXES = {
    ".pem",
    ".key",
    ".p12",
}


@dataclass(frozen=True)
class CommandResult:
    command: tuple[str, ...]
    exit_code: int
    stdout: str
    stderr: str


def project_root() -> Path:
    """Return the configured project root."""
    configured = os.environ.get("EXO_RESEARCH_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parent.parent


def is_allowed_project_path(path: Path, *, allow_runtime: bool = False) -> bool:
    """Return True when *path* is safe for project-file MCP access."""
    root = project_root()
    try:
        resolved = path.resolve()
        resolved.relative_to(root)
    except ValueError:
        return False

    rel_parts = resolved.relative_to(root).parts
    if not allow_runtime and any(part in _EXCLUDED_PARTS for part in rel_parts):
        return False
    if any(part.lower() in _SECRET_NAMES for part in rel_parts):
        return False
    return not any(
        part.lower().endswith(suffix)
        for part in rel_parts
        for suffix in _SECRET_SUFFIXES
    )


def read_project_file(relative_path: str) -> str:
    """Read a safe text file from inside the repository."""
    if Path(relative_path).is_absolute():
        raise ValueError("absolute paths are not allowed")
    target = project_root() / relative_path
    if not is_allowed_project_path(target):
        raise ValueError(f"path is outside the allowed project scope: {relative_path}")
    if not target.is_file():
        raise ValueError(f"path is not a file: {relative_path}")
    return target.read_text(errors="replace")


def list_project_files(*, limit: int = 200) -> list[str]:
    """List safe project files, excluding runtime artifacts and credentials."""
    root = project_root()
    files: list[str] = []
    for path in root.rglob("*"):
        if len(files) >= limit:
            break
        if path.is_file() and is_allowed_project_path(path):
            files.append(path.relative_to(root).as_posix())
    return files


def _run_fixed_command(
    command: Iterable[str],
    *,
    env: dict[str, str] | None = None,
) -> CommandResult:
    args = tuple(command)
    result = subprocess.run(
        args,
        cwd=project_root(),
        env={**os.environ, **(env or {})},
        text=True,
        capture_output=True,
        check=False,
    )
    return CommandResult(
        command=args,
        exit_code=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )


def run_git_read_command(name: str) -> CommandResult:
    """Run one allowed read-only git command."""
    commands: dict[str, tuple[str, ...]] = {
        "status_short_branch": ("git", "status", "--short", "--branch"),
        "diff": ("git", "diff"),
        "diff_staged": ("git", "diff", "--staged"),
        "log_recent": ("git", "log", "--oneline", "--decorate", "-n", "20"),
        "branch_current": ("git", "branch", "--show-current"),
    }
    if name not in commands:
        raise ValueError(f"unsupported git command: {name}")
    return _run_fixed_command(commands[name])


def _venv_python() -> str:
    candidate = project_root() / ".venv" / "bin" / "python"
    return str(candidate) if candidate.exists() else sys.executable


def _ruff_command() -> tuple[str, ...]:
    candidate = project_root() / ".venv" / "bin" / "ruff"
    if candidate.exists():
        return (str(candidate), "check", ".")
    found = shutil.which("ruff")
    if found is not None:
        return (found, "check", ".")
    return (_venv_python(), "-m", "ruff", "check", ".")


def _exo_command(*args: str) -> tuple[str, ...]:
    found = shutil.which("exo")
    if found is not None:
        return (found, *args)
    return (_venv_python(), "-m", "exo_toolkit.cli", *args)


def run_exo_guard_command(name: str) -> CommandResult:
    """Run one fixed validation command from the bootstrap policy."""
    py = _venv_python()
    commands: dict[str, tuple[tuple[str, ...], dict[str, str] | None]] = {
        "ruff_check": (_ruff_command(), None),
        "mypy_src": ((py, "-m", "mypy", "src"), None),
        "pytest_default": ((py, "-m", "pytest"), {"PYTHONPATH": "src"}),
        "pytest_cov": (
            (py, "-m", "pytest", "--cov=exo_toolkit", "--cov-report=term-missing"),
            {"PYTHONPATH": "src"},
        ),
        "background_run_once_dry_run": (
            _exo_command("background-run-once", "--dry-run"),
            {"PYTHONPATH": "src"},
        ),
        "run_summary": (_exo_command("run-summary"), {"PYTHONPATH": "src"}),
        "sqlite_integrity": (_exo_command("sqlite-integrity"), {"PYTHONPATH": "src"}),
    }
    if name not in commands:
        raise ValueError(f"unsupported exo_guard command: {name}")
    command, env = commands[name]
    return _run_fixed_command(command, env=env)


def _text_response(text: str) -> dict[str, list[dict[str, str]]]:
    return {"content": [{"type": "text", "text": text}]}


def _command_text(result: CommandResult) -> str:
    return json.dumps(
        {
            "command": list(result.command),
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
        },
        indent=2,
    )


def _tool_defs(mode: ServerMode) -> list[dict[str, object]]:
    if mode == "project_files":
        return [
            {
                "name": "list_project_files",
                "description": "List safe project files inside this repository.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 1000}},
                    "additionalProperties": False,
                },
            },
            {
                "name": "read_project_file",
                "description": "Read a safe text file from inside this repository.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
        ]
    if mode == "git_read":
        return [
            {
                "name": name,
                "description": f"Run fixed read-only git command: {name}.",
                "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
            }
            for name in (
                "status_short_branch",
                "diff",
                "diff_staged",
                "log_recent",
                "branch_current",
            )
        ]
    return [
        {
            "name": name,
            "description": f"Run fixed Exoplanet Research validation command: {name}.",
            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        }
        for name in (
            "ruff_check",
            "mypy_src",
            "pytest_default",
            "pytest_cov",
            "background_run_once_dry_run",
            "run_summary",
            "sqlite_integrity",
        )
    ]


def _call_tool(mode: ServerMode, name: str, arguments: dict[str, object]) -> dict[str, object]:
    if mode == "project_files":
        if name == "list_project_files":
            limit = int(arguments.get("limit", 200))
            return _text_response(json.dumps(list_project_files(limit=limit), indent=2))
        if name == "read_project_file":
            path = arguments.get("path")
            if not isinstance(path, str):
                raise ValueError("path must be a string")
            return _text_response(read_project_file(path))
    elif mode == "git_read":
        return _text_response(_command_text(run_git_read_command(name)))
    elif mode == "exo_guard":
        return _text_response(_command_text(run_exo_guard_command(name)))
    raise ValueError(f"unsupported tool for {mode}: {name}")


def handle_request(mode: ServerMode, request: dict[str, object]) -> dict[str, object] | None:
    """Handle one JSON-RPC MCP request."""
    method = request.get("method")
    request_id = request.get("id")
    if method == "notifications/initialized":
        return None
    try:
        if method == "initialize":
            result: dict[str, object] = {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": f"exo_{mode}", "version": "0.1.0"},
            }
        elif method == "tools/list":
            result = {"tools": _tool_defs(mode)}
        elif method == "tools/call":
            params = request.get("params")
            if not isinstance(params, dict):
                raise ValueError("params must be an object")
            name = params.get("name")
            if not isinstance(name, str):
                raise ValueError("tool name must be a string")
            arguments = params.get("arguments") or {}
            if not isinstance(arguments, dict):
                raise ValueError("arguments must be an object")
            result = _call_tool(mode, name, arguments)
        elif method == "ping":
            result = {}
        else:
            raise ValueError(f"unsupported method: {method}")
        return {"jsonrpc": "2.0", "id": request_id, "result": result}
    except Exception as exc:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32000, "message": f"{type(exc).__name__}: {exc}"},
        }


def serve_stdio(mode: ServerMode, *, stdin: Any = sys.stdin, stdout: Any = sys.stdout) -> None:
    """Serve MCP JSON-RPC messages over stdio."""
    for line in stdin:
        if not line.strip():
            continue
        response = handle_request(mode, json.loads(line))
        if response is not None:
            stdout.write(json.dumps(response) + "\n")
            stdout.flush()


def main(argv: list[str] | None = None) -> int:
    """Run the selected MCP server role."""
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1 or args[0] not in {"project_files", "git_read", "exo_guard"}:
        print("usage: mcp_bootstrap_server.py {project_files|git_read|exo_guard}", file=sys.stderr)
        return 2
    serve_stdio(args[0])  # type: ignore[arg-type]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
