"""Tests for the project-scoped MCP bootstrap server."""
from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from mcp_bootstrap_server import handle_request, is_allowed_project_path, project_root


def test_project_root_defaults_to_repository() -> None:
    assert project_root().name == "2026 Exoplanet Research"


def test_project_files_reject_runtime_artifacts() -> None:
    root = project_root()
    assert is_allowed_project_path(root / "README.md")
    assert not is_allowed_project_path(root / "logs" / "background_search.sqlite3")
    assert not is_allowed_project_path(root / "data" / "raw" / "example.csv")
    assert not is_allowed_project_path(root / ".env")


def test_project_files_tool_lists_safe_files() -> None:
    response = handle_request(
        "project_files",
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "list_project_files", "arguments": {"limit": 20}},
        },
    )

    assert response is not None
    text = response["result"]["content"][0]["text"]  # type: ignore[index]
    listed = json.loads(text)
    assert "README.md" in listed
    assert not any(path.startswith("logs/") for path in listed)


def test_git_read_tool_list_is_fixed() -> None:
    response = handle_request(
        "git_read",
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    )

    assert response is not None
    tools = response["result"]["tools"]  # type: ignore[index]
    names = {tool["name"] for tool in tools}
    assert names == {
        "status_short_branch",
        "diff",
        "diff_staged",
        "log_recent",
        "branch_current",
    }


def test_exo_guard_tool_list_is_fixed() -> None:
    response = handle_request(
        "exo_guard",
        {"jsonrpc": "2.0", "id": 3, "method": "tools/list"},
    )

    assert response is not None
    tools = response["result"]["tools"]  # type: ignore[index]
    names = {tool["name"] for tool in tools}
    assert names == {
        "ruff_check",
        "mypy_src",
        "pytest_default",
        "pytest_cov",
        "background_run_once_dry_run",
        "run_summary",
        "sqlite_integrity",
    }


def test_unknown_tool_returns_error() -> None:
    response = handle_request(
        "exo_guard",
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "arbitrary_shell", "arguments": {}},
        },
    )

    assert response is not None
    assert "error" in response
    assert "unsupported" in response["error"]["message"]  # type: ignore[index]


def test_mcp_json_config_has_only_bootstrap_servers() -> None:
    config = json.loads((project_root() / ".mcp.json").read_text())
    assert set(config["mcpServers"]) == {
        "exo_project_files",
        "exo_git_read",
        "exo_guard",
    }
    for server in config["mcpServers"].values():
        assert server["command"] == "python3"
        assert server["args"][0] == "Skills/mcp_bootstrap_server.py"
        assert server["env"] == {"EXO_RESEARCH_ROOT": "."}


def test_codex_config_has_only_bootstrap_servers() -> None:
    config = tomllib.loads((project_root() / ".codex" / "config.toml").read_text())
    servers = config["mcp_servers"]
    assert set(servers) == {
        "exo_project_files",
        "exo_git_read",
        "exo_guard",
    }
    for server in servers.values():
        assert server["command"] == "python3"
        assert server["args"][0] == "Skills/mcp_bootstrap_server.py"
        assert server["env"] == {"EXO_RESEARCH_ROOT": "."}
