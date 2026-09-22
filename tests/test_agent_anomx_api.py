import json

import pytest

from anomx.agent.base.tools import ToolExecutionContext
from anomx.agent.helpers.anomx_api import (
    AnomxApiConnection,
    AnomxApiError,
    _build_url,
    _response_preview,
)
from anomx.agent.helpers.mode import AgentMode
from anomx.agent.helpers.tool_manager import ApprovalChoice
from anomx.agent.runtime import AgentRuntime, RuntimeCallbacks
from anomx.agent.store import AnomxHome


def connected_runtime(tmp_path, mode: AgentMode, *, scoped: bool = False) -> AgentRuntime:
    home = AnomxHome(tmp_path / "home")
    home.ensure()
    home.set_platform_connection(
        url="https://platform.example.test",
        token="platform-token",
        user_email="user@example.test",
        organization_url="example",
        hostname="agent-host",
    )
    return AgentRuntime(home, tmp_path, mode=mode, background_api_scoped=scoped)


def execute_api(runtime, *, method, path, callbacks=None, body=None):
    tool = runtime._tool_for_call("use_anomx_api")
    assert tool is not None
    return json.loads(
        tool.execute(
            {
                "statement": "Calling the platform API",
                "method": method,
                "path": path,
                "body": body,
            },
            ToolExecutionContext(
                runtime=runtime,
                callbacks=callbacks or RuntimeCallbacks(),
            ),
        )
    )


def test_recommend_mode_exposes_only_read_and_recommendation_tools(tmp_path):
    runtime = connected_runtime(tmp_path, AgentMode.RECOMMEND)
    tools = {tool.name for tool in runtime._available_tools()}

    assert "use_anomx_api" in tools
    assert "ask_question" not in tools
    assert "get_background_runs" in tools
    assert "start_process" not in tools
    assert "memorize" not in tools
    assert "send_feedback" not in tools


def test_recommend_mode_allows_only_recommendation_creation(tmp_path, monkeypatch):
    runtime = connected_runtime(tmp_path, AgentMode.RECOMMEND)
    calls = []

    def fake_call(_connection, **kwargs):
        calls.append(kwargs)
        return {"ok": True, "response": {"id": "recommendation-id"}}

    monkeypatch.setattr("anomx.agent.tools.use_anomx_api.call_anomx_api", fake_call)

    created = execute_api(runtime, method="POST", path="/recommendations")
    created_absolute = execute_api(
        runtime,
        method="POST",
        path="https://platform.example.test/api/recommendations",
    )
    blocked_update = execute_api(
        runtime, method="PATCH", path="/recommendations/value", body={"name": "Updated"}
    )
    blocked_write = execute_api(runtime, method="POST", path="/folders")
    blocked_outside_api = execute_api(
        runtime,
        method="POST",
        path="https://platform.example.test/recommendations",
    )

    assert created["ok"] is True
    assert created_absolute["ok"] is True
    assert [call["path"] for call in calls] == [
        "/recommendations",
        "https://platform.example.test/api/recommendations",
    ]
    assert blocked_update["blocked_by_mode"] is True
    assert blocked_write["blocked_by_mode"] is True
    assert blocked_outside_api["blocked_by_mode"] is True


def test_standard_api_write_uses_command_approval_pipeline(tmp_path, monkeypatch):
    runtime = connected_runtime(tmp_path, AgentMode.STANDARD)
    approvals = []

    monkeypatch.setattr(
        "anomx.agent.tools.use_anomx_api.call_anomx_api",
        lambda _connection, **_kwargs: {"ok": True, "response": {}},
    )

    result = execute_api(
        runtime,
        method="POST",
        path="/folders",
        callbacks=RuntimeCallbacks(
            approval=lambda request: approvals.append(request) or ApprovalChoice.ALLOW,
        ),
    )

    assert result["ok"] is True
    assert len(approvals) == 1
    assert approvals[0].canonical_command == "anomx-api POST /folders"


def test_api_response_preview_is_bounded_and_structured():
    preview, truncated = _response_preview(
        {"count": 100, "results": [{"value": "x" * 5_000} for _ in range(100)]}
    )

    assert truncated is True
    assert preview["count"] == 100
    assert len(preview["results"]) == 20


def test_api_absolute_urls_cannot_leave_connected_origin(tmp_path):
    connection = AnomxApiConnection(
        base_url="https://platform.example.test/api",
        token="token",
        responses_dir=tmp_path,
    )

    with pytest.raises(AnomxApiError, match="connected platform origin"):
        _build_url(connection.base_url, "https://attacker.example/api/objects", None)


def test_background_legacy_mode_and_prompt(tmp_path):
    runtime = connected_runtime(tmp_path, AgentMode.BACKGROUND)
    assert AgentMode.parse("recommend") is AgentMode.BACKGROUND
    assert "Current mode: Background." in runtime._instructions()
    assert "conservative assumptions" in runtime._instructions()
    assert runtime._tool_for_call("ask_question") is None


def test_scoped_background_api_uses_server_permissions(tmp_path, monkeypatch):
    runtime = connected_runtime(tmp_path, AgentMode.BACKGROUND, scoped=True)
    calls = []
    monkeypatch.setattr(
        "anomx.agent.tools.use_anomx_api.call_anomx_api",
        lambda connection, **kwargs: calls.append(kwargs) or {"ok": True},
    )
    result = execute_api(
        runtime, method="PATCH", path="/channels/example", body={"description": "Updated"}
    )
    assert result["ok"] is True
    assert calls[0]["method"] == "PATCH"


def test_background_resume_retains_completed_actions_without_repeating_prompt(
    tmp_path, monkeypatch,
):
    runtime = connected_runtime(tmp_path, AgentMode.BACKGROUND, scoped=True)
    session = runtime.home.create_session(tmp_path, provider="openai", model="example")
    monkeypatch.setattr(runtime, "backend_response", lambda *args, **kwargs: "")
    monkeypatch.setattr(
        "anomx.agent.tools.use_anomx_api.call_anomx_api",
        lambda *args, **kwargs: {"ok": True, "response": {"id": "created-object"}},
    )
    runtime.backend_response_for_prompt("Create a finding", parent_session_path=session.path)
    runtime._execute_tool(
        "use_anomx_api", {"method": "POST", "path": "/findings", "body": {"name": "Drift"}},
        RuntimeCallbacks(), session.path,
    )
    runtime.backend_response_for_prompt(
        "Create a finding", parent_session_path=session.path, resume=True,
    )
    messages = runtime.conversation_messages(session.path)
    assert sum(message["role"] == "user" for message in messages) == 1
    assert any("created-object" in message["content"] for message in messages)


def test_model_budget_hook_stops_before_provider_request(tmp_path):
    from anomx.agent.base.backends import BaseBackend

    runtime = connected_runtime(tmp_path, AgentMode.BACKGROUND)

    def pause():
        raise RuntimeError("budget paused")

    runtime.before_model_request = pause
    with pytest.raises(RuntimeError, match="budget paused"):
        BaseBackend(runtime)._model_request_with_retries(
            provider_key="openai", provider_label="OpenAI", env_var="OPENAI_API_KEY",
            status_callback=None,
            stream_once=lambda: pytest.fail("A paused run must not call the provider"),
        )


def test_background_denies_shell_writes_without_asking(tmp_path):
    runtime = connected_runtime(tmp_path, AgentMode.BACKGROUND, scoped=True)
    result = json.loads(runtime._execute_tool(
        "run_command", {"command": "touch changed.txt", "statement": "Change a file"},
        RuntimeCallbacks(approval=lambda request: pytest.fail("Background must not ask")),
    ))
    assert result["approved"] is False
    assert not (tmp_path / "changed.txt").exists()
