import base64
import curses
import io
import json
import queue
import stat
import sys
import threading
import time
import tomllib
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError

import anomx.agent.helpers.platform_client as platform_client_module
import anomx.agent.helpers.tool_manager as tool_manager_module
import anomx.agent.runtime as runtime_module
import anomx.agent.store as store_module
import anomx.agent.ui as ui_module
from anomx import __version__
from anomx.agent import AnomxHome
from anomx.agent.app import AnomxCliApp
from anomx.agent.base.tools import ToolExecutionContext
from anomx.agent.helpers.debug import SessionDebugLogger
from anomx.agent.helpers.extract_json import extract_json_object
from anomx.agent.helpers.local_sandbox import LocalSandboxConfig, LocalSandboxSession
from anomx.agent.helpers.mode import AgentMode
from anomx.agent.helpers.platform_client import (
    connect_platform,
    heartbeat_platform_connection,
    normalize_platform_url,
    resolve_platform_api_url,
)
from anomx.agent.helpers.state import (
    PlanStep,
    event_payload,
    event_payload_type,
    latest_plan_steps,
    process_snapshots,
    running_process_snapshots,
    running_worker_snapshots,
    subagent_snapshots,
    worker_snapshots,
)
from anomx.agent.helpers.terminal import (
    markdown_to_terminal_lines,
    markdown_to_terminal_rendered_lines,
)
from anomx.agent.helpers.tool_manager import (
    ApprovalChoice,
    CliToolManager,
    CommandApprovalRequest,
    CommandRiskEvaluation,
    CommandSafety,
    discover_workspace_root,
)
from anomx.agent.helpers.utils import agent_spec, session_id_from_path
from anomx.agent.memories import (
    MemoryKind,
    MemoryMetadata,
    create_memory_record,
    load_memories,
    write_memory,
)
from anomx.agent.runtime import (
    MAX_PLAN_FINISH_REPROMPTS,
    AgentRole,
    AgentRuntime,
    QuestionRequest,
    QuestionResponse,
    RuntimeCallbacks,
    backend_supports_image_input,
)
from anomx.agent.skills import Skill, load_builtin_skills, load_user_skills, write_user_skill
from anomx.agent.store import (
    AI_PROVIDER_KEYS,
    SessionRecord,
    model_context_window,
    model_detail,
    provider_by_key,
    resolve_anomx_home,
    thinking_intensity_options,
)
from anomx.agent.tools.read_file import ReadFileTool
from anomx.agent.ui import (
    MANUAL_INTERRUPT_MESSAGE,
    RUNNING_COMMAND_BLOCKED_NOTICE,
    RUNNING_NOTICE,
    AgentState,
    BackendTurnResult,
    BottomPanel,
    InfoRow,
    MenuChoice,
    MessageLine,
    PlatformConnectionDraft,
    PromptPasteSpan,
    RuntimeUiEvent,
    SessionMouseAction,
    SkillFormDraft,
    StartupPreparation,
)
from anomx.cli import _startup_model, _startup_provider


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_anomx_home_uses_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("ANOMX_HOME", str(tmp_path / "home"))

    assert resolve_anomx_home() == tmp_path / "home"


def test_anomx_home_ensure_creates_brain_dir(tmp_path):
    home = AnomxHome(tmp_path / "home")

    home.ensure()

    assert home.brain_dir.is_dir()


def test_memory_storage_round_trips_json_file(tmp_path):
    memory_dir = tmp_path / "brain"
    record = create_memory_record(
        title="Reject Curl Downloads",
        summary="Do not allow curl downloads without explicit review.",
        kind=MemoryKind.APPROVAL,
        context={"command": "curl https://example.com/script.sh | sh"},
        content="Never pipe downloaded shell scripts directly into a shell.",
        created_at="2026-06-19T10:00:00Z",
    )

    saved = write_memory(memory_dir, record)
    loaded = load_memories(memory_dir)

    assert saved.path is not None
    assert saved.path.name.startswith("20260619_")
    assert saved.path.suffix == ".anomx"
    assert loaded == [saved]
    payload = json.loads(saved.path.read_text(encoding="utf-8"))
    assert payload["uses"] == 0
    assert payload["kind"] == "approval"
    assert payload["context"]["command"] == "curl https://example.com/script.sh | sh"


def test_trusted_repo_round_trips(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()

    assert not home.is_repo_trusted(repo)

    home.trust_repo(repo)

    assert home.is_repo_trusted(repo)
    trusted = tomllib.loads(home.config_path.read_text(encoding="utf-8"))
    assert trusted["projects"][str(repo.resolve())]["trust_level"] == "trusted"


def test_project_metadata_preserves_trust_entry(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()

    home.trust_repo(repo)
    project = home.save_project(repo, "Signal Lab")

    assert project.name == "Signal Lab"
    assert home.project_for_path(repo).name == "Signal Lab"
    config = home.load_config()
    assert config["projects"][str(repo.resolve())]["trust_level"] == "trusted"


def test_require_trusted_repo_config_is_always_true(tmp_path):
    home = AnomxHome(tmp_path / "home")
    home.ensure()
    home.config_path.write_text("require_trusted_repo = false\n", encoding="utf-8")

    config = home.load_config()
    assert config["require_trusted_repo"] is True

    config["require_trusted_repo"] = False

    home.save_config(config)

    assert home.load_config()["require_trusted_repo"] is True
    stored_config = tomllib.loads(home.config_path.read_text(encoding="utf-8"))
    assert "require_trusted_repo" not in stored_config


def test_agent_mode_config_defaults_and_normalizes(tmp_path):
    home = AnomxHome(tmp_path / "home")
    assert home.load_config()["agent_mode"] == AgentMode.CONFIRM.value

    home.ensure()
    home.config_path.write_text('agent_mode = "invalid"\n', encoding="utf-8")

    assert home.load_config()["agent_mode"] == AgentMode.CONFIRM.value

    home.config_path.write_text('agent_mode = "full-control"\n', encoding="utf-8")

    assert home.load_config()["agent_mode"] == AgentMode.AUTONOMOUS.value


def test_thinking_intensity_config_defaults_and_normalizes(tmp_path):
    home = AnomxHome(tmp_path / "home")
    assert home.load_config()["thinking_intensity"] == "auto"

    home.ensure()
    home.config_path.write_text('thinking_intensity = "turbo"\n', encoding="utf-8")

    assert home.load_config()["thinking_intensity"] == "auto"

    config = home.load_config()
    config["thinking_intensity"] = "high"
    home.save_config(config)

    assert home.load_config()["thinking_intensity"] == "high"


def test_debug_config_defaults_and_persists(tmp_path):
    home = AnomxHome(tmp_path / "home")

    config = home.load_config()

    assert config["debug_mode"] is False
    assert config["debug_full_session_logs"] is False
    assert config["debug_full_session_logs_path"] is None
    assert home.full_session_logs_dir(config) == tmp_path / "home"

    config["debug_mode"] = True
    config["debug_full_session_logs"] = True
    config["debug_full_session_logs_path"] = str(tmp_path / "logs")
    home.save_config(config)

    saved = home.load_config()
    assert saved["debug_mode"] is True
    assert saved["debug_full_session_logs"] is True
    assert saved["debug_full_session_logs_path"] == str(tmp_path / "logs")
    assert home.full_session_logs_enabled(saved) is True


def test_debug_crash_log_writes_only_when_debug_mode_is_active(tmp_path):
    home = AnomxHome(tmp_path / "home")

    try:
        raise RuntimeError("backend exploded")
    except RuntimeError as error:
        assert home.write_crash_log(error, context={"session": "s1"}) is None

    config = home.load_config()
    config["debug_mode"] = True
    home.save_config(config)

    try:
        raise RuntimeError("backend exploded")
    except RuntimeError as error:
        crash_log = home.write_crash_log(error, context={"session": "s1"})

    assert crash_log is not None
    assert crash_log.parent == home.crash_logs_dir
    text = crash_log.read_text(encoding="utf-8")
    assert "Anomx crash log" in text
    assert "RuntimeError: backend exploded" in text
    assert '"session": "s1"' in text


def test_discover_workspace_root_prefers_vcs_root(tmp_path):
    repo = tmp_path / "repo"
    nested = repo / "src" / "anomx" / "agent"
    nested.mkdir(parents=True)
    (repo / ".git").mkdir()
    (nested / "pyproject.toml").write_text("[project]\nname = 'nested'\n", encoding="utf-8")

    assert discover_workspace_root(nested) == repo


def test_session_storage_writes_metadata_and_events(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()

    session = home.create_session(repo, provider="openai", model="gpt-5")
    home.append_session_event(session.path, "user_message", {"message": "inspect the data"})

    events = _read_jsonl(session.path)
    assert session.path.name.startswith("rollout-")
    assert events[0]["type"] == "session_meta"
    assert events[0]["payload"]["model_provider"] == "openai"
    assert events[1]["type"] == "event_msg"
    assert events[1]["payload"]["type"] == "user_message"
    assert home.load_config()["last_session_id"] == session.session_id
    assert home.list_sessions()[0].title == "inspect the data"
    assert home.list_sessions()[0].message_count == 1
    assert home.list_sessions()[0].unread is False
    assert _read_jsonl(home.session_index_path)[0]["payload"]["session_id"] == session.session_id


def test_session_unread_flag_round_trips_without_changing_activity_time(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    home.append_session_event(session.path, "agent_message", {"message": "Done"})
    updated_at = home.list_sessions()[0].updated_at

    home.set_session_unread(session.path, True)

    unread_session = home.list_sessions()[0]
    assert unread_session.unread is True
    assert unread_session.updated_at == updated_at

    home.set_session_unread(session.path, False)

    read_session = home.list_sessions()[0]
    assert read_session.unread is False
    assert read_session.updated_at == updated_at


def test_read_session_events_skips_malformed_jsonl_rows(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()

    session = home.create_session(repo, provider="openai", model="gpt-5")
    with session.path.open("a", encoding="utf-8") as handle:
        handle.write('{"timestamp": "broken", "type": "event_msg", "payload":')
        handle.write("\n")
    home.append_session_event(session.path, "user_message", {"message": "still works"})

    events = home.read_session_events(session.path)

    assert [event.get("type") for event in events] == ["session_meta", "event_msg"]
    assert event_payload(events[-1]).get("message") == "still works"


def test_session_title_can_be_updated(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")

    home.update_session_title(session.path, "Repository Anomaly Scan")

    assert home.list_sessions()[0].title == "Repository Anomaly Scan"


def test_clear_sessions_keeps_current_session_and_resets_index(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    keep_session = home.create_session(repo, provider="openai", model="gpt-5.5")
    drop_session = home.create_session(repo, provider="openai", model="gpt-5.5")

    home.clear_sessions(keep_session_path=keep_session.path)

    sessions = home.list_sessions(limit=None)
    assert [session.session_id for session in sessions] == [keep_session.session_id]
    assert keep_session.path.exists()
    assert not drop_session.path.exists()
    assert home.load_config()["last_session_id"] == keep_session.session_id
    assert (
        _read_jsonl(home.session_index_path)[0]["payload"]["session_id"]
        == keep_session.session_id
    )


def test_delete_session_removes_one_session_and_resets_index(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    keep_session = home.create_session(repo, provider="openai", model="gpt-5.5")
    drop_session = home.create_session(repo, provider="desy", model="coding")

    assert home.delete_session(drop_session.path) is True

    sessions = home.list_sessions(limit=None)
    assert [session.session_id for session in sessions] == [keep_session.session_id]
    assert keep_session.path.exists()
    assert not drop_session.path.exists()
    assert home.load_config()["last_session_id"] == keep_session.session_id
    assert (
        _read_jsonl(home.session_index_path)[0]["payload"]["session_id"]
        == keep_session.session_id
    )


def test_session_message_count_ignores_hidden_work_context(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    home.append_session_event(session.path, "user_message", {"message": "Inspect this repo"})
    home.append_session_event(session.path, "agent_message", {"message": "Done"})
    home.append_session_event(
        session.path,
        "system_message",
        {"message": "Blocked command: reboot", "role": "forbidden"},
    )
    home.append_session_event(
        session.path,
        "system_message",
        {"message": "Question: continue?", "role": "question"},
    )
    home.append_session_event(
        session.path,
        "work_message",
        {"message": "Running tests", "role": "tool"},
    )

    assert home.list_sessions()[0].message_count == 3


def test_model_metadata_tracks_context_windows():
    assert model_context_window("gpt-5.5") == 1_000_000
    assert model_context_window("gpt-5.4-mini") == 400_000
    assert model_context_window("claude-haiku-4-5-20251001") == 200_000
    assert model_detail("claude-sonnet-4-6") == "1M context · 64K max output"


def test_provider_catalog_includes_desy_assistant():
    assert AI_PROVIDER_KEYS == ("openai", "anthropic", "desy", "ollama")


def test_thinking_intensity_options_are_model_specific():
    assert [option.value for option in thinking_intensity_options("openai", "gpt-5.5")] == [
        "auto",
        "minimal",
        "low",
        "medium",
        "high",
    ]
    assert [
        option.value for option in thinking_intensity_options("anthropic", "claude-opus-4-8")
    ] == ["auto", "low", "medium", "high", "xhigh", "max"]
    assert thinking_intensity_options("desy", "reasoning") == ()
    assert thinking_intensity_options("ollama", "qwen3.6") == ()


def test_api_key_is_written_to_owner_only_auth_file(tmp_path):
    home = AnomxHome(tmp_path / "home")

    home.set_api_key("openai", "test-key")

    assert home.has_api_key("openai")
    mode = stat.S_IMODE(home.auth_path.stat().st_mode)
    assert mode == 0o600


def test_platform_connection_is_written_to_owner_only_auth_file(tmp_path):
    home = AnomxHome(tmp_path / "home")
    home.set_platform_form_defaults(
        url="https://anomalies.msktools.desy.de",
        email="ada@example.com",
    )

    home.set_platform_connection(
        url="http://localhost:8000",
        token="platform-token",
        user_email="ada@example.com",
        organization_url="analytical-engines",
        hostname="edge-node-01",
    )

    assert home.load_config()["platform_url"] == "http://localhost:8000"
    assert home.platform_connection() == {
        "url": "http://localhost:8000",
        "token": "platform-token",
        "user_email": "ada@example.com",
        "organization_url": "analytical-engines",
        "hostname": "edge-node-01",
    }
    assert stat.S_IMODE(home.auth_path.stat().st_mode) == 0o600

    home.clear_platform_connection()

    assert home.load_config()["platform_url"] is None
    assert home.load_config()["platform_last_url"] == "https://anomalies.msktools.desy.de"
    assert home.load_config()["platform_last_email"] == "ada@example.com"
    assert home.platform_connection() is None


def test_normalize_platform_url_defaults_localhost_to_http():
    assert normalize_platform_url("localhost:8000/") == "http://localhost:8000"
    assert normalize_platform_url("platform.anomx.ai/") == "https://platform.anomx.ai"


def test_resolve_platform_api_url_falls_back_to_api_prefix(monkeypatch):
    calls: list[str] = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"allow_user_registration": True}).encode()

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        if request.full_url == "https://anomalies.msktools.desy.de/auth/registration":
            raise HTTPError(request.full_url, 404, "Not Found", hdrs=None, fp=None)
        return FakeResponse()

    monkeypatch.setattr(platform_client_module, "urlopen", fake_urlopen)

    assert (
        resolve_platform_api_url("https://anomalies.msktools.desy.de/")
        == "https://anomalies.msktools.desy.de/api"
    )
    assert calls == [
        "https://anomalies.msktools.desy.de/auth/registration",
        "https://anomalies.msktools.desy.de/api/auth/registration",
    ]


def test_connect_platform_uses_cli_agent_login_payload(monkeypatch):
    calls: list[dict[str, object]] = []

    class FakeResponse:
        def __init__(self, payload: dict[str, object]):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(self.payload).encode()

    def fake_urlopen(request, timeout):
        calls.append(
            {
                "url": request.full_url,
                "method": request.get_method(),
                "timeout": timeout,
                "payload": json.loads(request.data.decode()) if request.data else None,
            }
        )
        if request.full_url == "https://anomalies.msktools.desy.de/auth/registration":
            raise HTTPError(request.full_url, 404, "Not Found", hdrs=None, fp=None)
        if request.full_url == "https://anomalies.msktools.desy.de/api/auth/registration":
            return FakeResponse({"allow_user_registration": True})
        if request.full_url == "https://anomalies.msktools.desy.de/api/auth/login":
            return FakeResponse(
                {
                    "token": "platform-token",
                    "user": {
                        "email": "ada@example.com",
                        "organization": {"url": "analytical-engines"},
                    },
                }
            )
        return FakeResponse(
            {
                "token_type": "cli_agent",
                "client_hostname": "edge-node-01",
                "client_version": platform_client_module.__version__,
            }
        )

    monkeypatch.setattr(platform_client_module, "local_hostname", lambda: "edge-node-01")
    monkeypatch.setattr(platform_client_module, "urlopen", fake_urlopen)

    result = connect_platform(
        "https://anomalies.msktools.desy.de/",
        "ada@example.com",
        "correcthorse",
    )

    assert result.url == "https://anomalies.msktools.desy.de/api"
    assert result.token == "platform-token"
    assert result.user_email == "ada@example.com"
    assert result.organization_url == "analytical-engines"
    assert result.hostname == "edge-node-01"
    assert calls[-2]["url"] == "https://anomalies.msktools.desy.de/api/auth/login"
    assert calls[-2]["method"] == "POST"
    assert calls[-2]["timeout"] == platform_client_module.DEFAULT_TIMEOUT_SECONDS
    assert calls[-2]["payload"] == {
        "email": "ada@example.com",
        "password": "correcthorse",
        "client": "cli_agent",
        "client_hostname": "edge-node-01",
        "client_version": platform_client_module.__version__,
    }
    assert calls[-1]["url"] == "https://anomalies.msktools.desy.de/api/auth/me/agent/heartbeat"
    assert calls[-1]["method"] == "POST"
    assert calls[-1]["payload"] == {
        "client_hostname": "edge-node-01",
        "client_version": platform_client_module.__version__,
    }


def test_heartbeat_platform_connection_uses_saved_bearer_token(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    home.set_platform_connection(
        url="http://localhost:8000",
        token="platform-token",
        user_email="ada@example.com",
        organization_url="analytical-engines",
        hostname="edge-node-01",
    )
    captured: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b"{}"

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["authorization"] = request.get_header("Authorization")
        captured["payload"] = json.loads(request.data.decode())
        return FakeResponse()

    monkeypatch.setattr(platform_client_module, "local_hostname", lambda: "edge-node-02")
    monkeypatch.setattr(platform_client_module, "urlopen", fake_urlopen)

    assert heartbeat_platform_connection(home) is True
    assert captured["url"] == "http://localhost:8000/auth/me/agent/heartbeat"
    assert captured["authorization"] == "Bearer platform-token"
    assert captured["payload"] == {
        "client_hostname": "edge-node-02",
        "client_version": platform_client_module.__version__,
    }


def test_heartbeat_platform_connection_repairs_root_frontend_url(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    home.set_platform_connection(
        url="https://anomalies.msktools.desy.de",
        token="platform-token",
        user_email="ada@example.com",
        organization_url="analytical-engines",
        hostname="edge-node-01",
    )
    calls: list[str] = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b"{}"

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        if request.full_url == "https://anomalies.msktools.desy.de/auth/me/agent/heartbeat":
            raise HTTPError(request.full_url, 404, "Not Found", hdrs=None, fp=None)
        return FakeResponse()

    monkeypatch.setattr(platform_client_module, "local_hostname", lambda: "edge-node-02")
    monkeypatch.setattr(platform_client_module, "urlopen", fake_urlopen)

    assert heartbeat_platform_connection(home) is True
    assert calls == [
        "https://anomalies.msktools.desy.de/auth/me/agent/heartbeat",
        "https://anomalies.msktools.desy.de/api/auth/me/agent/heartbeat",
    ]
    assert home.platform_connection()["url"] == "https://anomalies.msktools.desy.de/api"


def test_heartbeat_platform_connection_requires_agent_heartbeat_endpoint(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    home.set_platform_connection(
        url="https://anomalies.msktools.desy.de",
        token="platform-token",
        user_email="ada@example.com",
        organization_url="analytical-engines",
        hostname="edge-node-01",
    )
    calls: list[tuple[str, str]] = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b"{}"

    def fake_urlopen(request, timeout):
        calls.append((request.get_method(), request.full_url))
        if request.full_url in {
            "https://anomalies.msktools.desy.de/auth/me/agent/heartbeat",
            "https://anomalies.msktools.desy.de/api/auth/me/agent/heartbeat",
        }:
            raise HTTPError(request.full_url, 404, "Not Found", hdrs=None, fp=None)
        return FakeResponse()

    monkeypatch.setattr(platform_client_module, "local_hostname", lambda: "edge-node-02")
    monkeypatch.setattr(platform_client_module, "urlopen", fake_urlopen)

    assert heartbeat_platform_connection(home) is False
    assert calls == [
        ("POST", "https://anomalies.msktools.desy.de/auth/me/agent/heartbeat"),
        ("POST", "https://anomalies.msktools.desy.de/api/auth/me/agent/heartbeat"),
    ]
    assert home.platform_connection()["url"] == "https://anomalies.msktools.desy.de"


def test_user_skill_storage_round_trips_global_home(tmp_path):
    home = AnomxHome(tmp_path / "home")
    skill = Skill(
        command="profile-data",
        title="Profile data",
        description="Inspect dataset shape and quality.",
        body="Profile the current dataset and report useful statistics.",
        source="user",
    )

    path = write_user_skill(home.skills_dir, skill)

    assert path == home.skills_dir / "profile-data"
    assert (path / "README.md").exists()
    assert load_user_skills(home.skills_dir) == (
        Skill(
            command="profile-data",
            title="profile-data",
            description="Inspect dataset shape and quality.",
            body="Profile the current dataset and report useful statistics.",
            source="user",
            path=path,
        ),
    )


def test_bundled_starter_skills_are_hidden_and_callable():
    skills = load_builtin_skills()

    assert {skill.command for skill in skills} >= {
        "map-folder",
        "find-issues",
        "make-report",
    }
    assert all(skill.hidden for skill in skills)


def test_anomx_api_tool_is_hidden_without_platform_connection(tmp_path):
    home = AnomxHome(tmp_path / "home")
    runtime = AgentRuntime(home, tmp_path, mode=AgentMode.CONFIRM)

    assert "use_anomx_api" not in {tool.name for tool in runtime._available_tools()}


def test_platform_connection_exposes_api_tool_and_hidden_system_skill(tmp_path):
    home = AnomxHome(tmp_path / "home")
    home.ensure()
    home.set_platform_connection(
        url="http://localhost:8000",
        token="platform-token",
        user_email="theo@example.test",
        organization_url="desy",
        hostname="agent-host",
    )
    runtime = AgentRuntime(home, tmp_path, mode=AgentMode.CONFIRM)

    assert "use_anomx_api" in {tool.name for tool in runtime._available_tools()}
    assert runtime.tool_manager.subprocess_env is not None
    assert (
        runtime.tool_manager.subprocess_env["ANOMX_PLATFORM_API_URL"]
        == "http://localhost:8000/api/v1"
    )
    assert runtime.tool_manager.subprocess_env["ANOMX_API_KEY"] == "platform-token"
    assert (home.skills_dir / "use-anomx-api" / "api.py").exists()
    assert "Connected Anomx Platform" in runtime._instructions()

    app = AnomxCliApp(home=home, cwd=tmp_path)
    assert "/use-anomx-api" not in {spec.command for spec in app._command_specs()}


def test_startup_ollama_configures_local_backend(tmp_path):
    home = AnomxHome(tmp_path / "home")

    AnomxCliApp(
        home=home,
        startup_provider="ollama",
        startup_model="qwen3.6",
    ).prepare_startup_config()

    config = home.load_config()
    assert config["onboarding_complete"] is True
    assert config["provider"] == "ollama"
    assert config["model"] == "qwen3.6"


def test_ollama_env_implies_local_provider(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL_ID", "qwen3.6")

    assert _startup_provider(None, False) == "ollama"
    assert _startup_model(None) == "qwen3.6"


def test_session_id_from_rollout_path_ignores_hyphenated_timestamp():
    path = Path("rollout-2026-06-17T15-13-40Z-abc123.jsonl")

    assert session_id_from_path(path) == "abc123"


def test_slash_commands_show_skills_on_empty_slash(tmp_path):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))

    all_commands = app._filtered_commands("/")
    model_commands = app._filtered_commands("/mo")

    assert [command.command for command in all_commands[:5]] == [
        "/new",
        "/rename",
        "/config",
        "/model",
        "/exit",
    ]
    removed_commands = {"/open", "/debug", "/skills"}
    assert removed_commands.isdisjoint({command.command for command in all_commands})
    assert {"/map-folder", "/find-issues", "/make-report"}.issubset(
        {command.command for command in all_commands}
    )
    map_folder = next(command for command in all_commands if command.command == "/map-folder")
    assert map_folder.description.startswith("Map the folder · Understand the files")
    assert [command.command for command in model_commands] == ["/model"]
    assert [command.command for command in app._filtered_commands("/ex")] == ["/exit"]
    assert [command.command for command in app._filtered_commands("/map")] == ["/map-folder"]


def test_user_skill_command_enters_slash_command_menu(tmp_path):
    home = AnomxHome(tmp_path / "home")
    write_user_skill(
        home.skills_dir,
        Skill(
            command="quality-scan",
            title="Quality scan",
            description="Inspect data quality.",
            body="Look for missing values and bad timestamps.",
            source="user",
        ),
    )
    app = AnomxCliApp(home=home)

    assert [command.command for command in app._filtered_commands("/qua")] == ["/quality-scan"]
    assert "/quality-scan" in [command.command for command in app._filtered_commands("/")]


def test_submitted_slash_command_prefers_exact_command(tmp_path):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))
    suggestions = app._filtered_commands("/")

    assert app._submitted_command("/config", suggestions, selected=0) == "/config"
    assert app._submitted_command("/open", suggestions, selected=0) == "/new"
    assert app._submitted_command("/rename Data review", suggestions, selected=0) == "/rename"
    assert app._submitted_command("/map-folder data", suggestions, selected=0) == "/map-folder"


def test_running_slash_commands_only_show_non_message_commands(tmp_path):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))

    assert [command.command for command in app._filtered_running_commands("/")] == [
        "/config",
        "/model",
    ]
    assert [command.command for command in app._filtered_running_commands("/con")] == [
        "/config"
    ]
    assert app._filtered_running_commands("/map") == []


def test_running_enter_submits_plain_message(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=repo)

    result = app._handle_running_key(
        object(),
        session,
        "\n",
        "can you do another thing?",
        25,
        "",
        0.0,
    )

    assert result.command == ""
    assert result.input_text == "can you do another thing?"
    assert result.notice == RUNNING_NOTICE
    assert result.submitted_message == "can you do another thing?"


def test_interrupt_requeue_appends_user_after_draining_old_events(tmp_path, monkeypatch):
    class Runtime:
        def __init__(self):
            self.aborted = []

        def abort_current_turn(self, session_path):
            self.aborted.append(session_path)

    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=repo, use_color=False)
    runtime = Runtime()
    events: queue.SimpleQueue[RuntimeUiEvent] = queue.SimpleQueue()
    events.put(RuntimeUiEvent("message", "Old response above the correction"))
    worker = threading.Thread(target=lambda: None)
    worker.start()
    worker.join()
    turn = ui_module.ActiveSessionTurn(
        session=session,
        runtime=runtime,
        events=events,
        result={},
        turn_id="turn-1",
        started_at=time.monotonic(),
        worker=worker,
        mode=AgentMode.CONFIRM,
    )
    replacement_turn = ui_module.ActiveSessionTurn(
        session=session,
        runtime=runtime,
        events=queue.SimpleQueue(),
        result={},
        turn_id="turn-2",
        started_at=time.monotonic(),
        worker=worker,
        mode=AgentMode.CONFIRM,
    )
    started = []

    def fake_start_session_turn(next_session):
        started.append(next_session)
        return replacement_turn

    monkeypatch.setattr(
        app,
        "_start_session_turn",
        fake_start_session_turn,
    )

    app._interrupt_and_requeue_session_turn(
        object(),
        turn,
        session,
        "Intermediate user correction",
    )

    session_events = [
        event for event in home.read_session_events(session.path) if event["type"] == "event_msg"
    ]
    assert [
        (event["payload"]["type"], event["payload"]["message"])
        for event in session_events
    ] == [
        ("agent_message", "Old response above the correction"),
        ("user_message", "Intermediate user correction"),
    ]
    assert session_events[1]["payload"]["turn_id"] == "turn-1"
    assert session_events[1]["payload"]["intermediate"] is True
    assert runtime.aborted == [session.path]
    assert started == [session]
    assert replacement_turn.turn_id == "turn-1"


def test_running_ctrl_c_clears_prompt_before_abort_confirmation(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=repo)

    result = app._handle_running_key(
        object(),
        session,
        "\x03",
        "draft text",
        10,
        "Ctrl+C",
        time.monotonic() + 3.0,
    )

    assert result.input_text == ""
    assert result.cursor == 0
    assert result.notice == RUNNING_NOTICE
    assert result.abort_key == ""
    assert result.abort_deadline == 0.0
    assert not result.exit_requested


def test_running_escape_returns_to_project_without_aborting(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=repo)

    result = app._handle_running_key(
        object(),
        session,
        "\x1b",
        "",
        0,
        "",
        0.0,
    )

    assert result.back_requested is True
    assert result.exit_requested is False


def test_running_enter_accepts_config_command(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=repo)
    suggestions = app._filtered_running_commands("/con")

    result = app._handle_running_key(
        object(),
        session,
        "\n",
        "/con",
        4,
        "",
        0.0,
        suggestions,
        0,
    )

    assert result.command == "/config"
    assert result.submitted == "/con"
    assert result.input_text == ""
    assert result.cursor == 0


def test_running_enter_blocks_skill_command(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=repo)

    result = app._handle_running_key(
        object(),
        session,
        "\n",
        "/map-folder data",
        16,
        "",
        0.0,
        app._filtered_running_commands("/map"),
        0,
    )

    assert result.command == ""
    assert result.input_text == "/map-folder data"
    assert result.notice == RUNNING_COMMAND_BLOCKED_NOTICE


def test_info_command_opens_session_info_panel(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=repo)
    opened = []

    monkeypatch.setattr(app, "_run_info_panel", lambda _stdscr, session: opened.append(session))

    assert app._handle_command(object(), "/info", session) is None
    assert opened == [session]


def test_config_manage_skills_opens_skills_panel(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=repo)
    opened = []
    selections = iter(("skills", None))

    monkeypatch.setattr(app, "_menu", lambda *_args, **_kwargs: next(selections))
    monkeypatch.setattr(app, "_run_skills_panel", lambda _stdscr, session: opened.append(session))

    app._run_config_panel(object(), session)

    assert opened == [session]


def test_rename_command_updates_current_session_title(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=repo)
    prompts = []

    def fake_prompt(_stdscr, **kwargs):
        prompts.append(kwargs)
        return "Repository Scan"

    monkeypatch.setattr(app, "_prompt_text", fake_prompt)

    renamed = app._handle_command(object(), "/rename", session)

    assert isinstance(renamed, SessionRecord)
    assert renamed.title == "Repository Scan"
    assert home.list_sessions()[0].title == "Repository Scan"
    assert prompts == [
        {
            "title": "Rename Session",
            "label": "Title",
            "default": "New session",
        }
    ]


def test_rename_command_accepts_inline_title(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=repo)

    renamed = app._handle_command(
        object(),
        "/rename",
        session,
        "/rename Repository Scan",
    )

    assert isinstance(renamed, SessionRecord)
    assert renamed.title == "Repository Scan"
    assert home.list_sessions()[0].title == "Repository Scan"


def test_skills_menu_lists_create_then_user_skills_only(tmp_path):
    home = AnomxHome(tmp_path / "home")
    write_user_skill(
        home.skills_dir,
        Skill(
            command="quality-scan",
            title="Quality scan",
            description="Inspect data quality.",
            body="Look for missing values and bad timestamps.",
            source="user",
        ),
    )
    app = AnomxCliApp(home=home)

    choices = app._skills_menu_choices()

    assert choices[0] == MenuChoice(
        "Create new Skill",
        "__create_skill__",
        "Define a global slash-command skill",
    )
    assert [(choice.label, choice.value) for choice in choices[1:]] == [
        ("/quality-scan", "quality-scan")
    ]
    assert all(
        choice.value not in {"map-folder", "find-issues", "make-report"}
        for choice in choices
    )


def test_draw_skill_editor_panel_marks_selected_field_in_accent(tmp_path):
    class Window:
        def __init__(self):
            self.writes = []

        def erase(self):
            pass

        def getmaxyx(self):
            return 28, 100

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

        def refresh(self):
            pass

    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)
    app._colors = {"light": 10, "accent": 20}
    window = Window()
    draft = SkillFormDraft(
        command="test",
        description="Testing skill edits",
        body="Do the test.",
        path=tmp_path / "home" / "skills" / "test",
    )

    app._draw_skill_editor_panel(
        window,
        "Edit Skill",
        draft,
        selected=1,
    )

    assert any(text == "Command" and attr == 10 for _, _, text, attr in window.writes)
    assert any(text == "/test" and attr == 10 for _, _, text, attr in window.writes)
    assert any(text == "Description" and attr == 20 for _, _, text, attr in window.writes)
    assert any(text == "Testing skill edits" and attr == 0 for _, _, text, attr in window.writes)
    assert any(text.startswith("Stored at:") and attr == 10 for _, _, text, attr in window.writes)
    assert any(text == "Skill" and attr == 10 for _, _, text, attr in window.writes)


def test_draw_skill_detail_panel_keeps_skill_label_light(tmp_path):
    class Window:
        def __init__(self):
            self.writes = []

        def erase(self):
            pass

        def getmaxyx(self):
            return 28, 100

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

        def refresh(self):
            pass

    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)
    app._colors = {"light": 10, "accent": 20}
    window = Window()
    skill = Skill(
        command="test",
        title="test",
        description="Testing skill detail.",
        body="Do the test.",
        source="user",
        path=tmp_path / "home" / "skills" / "test",
    )

    app._draw_skill_detail_panel(window, skill)

    assert any(text == "Skill" and attr == 10 for _, _, text, attr in window.writes)
    assert not any(text == "Skill" and attr == 20 for _, _, text, attr in window.writes)


def test_run_skill_editor_saves_create_form(tmp_path):
    class Window:
        def __init__(self):
            self._keys = iter(
                (
                    *"test",
                    curses.KEY_DOWN,
                    *"Test description",
                    curses.KEY_DOWN,
                    *"Do the test.",
                    "\x13",
                )
            )
            self.writes = []

        def erase(self):
            pass

        def getmaxyx(self):
            return 28, 100

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

        def refresh(self):
            pass

        def get_wch(self):
            return next(self._keys)

    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)
    window = Window()

    saved = app._run_skill_editor(window, title="Create Skill")

    assert saved is not None
    assert saved.command == "test"
    assert saved.title == "test"
    assert saved.description == "Test description"
    assert saved.body == "Do the test."
    assert (tmp_path / "home" / "skills" / "test" / "README.md").exists()
    assert any(
        text == "Esc Cancel · Ctrl+S Save · ↑↓ Navigate · Enter Next"
        for _, _, text, _ in window.writes
    )


def test_run_skill_editor_saves_existing_skill_and_renames_file(tmp_path):
    class Window:
        def __init__(self):
            self._keys = iter(
                (
                    *"renamed",
                    "\x13",
                )
            )
            self.writes = []

        def erase(self):
            pass

        def getmaxyx(self):
            return 28, 100

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

        def refresh(self):
            pass

        def get_wch(self):
            return next(self._keys)

    home = AnomxHome(tmp_path / "home")
    old_path = write_user_skill(
        home.skills_dir,
        Skill(
            command="test",
            title="Test skills",
            description="Testing skill edits",
            body="Do the test.",
            source="user",
        ),
    )
    existing = load_user_skills(home.skills_dir)[0]
    app = AnomxCliApp(home=home, use_color=False)

    saved = app._run_skill_editor(Window(), title="Edit Skill", existing_skill=existing)

    assert saved is not None
    assert saved.command == "testrenamed"
    assert saved.path == home.skills_dir / "testrenamed"
    assert not old_path.exists()
    assert (saved.path / "README.md").exists()


def test_prompt_multiline_text_saves_with_ctrl_s(tmp_path, monkeypatch):
    class Window:
        def __init__(self):
            self._keys = iter((*"Remember this", "\x13"))

        def getmaxyx(self):
            return 28, 100

        def get_wch(self):
            return next(self._keys)

    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)
    footers: list[str] = []

    def draw_overlay(_stdscr, **kwargs):
        footers.append(str(kwargs.get("footer") or ""))

    monkeypatch.setattr(app, "_draw_overlay", draw_overlay)

    result = app._prompt_multiline_text(
        Window(),
        "Create Memory",
        "Memory content. Ctrl+S saves.",
        optional=False,
    )

    assert result == "Remember this"
    assert "Ctrl+S Save" in footers[-1]


def test_skill_detail_panel_edit_shortcut_updates_current_skill(tmp_path, monkeypatch):
    class Window:
        def __init__(self):
            self._keys = iter(("\x05", "\n"))
            self.writes = []

        def erase(self):
            pass

        def getmaxyx(self):
            return 28, 100

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

        def refresh(self):
            pass

        def get_wch(self):
            return next(self._keys)

    home = AnomxHome(tmp_path / "home")
    path = write_user_skill(
        home.skills_dir,
        Skill(
            command="test",
            title="Test skills",
            description="Testing skill edits",
            body="Do the test.",
            source="user",
        ),
    )
    skill = Skill(
        command="test",
        title="Test skills",
        description="Testing skill edits",
        body="Do the test.",
        source="user",
        path=path,
    )
    edited = Skill(
        command="test",
        title="test",
        description="Edited skill description",
        body="Do the test.",
        source="user",
        path=path,
    )
    app = AnomxCliApp(home=home, use_color=False)
    monkeypatch.setattr(app, "_edit_user_skill", lambda _stdscr, _skill: edited)
    window = Window()

    app._run_skill_detail_panel(window, skill)

    assert any(text == "Edited skill description" for _, _, text, _ in window.writes)


def test_delete_user_skill_removes_file_when_confirmed(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    path = write_user_skill(
        home.skills_dir,
        Skill(
            command="test",
            title="Test skills",
            description="Testing skill edits",
            body="Do the test.",
            source="user",
        ),
    )
    skill = Skill(
        command="test",
        title="Test skills",
        description="Testing skill edits",
        body="Do the test.",
        source="user",
        path=path,
    )
    app = AnomxCliApp(home=home, use_color=False)
    monkeypatch.setattr(app, "_menu", lambda *_args, **_kwargs: "delete")
    monkeypatch.setattr(app, "_message", lambda *_args, **_kwargs: None)

    assert app._delete_user_skill(object(), skill) is True
    assert not path.exists()


def test_skill_invocation_records_prompt_payload(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=repo)
    skill = app._skill_for_command("/map-folder")
    assert skill is not None

    monkeypatch.setattr(app, "_maybe_start_session_rename", lambda _session: None)
    monkeypatch.setattr(app, "_latest_user_anchor_line", lambda _stdscr, _session: None)
    monkeypatch.setattr(app, "_animate_message_anchor", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        app,
        "_run_backend_turn",
        lambda *_args, **_kwargs: BackendTurnResult("", 0),
    )

    assert app._invoke_skill(object(), session, skill, "/map-folder data/raw") is None

    payload = _read_jsonl(session.path)[-1]["payload"]
    assert payload["type"] == "skill_invocation"
    assert payload["message"] == "/map-folder data/raw"
    assert "Use the Anomx skill /map-folder" in payload["prompt"]
    assert "User arguments:\n\ndata/raw" in payload["prompt"]
    assert AgentRuntime(home, repo).conversation_messages(session.path)[-1] == {
        "role": "user",
        "content": payload["prompt"],
    }
    assert app._read_message_lines(session.path) == [MessageLine("user", "/map-folder data/raw")]


def test_config_menu_shows_only_requested_entries(tmp_path):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))

    choices = app._config_menu_choices()

    assert [(choice.label, choice.value, choice.detail) for choice in choices] == [
        ("Choose Backend", "backend", "Select provider and enter API key"),
        ("Choose Model", "model", "Pick the model for the selected backend"),
        (
            "Connect Platform",
            "platform",
            "Send agent activity, results, and findings to Anomx Platform",
        ),
        ("Manage Debug Mode", "debug", "debug mode false"),
        ("Manage Skills", "skills", "Create or open user slash-command skills"),
        ("Manage Memories", "memories", "Create, view, or remove local agent memories"),
        (
            "Manage Instructions",
            "manage_instructions",
            "Add, edit, view, or remove custom agent instructions",
        ),
        ("Manage Sandbox", "sandbox", "sandbox disabled"),
        ("Manage Commands", "commands", "Review globally approved and rejected commands"),
    ]


def test_config_menu_shows_manage_platform_when_connected(tmp_path):
    home = AnomxHome(tmp_path / "home")
    home.set_platform_connection(
        url="https://anomalies.msktools.desy.de/api",
        token="platform-token",
        user_email="ada@example.com",
    )
    app = AnomxCliApp(home=home)

    choices = app._config_menu_choices()

    assert choices[2].label == "Manage Platform"
    assert choices[2].value == "platform"
    assert choices[2].detail == "Connected to anomalies.msktools.desy.de"


def test_debug_menu_choices_show_current_values(tmp_path):
    home = AnomxHome(tmp_path / "home")
    config = home.load_config()
    config["debug_mode"] = True
    config["debug_full_session_logs"] = True
    config["debug_full_session_logs_path"] = str(tmp_path / "logs")
    home.save_config(config)
    app = AnomxCliApp(home=home)

    choices = app._debug_menu_choices(home.load_config())

    assert [(choice.label, choice.value, choice.detail) for choice in choices] == [
        ("Debug mode active", "debug_mode", "true"),
        ("Full session logs", "full_session_logs", "true"),
        ("Debug location", "full_session_logs_path", str(tmp_path / "logs")),
    ]


def test_run_debug_panel_toggles_values_and_sets_log_path(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home)
    selections = iter(
        (
            "debug_mode",
            "full_session_logs",
            "full_session_logs_path",
            None,
        )
    )

    monkeypatch.setattr(app, "_menu", lambda *_args, **_kwargs: next(selections))
    monkeypatch.setattr(
        app,
        "_prompt_text",
        lambda *_args, **_kwargs: str(tmp_path / "request-logs"),
    )

    app._run_debug_panel(object(), session)

    config = home.load_config()
    assert config["debug_mode"] is True
    assert config["debug_full_session_logs"] is True
    assert config["debug_full_session_logs_path"] == str(tmp_path / "request-logs")


def test_untrusted_workspace_requires_access_check_when_config_disables_it(
    tmp_path, monkeypatch
):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    config = home.load_config()
    config["onboarding_complete"] = True
    config["require_trusted_repo"] = False
    home.save_config(config)
    app = AnomxCliApp(home=home, cwd=repo)
    access_checks: list[Path] = []

    monkeypatch.setattr(app, "_configure_terminal", lambda _stdscr: None)
    monkeypatch.setattr(app, "_run_startup_loading", lambda _stdscr: False)

    def fail_access_check(_stdscr):
        access_checks.append(app.workspace_root)
        return False

    monkeypatch.setattr(app, "_run_access_check", fail_access_check)

    assert app._run(object()) == 1
    assert access_checks == [repo.resolve()]


def test_latest_continuable_session_uses_same_workspace_root(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    nested = repo / "src"
    nested.mkdir(parents=True)
    (repo / ".git").mkdir()
    other = tmp_path / "other"
    other.mkdir()

    home.create_session(repo, provider="openai", model="gpt-5.5")
    other_session = home.create_session(other, provider="openai", model="gpt-5.5")
    home.append_session_event(
        other_session.path,
        "user_message",
        {"message": "Inspect another workspace"},
    )
    matching_session = home.create_session(nested, provider="openai", model="gpt-5.5")
    home.append_session_event(
        matching_session.path,
        "user_message",
        {"message": "Identify important data channels"},
    )
    app = AnomxCliApp(home=home, cwd=repo)

    assert app._latest_continuable_session().session_id == matching_session.session_id


def test_latest_continuable_session_includes_empty_latest_session(tmp_path, monkeypatch):
    timestamps = iter(
        (
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:00:01Z",
            "2026-01-01T00:00:02Z",
        )
    )
    monkeypatch.setattr("anomx.agent.store.utc_now_iso", lambda: next(timestamps))
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    older_session = home.create_session(repo, provider="openai", model="gpt-5.5")
    home.append_session_event(
        older_session.path,
        "user_message",
        {"message": "Identify important data channels"},
    )
    latest_session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=repo)

    assert latest_session.message_count == 0
    assert app._latest_continuable_session().session_id == latest_session.session_id


def test_startup_launches_project_page_for_workspace(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    home.trust_repo(repo)
    project = home.save_project(repo, "Repository Signals")
    config = home.load_config()
    config["onboarding_complete"] = True
    home.save_config(config)
    previous_session = home.create_session(repo, provider="openai", model="gpt-5.5")
    home.append_session_event(
        previous_session.path,
        "user_message",
        {"message": "Identify important data channels"},
    )
    home.append_session_event(
        previous_session.path,
        "worker_event",
        {
            "worker_id": "stale-worker",
            "name": "Engineer",
            "status": "working",
            "statement": "Thinking",
        },
    )
    app = AnomxCliApp(home=home, cwd=repo)
    opened_projects = []

    monkeypatch.setattr(app, "_configure_terminal", lambda _stdscr: None)
    monkeypatch.setattr(app, "_run_startup_loading", lambda _stdscr: False)
    monkeypatch.setattr(
        app,
        "_run_project",
        lambda _stdscr, current_project: opened_projects.append(current_project) or 0,
    )
    monkeypatch.setattr(
        app,
        "_run_continue_session_prompt",
        lambda *_args: (_ for _ in ()).throw(AssertionError("should not prompt")),
    )

    assert app._run(object()) == 0
    assert opened_projects == [project]
    assert worker_snapshots(home.read_session_events(previous_session.path)) == ()


def test_startup_project_uses_prepared_project(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    project = home.save_project(repo, "Prepared Project")
    app = AnomxCliApp(home=home, cwd=repo)

    monkeypatch.setattr(
        app,
        "_ensure_project_with_loading",
        lambda _stdscr: (_ for _ in ()).throw(AssertionError("should be prepared")),
    )

    opened = app._startup_project(
        object(),
        StartupPreparation(project=project),
    )

    assert opened == project


def test_startup_creates_project_name_when_location_is_new(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    home.trust_repo(repo)
    config = home.load_config()
    config["onboarding_complete"] = True
    home.save_config(config)
    app = AnomxCliApp(home=home, cwd=repo)
    opened_projects = []

    monkeypatch.setattr(app, "_configure_terminal", lambda _stdscr: None)
    monkeypatch.setattr(app, "_run_startup_loading", lambda _stdscr: False)
    monkeypatch.setattr(app.runtime, "suggest_project_name", lambda *_args: "Signal Lab")
    monkeypatch.setattr(app, "_ensure_project_with_loading", lambda _stdscr: app._ensure_project())
    monkeypatch.setattr(
        app,
        "_run_project",
        lambda _stdscr, project: opened_projects.append(project) or 0,
    )

    assert app._run(object()) == 0
    assert [project.name for project in opened_projects] == ["Signal Lab"]
    assert home.project_for_path(repo).name == "Signal Lab"


def test_startup_loading_renders_matrix_wall_and_brand(tmp_path):
    class Window:
        def __init__(self):
            self.writes = []

        def erase(self):
            pass

        def getmaxyx(self):
            return 12, 48

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

        def refresh(self):
            pass

    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)
    app._colors = {
        "matrix_dim": 10,
        "matrix_function": 50,
        "matrix_brand": 50,
    }
    window = Window()

    app._draw_startup_loading(window, frame=3)

    assert window.writes[0][0:2] == (0, 0)
    assert len(window.writes[0][2]) == 48
    assert window.writes[0][2].isalnum()
    assert window.writes[0][3] == 10
    assert not any(text == " ANOMX " for _, _, text, _ in window.writes)
    brand_character_count = sum(
        1
        for y, _, text, attr in window.writes
        if y < 10 and len(text) == 1 and attr == 50
    )
    assert brand_character_count > 20


def test_startup_loading_reveals_matrix_before_overlays(tmp_path):
    class Window:
        def __init__(self):
            self.writes = []

        def erase(self):
            pass

        def getmaxyx(self):
            return 10, 24

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

        def refresh(self):
            pass

    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)
    app._colors = {
        "background": 0,
        "matrix_dim": 10,
        "matrix_function": 50,
        "matrix_brand": 50,
    }
    window = Window()

    app._draw_startup_loading(
        window,
        frame=1,
        elapsed=ui_module.STARTUP_REVEAL_SECONDS / 2,
    )

    rows = {
        y: text
        for y, x, text, attr in window.writes
        if x == 0 and len(text) == 24 and attr in {0, 10}
    }
    column_heights = []
    for x in range(0, 24, ui_module.STARTUP_COLUMN_WIDTH):
        column_heights.append(
            sum(1 for y in range(10) if rows.get(y, "")[x : x + 2].strip())
        )

    assert min(column_heights) < max(column_heights)
    assert min(column_heights) < 10
    assert max(column_heights) > 0
    assert not any(attr == 50 for _, _, _, attr in window.writes)


def test_configure_terminal_uses_terminal_default_background(tmp_path, monkeypatch):
    class Window:
        def __init__(self):
            self.background = None

        def keypad(self, _enabled):
            pass

        def bkgd(self, ch, attr):
            self.background = (ch, attr)

    init_pairs = []
    monkeypatch.setattr(curses, "has_colors", lambda: True)
    monkeypatch.setattr(curses, "start_color", lambda: None)
    monkeypatch.setattr(curses, "use_default_colors", lambda: None)
    monkeypatch.setattr(curses, "init_pair", lambda pair, fg, bg: init_pairs.append((pair, fg, bg)))
    monkeypatch.setattr(curses, "color_pair", lambda pair: pair << 8)
    monkeypatch.setattr(curses, "COLORS", 256, raising=False)
    monkeypatch.setattr(curses, "COLOR_PAIRS", 16, raising=False)
    for name in (
        "noecho",
        "raw",
        "set_escdelay",
        "curs_set",
        "mousemask",
        "mouseinterval",
    ):
        monkeypatch.setattr(curses, name, lambda *args: None)

    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=True)
    app._enable_bracketed_paste = lambda: None
    window = Window()

    app._configure_terminal(window)

    assert (1, curses.COLOR_CYAN, -1) in init_pairs
    assert (6, -1, -1) in init_pairs
    assert (7, -1, -1) in init_pairs
    assert (8, 208, -1) in init_pairs
    assert window.background == (" ", 7 << 8)
    assert app._colors["selected"] & curses.A_REVERSE
    assert app._colors["user"] & curses.A_REVERSE


def test_terminal_default_colors_fall_back_when_defaults_are_unavailable(tmp_path, monkeypatch):
    def raise_curses_error():
        raise curses.error

    monkeypatch.setattr(curses, "use_default_colors", raise_curses_error)
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=True)

    assert app._terminal_default_colors() == (curses.COLOR_WHITE, curses.COLOR_BLACK)


def test_startup_loading_reveals_function_from_left_to_right(tmp_path):
    class Window:
        def __init__(self):
            self.writes = []

        def erase(self):
            pass

        def getmaxyx(self):
            return 14, 80

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

        def refresh(self):
            pass

    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)
    app._colors = {
        "background": 0,
        "matrix_dim": 10,
        "matrix_function": 50,
        "matrix_brand": 60,
    }
    window = Window()

    app._draw_startup_loading(
        window,
        frame=8,
        visible_rows=14,
        show_overlays=True,
        line_progress=0.35,
        brand_progress=0.0,
    )

    function_writes = [(y, x) for y, x, text, attr in window.writes if attr == 50 and text]

    assert function_writes
    assert min(x for _, x in function_writes) == 0
    assert max(x for _, x in function_writes) <= int((80 - 1) * 0.35)


def test_startup_loading_reveals_brand_pixel_by_pixel(tmp_path):
    class Window:
        def __init__(self):
            self.writes = []

        def erase(self):
            pass

        def getmaxyx(self):
            return 16, 100

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

        def refresh(self):
            pass

    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)
    app._colors = {
        "background": 0,
        "matrix_dim": 10,
        "matrix_function": 50,
        "matrix_brand": 60,
    }
    partial_window = Window()
    full_window = Window()

    app._draw_startup_loading(
        partial_window,
        frame=11,
        visible_rows=16,
        show_overlays=True,
        line_progress=0.0,
        brand_progress=0.35,
    )
    app._draw_startup_loading(
        full_window,
        frame=11,
        visible_rows=16,
        show_overlays=True,
        line_progress=0.0,
        brand_progress=1.0,
    )

    partial_brand_count = sum(1 for *_, attr in partial_window.writes if attr == 60)
    full_brand_count = sum(1 for *_, attr in full_window.writes if attr == 60)

    assert 0 < partial_brand_count < full_brand_count


def test_startup_loading_randomly_removes_matrix_bits(tmp_path):
    class Window:
        def __init__(self):
            self.writes = []

        def erase(self):
            pass

        def getmaxyx(self):
            return 8, 32

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

        def refresh(self):
            pass

    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)
    app._colors = {
        "background": 0,
        "matrix_dim": 10,
        "matrix_function": 50,
        "matrix_brand": 50,
    }
    window = Window()

    app._draw_startup_loading(
        window,
        frame=5,
        visible_rows=8,
        removal_progress=0.5,
        show_overlays=False,
    )

    rows = {
        y: text
        for y, x, text, attr in window.writes
        if x == 0 and len(text) == 32 and attr in {0, 10}
    }
    mixed_rows = [
        text
        for text in rows.values()
        if text.strip() and any(character == " " for character in text)
    ]
    assert len(mixed_rows) >= 4
    assert rows[0].strip()
    assert rows[7].strip()
    assert not any(attr == 50 for _, _, _, attr in window.writes)


def test_startup_loading_randomly_dissolves_overlays(tmp_path):
    class Window:
        def __init__(self):
            self.writes = []

        def erase(self):
            pass

        def getmaxyx(self):
            return 14, 80

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

        def refresh(self):
            pass

    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)
    app._colors = {
        "background": 0,
        "matrix_dim": 10,
        "matrix_function": 50,
        "matrix_brand": 60,
    }
    full_window = Window()
    dissolving_window = Window()

    app._draw_startup_loading(
        full_window,
        frame=13,
        visible_rows=14,
        show_overlays=True,
        line_progress=1.0,
        brand_progress=1.0,
    )
    app._draw_startup_loading(
        dissolving_window,
        frame=13,
        visible_rows=14,
        removal_progress=0.5,
        show_overlays=True,
        line_progress=1.0,
        brand_progress=1.0,
    )

    full_overlay_count = sum(1 for *_, attr in full_window.writes if attr in {50, 60})
    dissolving_overlay_count = sum(
        1 for *_, attr in dissolving_window.writes if attr in {50, 60}
    )
    rows = {
        y: text
        for y, x, text, attr in dissolving_window.writes
        if x == 0 and len(text) == 80 and attr in {0, 10}
    }

    assert 0 < dissolving_overlay_count < full_overlay_count
    assert sum(any(character == " " for character in text) for text in rows.values()) >= 10
    remaining_by_row = [
        sum(character != " " for character in rows[y])
        for y in range(14)
    ]
    remaining_by_column = [
        sum(rows[y][x] != " " for y in range(14))
        for x in range(80)
    ]
    assert min(remaining_by_row) > 10
    assert max(remaining_by_row) < 70
    assert min(remaining_by_column) > 0
    assert max(remaining_by_column) < 14


def test_startup_loading_activity_phases_are_stable(tmp_path):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)

    assert app._startup_loading_activity(0.0) == "Booting"
    assert app._startup_loading_activity(1.99) == "Booting"
    assert app._startup_loading_activity(2.0) == "Connecting"
    assert app._startup_loading_activity(3.99) == "Connecting"
    assert app._startup_loading_activity(4.0) == "Screening"


def test_startup_loading_draws_activity_text(tmp_path):
    class Window:
        def __init__(self):
            self.writes = []

        def erase(self):
            pass

        def getmaxyx(self):
            return 10, 40

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

        def refresh(self):
            pass

    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)
    window = Window()

    app._draw_startup_loading(window, frame=0, elapsed=0.0, activity_text="Booting")

    assert any(
        text == "Booting" and attr == app._attr("accent")
        for _, _, text, attr in window.writes
    )


def test_startup_loading_exits_after_deadline_without_platform(tmp_path, monkeypatch):
    class Window:
        def __init__(self):
            self.nodelay_calls = []
            self.writes = []

        def erase(self):
            pass

        def getmaxyx(self):
            return 8, 32

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

        def refresh(self):
            pass

        def nodelay(self, flag):
            self.nodelay_calls.append(flag)

        def get_wch(self):
            raise curses.error

    clock = {"now": 0.0}
    sleeps = []

    def fake_sleep(seconds):
        sleeps.append(seconds)
        clock["now"] += 2.0

    monkeypatch.setattr(ui_module.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(ui_module.time, "sleep", fake_sleep)
    monkeypatch.setattr(
        ui_module,
        "heartbeat_platform_connection",
        lambda _home: (_ for _ in ()).throw(AssertionError("unexpected heartbeat")),
    )
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)
    app._colors = {
        "matrix_dim": 10,
        "matrix_function": 50,
        "matrix_brand": 50,
    }
    window = Window()

    assert app._run_startup_loading(window) is False
    assert window.nodelay_calls == [True, False]
    assert clock["now"] >= ui_module.STARTUP_LOADING_SECONDS
    assert len(sleeps) >= 3
    assert all(seconds == ui_module.STARTUP_FRAME_SECONDS for seconds in sleeps)


def test_startup_loading_exits_when_platform_heartbeat_succeeds(
    tmp_path,
    monkeypatch,
):
    class Window:
        def __init__(self):
            self.nodelay_calls = []
            self.writes = []

        def erase(self):
            pass

        def getmaxyx(self):
            return 8, 48

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

        def refresh(self):
            pass

        def nodelay(self, flag):
            self.nodelay_calls.append(flag)

        def get_wch(self):
            raise curses.error

    home = AnomxHome(tmp_path / "home")
    home.set_platform_connection(
        url="https://anomalies.msktools.desy.de/api",
        token="platform-token",
        user_email="ada@example.com",
    )
    app = AnomxCliApp(home=home, use_color=False)
    app._colors = {
        "matrix_dim": 10,
        "matrix_function": 50,
        "matrix_brand": 50,
    }
    real_sleep = time.sleep
    clock = {"now": 0.0}
    sleeps = []

    def fake_sleep(seconds):
        sleeps.append(seconds)
        clock["now"] += 2.0
        real_sleep(0)

    monkeypatch.setattr(ui_module.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(ui_module.time, "sleep", fake_sleep)
    monkeypatch.setattr(ui_module, "heartbeat_platform_connection", lambda _home: True)
    window = Window()

    assert app._run_startup_loading(window) is True
    assert window.nodelay_calls == [True, False]
    assert clock["now"] >= ui_module.STARTUP_LOADING_SECONDS
    assert len(sleeps) >= 3


def test_configure_backend_requires_api_key_for_hosted_provider(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    app = AnomxCliApp(home=home)
    prompts: list[tuple[str, str]] = []

    monkeypatch.setattr(app, "_select_provider", lambda _stdscr: provider_by_key("openai"))

    def fake_prompt(_stdscr, title, label, mask=False, optional=True):
        prompts.append((title, label))
        assert mask is True
        assert optional is False
        return "test-api-key"

    monkeypatch.setattr(app, "_prompt_text", fake_prompt)

    assert app._configure_backend(object()) is True

    config = home.load_config()
    assert config["provider"] == "openai"
    assert home.has_api_key("openai") is True
    assert prompts == [("OpenAI", "API key")]


def test_configure_backend_provider_change_requires_model_selection(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    app = AnomxCliApp(home=home)
    prompts: list[tuple[str, str]] = []
    selected_models: list[str] = []

    monkeypatch.setattr(app, "_select_provider", lambda _stdscr: provider_by_key("desy"))
    monkeypatch.setattr(
        app,
        "_select_model",
        lambda _stdscr, provider: selected_models.append(provider.key) or "reasoning",
    )

    def fake_prompt(_stdscr, title, label, mask=False, optional=True):
        prompts.append((title, label))
        assert mask is True
        assert optional is False
        return "desy-api-key"

    monkeypatch.setattr(app, "_prompt_text", fake_prompt)

    assert app._configure_backend(object()) is True

    config = home.load_config()
    assert config["provider"] == "desy"
    assert config["model"] == "reasoning"
    assert home.has_api_key("desy") is True
    assert prompts == [("DESY Assistant", "API key")]
    assert selected_models == ["desy"]


def test_configure_backend_can_keep_existing_api_key(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    home.set_api_key("desy", "saved-desy-key")
    app = AnomxCliApp(home=home)
    prompts: list[tuple[str, str]] = []
    selected_models: list[str] = []

    monkeypatch.setattr(app, "_select_provider", lambda _stdscr: provider_by_key("desy"))
    monkeypatch.setattr(
        app,
        "_menu",
        lambda _stdscr, title, subtitle, choices: (
            prompts.append((title, subtitle)),
            "keep",
        )[1],
    )
    monkeypatch.setattr(
        app,
        "_select_model",
        lambda _stdscr, provider: selected_models.append(provider.key) or "coding",
    )
    monkeypatch.setattr(
        app,
        "_prompt_text",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("prompt should not run")),
    )

    assert app._configure_backend(object()) is True

    config = home.load_config()
    assert config["provider"] == "desy"
    assert config["model"] == "coding"
    assert prompts == [("DESY Assistant", "API key already configured")]
    assert selected_models == ["desy"]


def test_configure_backend_does_not_save_provider_change_when_model_selection_is_cancelled(
    tmp_path, monkeypatch
):
    home = AnomxHome(tmp_path / "home")
    app = AnomxCliApp(home=home)

    monkeypatch.setattr(app, "_select_provider", lambda _stdscr: provider_by_key("desy"))
    monkeypatch.setattr(app, "_prompt_text", lambda *_args, **_kwargs: "desy-api-key")
    monkeypatch.setattr(app, "_select_model", lambda _stdscr, provider: None)

    assert app._configure_backend(object()) is False

    config = home.load_config()
    assert config["provider"] == "openai"
    assert config["model"] == "gpt-5.5"


def test_run_session_executes_selected_slash_command(tmp_path, monkeypatch):
    class Window:
        def __init__(self):
            self._keys = iter(("/", curses.KEY_DOWN, "\n"))

        def get_wch(self):
            return next(self._keys)

    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, use_color=False)
    executed: list[str] = []

    monkeypatch.setattr(app, "_draw_session", lambda *args, **kwargs: None)

    def record_command(_stdscr, command, _current_session, _submitted=""):
        executed.append(command)
        return "exit"

    monkeypatch.setattr(app, "_handle_command", record_command)

    assert app._run_session(Window(), session) == 0
    assert executed == ["/rename"]


def test_run_config_panel_closes_after_backend_configuration(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, use_color=False)
    stdscr = object()
    menu_calls = 0
    configured: list[object] = []

    def fake_menu(*_args, **_kwargs):
        nonlocal menu_calls
        menu_calls += 1
        return "backend"

    monkeypatch.setattr(app, "_menu", fake_menu)
    monkeypatch.setattr(
        app,
        "_configure_backend",
        lambda stdscr: configured.append(stdscr) or True,
    )

    app._run_config_panel(stdscr, session)

    assert menu_calls == 1
    assert configured == [stdscr]
    assert app.state == AgentState.NEW_SESSION


def test_run_config_panel_closes_after_model_selection(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, use_color=False)
    stdscr = object()
    menu_calls = 0
    selected: list[tuple[object, object]] = []

    def fake_menu(*_args, **_kwargs):
        nonlocal menu_calls
        menu_calls += 1
        return "model"

    monkeypatch.setattr(app, "_menu", fake_menu)
    monkeypatch.setattr(
        app,
        "_run_model_panel",
        lambda stdscr, current_session, **_kwargs: (
            selected.append((stdscr, current_session)) or True
        ),
    )

    app._run_config_panel(stdscr, session)

    assert menu_calls == 1
    assert selected == [(stdscr, session)]
    assert app.state == AgentState.NEW_SESSION


def test_run_model_panel_saves_thinking_intensity_after_model_selection(
    tmp_path,
    monkeypatch,
):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, use_color=False)
    intensity_prompts: list[tuple[str, str]] = []

    monkeypatch.setattr(app, "_bottom_menu", lambda *_args, **_kwargs: "gpt-5.4")

    def fake_intensity(_stdscr, provider, model):
        intensity_prompts.append((provider.key, model))
        return "high"

    monkeypatch.setattr(app, "_select_thinking_intensity", fake_intensity)

    assert app._run_model_panel(object(), session) is True

    config = home.load_config()
    assert config["provider"] == "openai"
    assert config["model"] == "gpt-5.4"
    assert config["thinking_intensity"] == "high"
    assert intensity_prompts == [("openai", "gpt-5.4")]


def test_open_session_choices_show_location_without_model(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    current_session = home.create_session(repo, provider="openai", model="gpt-5.5")
    home.append_session_event(current_session.path, "user_message", {"message": "Inspect"})
    home.append_session_event(current_session.path, "agent_message", {"message": "Done"})
    app = AnomxCliApp(home=home, use_color=False)

    choices = app._open_session_choices([home.list_sessions(limit=None)[0]], selected=0)

    assert choices[0].detail == (
        f"2 messages · {current_session.created_at} · {current_session.cwd}"
    )
    assert "openai/gpt-5.5" not in choices[0].detail
    assert "ctrl+d Delete" not in choices[0].detail
    assert "Enter to confirm" not in choices[0].detail


def test_open_session_subtitle_shows_selected_action(tmp_path):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)

    assert app._open_session_subtitle() == "Choose a stored session · ctrl+d Delete"
    assert app._open_session_subtitle(delete_pending_index=0) == (
        "Choose a stored session · Enter to confirm"
    )


def test_open_session_panel_deletes_selected_session_after_confirmation(
    tmp_path,
    monkeypatch,
):
    class Window:
        def __init__(self):
            self._keys = iter(("\x04", "\n", "\x1b"))

        def get_wch(self):
            return next(self._keys)

        def nodelay(self, _flag):
            pass

    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    current_session = home.create_session(repo, provider="openai", model="gpt-5.5")
    drop_session = home.create_session(repo, provider="desy", model="coding")
    app = AnomxCliApp(home=home, use_color=False)
    panels = []

    def capture_draw(*_args, **kwargs):
        panels.append(kwargs["bottom_panel"])
        return None

    monkeypatch.setattr(app, "_draw_session", capture_draw)

    assert app._open_session_panel(Window(), current_session) is None
    assert not drop_session.path.exists()
    assert current_session.path.exists()
    assert [session.session_id for session in home.list_sessions(limit=None)] == [
        current_session.session_id
    ]
    assert panels[0].subtitle.endswith(" · ctrl+d Delete")
    assert panels[1].subtitle.endswith(" · Enter to confirm")
    assert not panels[0].choices[0].detail.endswith(" · ctrl+d Delete")
    assert not panels[1].choices[0].detail.endswith(" · Enter to confirm")


def test_open_session_panel_escape_cancels_delete_confirmation(tmp_path, monkeypatch):
    class Window:
        def __init__(self):
            self._keys = iter(("\x04", "\x1b", "\x1b"))

        def get_wch(self):
            return next(self._keys)

        def nodelay(self, _flag):
            pass

    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    current_session = home.create_session(repo, provider="openai", model="gpt-5.5")
    target_session = home.create_session(repo, provider="desy", model="coding")
    app = AnomxCliApp(home=home, use_color=False)
    panels = []

    def capture_draw(*_args, **kwargs):
        panels.append(kwargs["bottom_panel"])
        return None

    monkeypatch.setattr(app, "_draw_session", capture_draw)

    assert app._open_session_panel(Window(), current_session) is None
    assert target_session.path.exists()
    assert panels[0].subtitle.endswith(" · ctrl+d Delete")
    assert panels[1].subtitle.endswith(" · Enter to confirm")
    assert panels[2].subtitle.endswith(" · ctrl+d Delete")
    assert not panels[0].choices[0].detail.endswith(" · ctrl+d Delete")
    assert not panels[1].choices[0].detail.endswith(" · Enter to confirm")


def test_prompt_lines_wrap_and_keep_cursor_visible(tmp_path):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))

    lines = app._prompt_lines("abcdefghijklmnopqrstuvwxyz", width=10)

    assert lines == ["abcdefghij", "klmnopqrst", "uvwxyz"]


def test_prompt_layout_starts_one_line_and_grows_with_input(tmp_path):
    class Window:
        def getmaxyx(self):
            return 40, 28

    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))

    empty = app._prompt_layout(Window(), "")
    short = app._prompt_layout(Window(), "short")
    wrapped = app._prompt_layout(Window(), "x" * 45)
    capped = app._prompt_layout(Window(), "x" * 500)

    assert empty.prompt_height == 1
    assert short.prompt_height == 1
    assert wrapped.prompt_height == 3
    assert capped.prompt_height == 10


def test_prompt_layout_uses_collapsed_paste_marker_height(tmp_path):
    class Window:
        def getmaxyx(self):
            return 40, 28

    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))
    text = "x" * 500
    pasted_spans = [PromptPasteSpan(0, len(text))]

    layout = app._prompt_layout(Window(), text, pasted_spans=pasted_spans)

    assert layout.prompt_height == 2


def test_prompt_display_collapses_pasted_span_and_maps_cursor(tmp_path):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))
    text = "ask: pasted payload"
    pasted_spans = [PromptPasteSpan(5, len(text))]

    marker = "[14\xa0pasted characters]"

    assert app._prompt_display_text(text, pasted_spans) == f"ask: {marker}"
    assert app._prompt_display_cursor(text, 5, pasted_spans) == 5
    assert app._prompt_display_cursor(text, len(text), pasted_spans) == 5 + len(marker)
    assert app._prompt_real_cursor(text, 6, pasted_spans) == len(text)


def test_prompt_paste_spans_track_real_text_edits(tmp_path):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))
    pasted_spans: list[PromptPasteSpan] = []

    text, cursor = app._insert_prompt_text(
        "ask: ",
        5,
        "pasted payload",
        pasted_spans,
        pasted=True,
    )

    assert text == "ask: \n\npasted payload\n\n"
    assert cursor == len(text)
    assert pasted_spans == [PromptPasteSpan(5, len(text), len("pasted payload"))]

    text, cursor = app._insert_prompt_text(text, 5, "typed ", pasted_spans)

    assert text == "ask: typed \n\npasted payload\n\n"
    assert cursor == 11
    assert pasted_spans == [PromptPasteSpan(11, len(text), len("pasted payload"))]

    text, cursor = app._replace_prompt_range(text, 11, len(text), "", pasted_spans)

    assert text == "ask: typed "
    assert cursor == 11
    assert pasted_spans == []


def test_prompt_bar_draws_wrapped_input_on_multiple_rows(tmp_path):
    class Window:
        def __init__(self):
            self.writes = []

        def getmaxyx(self):
            return 20, 18

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n]))

        def move(self, y, x):
            self.cursor = (y, x)

    window = Window()
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)

    app._draw_prompt_bar(window, "x" * 15, cursor=15)

    assert (16, 6, "x" * 8) in window.writes
    assert (17, 6, "x" * 7) in window.writes


def test_prompt_bar_draws_paste_marker_with_accent_attr(tmp_path):
    class Window:
        def __init__(self):
            self.writes = []

        def getmaxyx(self):
            return 20, 80

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

        def move(self, y, x):
            self.cursor = (y, x)

    window = Window()
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)

    app._draw_prompt_bar(
        window,
        "before pasted content after",
        cursor=len("before pasted content after"),
        pasted_spans=[PromptPasteSpan(7, 21)],
    )

    assert any(
        text == "[14\xa0pasted characters]" and attr == app._attr("accent")
        for _, _, text, attr in window.writes
    )
    assert not any(text == "pasted content" for _, _, text, _ in window.writes)


def test_prompt_bar_draws_visible_cursor_cell(tmp_path):
    class Window:
        def __init__(self):
            self.writes = []

        def getmaxyx(self):
            return 20, 80

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

        def move(self, y, x):
            self.cursor = (y, x)

    window = Window()
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)

    app._draw_prompt_bar(window, "abc", cursor=1)

    assert any(text == "b" and attr == app._attr("cursor") for _, _, text, attr in window.writes)


def test_prompt_bar_draws_current_mode_hint(tmp_path):
    class Window:
        def __init__(self):
            self.writes = []

        def getmaxyx(self):
            return 20, 80

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

        def move(self, y, x):
            self.cursor = (y, x)

    window = Window()
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)

    app._draw_prompt_bar(window, "", cursor=0)

    assert (19, 4, "Ω  Standard (shift+tab to cycle)", 0) in window.writes
    assert AgentMode.AUTO.prompt_hint == "Λ  Automatic Mode (shift+tab to cycle)"
    assert AgentMode.AUTONOMOUS.prompt_hint == "Δ  Autonomous Mode (shift+tab to cycle)"


def test_prompt_bar_draws_notice_instead_of_mode_hint(tmp_path):
    class Window:
        def __init__(self):
            self.writes = []

        def getmaxyx(self):
            return 20, 80

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

        def move(self, y, x):
            self.cursor = (y, x)

    window = Window()
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)
    app._colors = {
        "accent": 1,
        "bold": 2,
        "light": 3,
        "warning": 4,
    }
    notice = "Do you really want to exit anomx? Press Ctrl+C again to confirm."

    app._draw_prompt_bar(window, "", cursor=0, notice=notice, notice_role="light")

    assert (19, 4, notice, 3) in window.writes
    assert not any("Confirm Mode" in text for _, _, text, _ in window.writes)


def test_prompt_bar_can_skip_top_rule_below_activity_panel(tmp_path):
    class Window:
        def __init__(self):
            self.writes = []

        def getmaxyx(self):
            return 20, 80

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

        def move(self, y, x):
            self.cursor = (y, x)

    window = Window()
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)

    app._draw_prompt_bar(window, "", cursor=0, draw_top_rule=False)

    assert not any(y == 16 for y, _, _, _ in window.writes)
    assert not any(
        y == 16 and x == 2 and set(text) == {"─"}
        for y, x, text, _ in window.writes
    )
    assert any(y == 17 and x == 4 and text == "›" for y, x, text, _ in window.writes)


def test_agent_mode_cycles_and_updates_runtime(tmp_path):
    home = AnomxHome(tmp_path / "home")
    app = AnomxCliApp(home=home)

    app._cycle_agent_mode()
    assert app.agent_mode == AgentMode.AUTO
    assert app.runtime.tool_manager.mode == AgentMode.AUTO
    assert app.active_agent.kind.value == "automatic"
    assert app._mode_hint_attr_name() == "warning"
    assert home.load_config()["agent_kind"] == "automatic"
    assert home.load_config()["agent_mode"] == AgentMode.AUTO.value

    app._cycle_agent_mode()
    assert app.agent_mode == AgentMode.AUTONOMOUS
    assert app.runtime.tool_manager.mode == AgentMode.AUTONOMOUS
    assert app.active_agent.kind.value == "autonomous"
    assert app._mode_hint_attr_name() == "danger"
    assert home.load_config()["agent_kind"] == "autonomous"
    assert home.load_config()["agent_mode"] == AgentMode.AUTONOMOUS.value

    app._cycle_agent_mode()
    assert app.agent_mode == AgentMode.CONFIRM
    assert app.runtime.tool_manager.mode == AgentMode.CONFIRM
    assert app.active_agent.kind.value == "standard"
    assert app._mode_hint_attr_name() == "light"
    assert home.load_config()["agent_kind"] == "standard"
    assert home.load_config()["agent_mode"] == AgentMode.CONFIRM.value


def test_agent_mode_cycles_persist_on_selected_session(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    app = AnomxCliApp(home=home, cwd=repo)
    session = home.create_session(
        repo,
        provider="openai",
        model="gpt-5.5",
        mode=AgentMode.CONFIRM,
    )

    next_mode = app._cycle_agent_mode(session)
    stored_session = home.list_sessions(limit=None)[0]
    other_session = home.create_session(
        repo,
        provider="openai",
        model="gpt-5.5",
        mode=AgentMode.CONFIRM,
    )

    assert next_mode == AgentMode.AUTO
    assert stored_session.mode == AgentMode.AUTO
    assert stored_session.agent_kind.value == "automatic"
    assert other_session.mode == AgentMode.CONFIRM
    assert app._session_mode_symbol(stored_session) == AgentMode.AUTO.symbol
    assert home.load_config()["agent_mode"] == AgentMode.CONFIRM.value


def test_app_restores_saved_agent_kind_from_config(tmp_path):
    home = AnomxHome(tmp_path / "home")
    config = home.load_config()
    config["agent_kind"] = "autonomous"
    home.save_config(config)

    app = AnomxCliApp(home=home)

    assert app.active_agent.kind.value == "autonomous"
    assert app.agent_mode == AgentMode.AUTONOMOUS
    assert app.runtime.tool_manager.mode == AgentMode.AUTONOMOUS


def test_running_ctrl_c_confirmation_requests_interrupt_without_writing(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, use_color=False)
    before = _read_jsonl(session.path)

    first = app._handle_running_key(object(), session, "\x03", "", 0, "", 0.0)

    assert first.notice == (
        "Do you really want to abort the agent? Press Ctrl+C again to confirm."
    )
    assert first.notice_role == "light"
    assert first.abort_key == "Ctrl+C"
    assert not first.exit_requested
    assert _read_jsonl(session.path) == before

    second = app._handle_running_key(
        object(),
        session,
        "\x03",
        first.input_text,
        first.cursor,
        first.abort_key,
        first.abort_deadline,
    )

    assert second.exit_requested
    assert second.notice == "Aborting agent"
    assert second.notice_role == "light"
    assert _read_jsonl(session.path) == before


def test_running_ctrl_x_confirmation_requests_interrupt_without_writing(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, use_color=False)

    first = app._handle_running_key(object(), session, "\x18", "", 0, "", 0.0)

    assert first.notice == (
        "Do you really want to abort the agent? Press Ctrl+X again to confirm."
    )
    assert first.abort_key == "Ctrl+X"
    assert not first.exit_requested

    second = app._handle_running_key(
        object(),
        session,
        "\x18",
        first.input_text,
        first.cursor,
        first.abort_key,
        first.abort_deadline,
    )

    assert second.exit_requested
    assert app._read_message_lines(session.path) == []


def test_running_ctrl_c_confirmation_aborts_turn_without_exiting_cli(tmp_path, monkeypatch):
    class Window:
        def __init__(self):
            self._keys = iter(("\x03", "\x03"))

        def get_wch(self):
            try:
                return next(self._keys)
            except StopIteration as error:
                raise curses.error from error

        def nodelay(self, flag):
            self.nodelay_flag = flag

    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, use_color=False)

    def fake_backend_response(_session_path, callbacks=None):
        del callbacks
        while not app.runtime._turn_aborted():
            runtime_module.time.sleep(0.01)
        return ""

    monkeypatch.setattr(app.runtime, "backend_response", fake_backend_response)
    monkeypatch.setattr(app, "_draw_session", lambda *args, **kwargs: None)

    result = app._run_backend_turn(Window(), session)

    assert result.exit_requested is False
    assert app.runtime._turn_aborted() is True
    assert app._read_message_lines(session.path) == [
        MessageLine("agent", MANUAL_INTERRUPT_MESSAGE)
    ]


def test_running_interrupt_collapses_completed_work_before_manual_notice(
    tmp_path, monkeypatch
):
    class Window:
        def __init__(self):
            self._calls = 0

        def get_wch(self):
            self._calls += 1
            if self._calls <= 2:
                runtime_module.time.sleep(0.05)
                raise curses.error
            if self._calls == 3:
                return "\x03"
            if self._calls == 4:
                return "\x03"
            raise curses.error

        def nodelay(self, flag):
            self.nodelay_flag = flag

    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, use_color=False)

    def fake_backend_response(_session_path, callbacks=None):
        assert callbacks is not None
        assert callbacks.tool_message is not None
        callbacks.tool_message("Checking repository")
        while not app.runtime._turn_aborted():
            runtime_module.time.sleep(0.01)
        return ""

    monkeypatch.setattr(app.runtime, "backend_response", fake_backend_response)
    monkeypatch.setattr(app, "_draw_session", lambda *args, **kwargs: None)

    app._run_backend_turn(Window(), session)

    messages = app._read_message_lines(session.path)
    assert len(messages) == 2
    assert messages[0].role == "work_summary"
    assert messages[0].text.startswith("Worked for ")
    assert messages[0].text.endswith(" min · expand")
    assert messages[1] == MessageLine("agent", MANUAL_INTERRUPT_MESSAGE)


def test_abort_current_turn_interrupts_working_workers(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    runtime = AgentRuntime(home, repo)
    worker = runtime_module.WorkerAgentState(
        worker_id="worker-1",
        name="Engineer",
        prompt="Fix this",
        status="working",
        statement="Thinking",
        started_at=runtime_module.utc_now_iso(),
        cancel_event=runtime_module.threading.Event(),
    )
    with runtime._worker_lock:
        runtime._workers[worker.worker_id] = worker

    runtime.abort_current_turn(session.path)

    assert runtime._turn_aborted()
    assert worker.cancel_event.is_set()
    assert worker.status == "interrupted"
    assert worker.statement == ""
    assert worker.finished_at
    events = _read_jsonl(session.path)
    assert any(
        event["payload"].get("type") == "worker_event"
        and event["payload"].get("worker_id") == "worker-1"
        and event["payload"].get("status") == "interrupted"
        for event in events
    )
    assert any(
        event["payload"].get("type") == "system_message"
        and event["payload"].get("role") == "worker"
        and "Worker was interrupted because Anomx was interrupted." in event["payload"].get(
            "message", ""
        )
        for event in events
    )


def test_cleanup_session_runtime_state_removes_stale_workers_and_processes(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    home.append_session_event(
        session.path,
        "worker_event",
        {
            "worker_id": "worker-1",
            "name": "Engineer",
            "status": "working",
            "statement": "Thinking",
            "prompt": "Inspect files",
        },
    )
    home.append_session_event(
        session.path,
        "worker_event",
        {
            "worker_id": "worker-2",
            "name": "Reviewer",
            "status": "ready",
            "statement": "",
            "prompt": "Review changes",
            "response": "Looks done",
        },
    )
    home.append_session_event(
        session.path,
        "process_event",
        {
            "process_id": "process-1",
            "command": "npm run dev",
            "status": "running",
            "statement": "Starting dev server",
            "source": "process",
            "pid": 12345,
        },
    )
    runtime = AgentRuntime(home, repo)

    result = runtime.cleanup_session_runtime_state(session.path)

    events = home.read_session_events(session.path)
    cleanup_payloads = [
        event_payload(event)
        for event in events
        if event_payload_type(event) in {"worker_event", "process_event"}
    ]
    assert result.workers_removed == 2
    assert result.processes_ended == 1
    assert worker_snapshots(events) == ()
    assert running_process_snapshots(events) == ()
    assert any(
        payload.get("worker_id") == "worker-1"
        and payload.get("status") == "removed"
        for payload in cleanup_payloads
    )
    assert any(
        payload.get("process_id") == "process-1"
        and payload.get("status") == "ended"
        and payload.get("pid") == 12345
        for payload in cleanup_payloads
    )


def test_runtime_shutdown_removes_workers_and_ends_live_processes(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    runtime = AgentRuntime(home, repo)
    started = json.loads(
        runtime._execute_tool(
            "start_process",
            {"statement": "Starting process", "command": "sleep 10"},
            RuntimeCallbacks(approval=lambda _request: ApprovalChoice.ALLOW),
            session.path,
        )
    )
    worker = runtime_module.WorkerAgentState(
        worker_id="worker-1",
        name="Engineer",
        prompt="Fix this",
        status="ready",
        statement="",
        started_at=runtime_module.utc_now_iso(),
        cancel_event=runtime_module.threading.Event(),
        session_path=session.path,
    )
    with runtime._worker_lock:
        runtime._workers[worker.worker_id] = worker
    with runtime._process_lock:
        process_state = runtime._processes[started["process_id"]]

    result = runtime.shutdown()

    events = home.read_session_events(session.path)
    assert result.workers_removed == 1
    assert result.processes_ended == 1
    assert worker.cancel_event.is_set()
    assert process_state.process.poll() is not None
    assert worker_snapshots(events) == ()
    assert running_process_snapshots(events) == ()


def test_worker_cancel_event_counts_as_turn_abort(tmp_path):
    cancel_event = runtime_module.threading.Event()
    runtime = AgentRuntime(
        AnomxHome(tmp_path / "home"),
        tmp_path,
        role=AgentRole.WORKER,
        cancel_event=cancel_event,
    )

    assert not runtime._turn_aborted()

    cancel_event.set()

    assert runtime._turn_aborted()


def test_idle_ctrl_c_requires_confirmation_in_prompt_notice(tmp_path, monkeypatch):
    class Window:
        def __init__(self):
            self._keys = iter(("\x03", "\x03"))

        def get_wch(self):
            return next(self._keys)

    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, use_color=False)
    notices = []

    def capture_draw(*_args, **kwargs):
        notices.append(kwargs.get("prompt_notice", ""))

    monkeypatch.setattr(app, "_draw_session", capture_draw)

    assert app._run_session(Window(), session) == 0
    assert notices == [
        "",
        "Do you really want to exit anomx? Press Ctrl+C again to confirm.",
    ]


def test_idle_ctrl_c_clears_prompt_before_exit_confirmation(tmp_path, monkeypatch):
    class Window:
        def __init__(self):
            self._keys = iter(("h", "\x03", "\x03", "\x03"))

        def get_wch(self):
            return next(self._keys)

    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, use_color=False)
    draws: list[tuple[str, str]] = []

    def capture_draw(*args, **kwargs):
        draws.append((str(args[3]), str(kwargs.get("prompt_notice", ""))))

    monkeypatch.setattr(app, "_draw_session", capture_draw)

    assert app._run_session(Window(), session) == 0
    assert ("h", "") in draws
    assert ("", "") in draws
    assert draws[-1] == (
        "",
        "Do you really want to exit anomx? Press Ctrl+C again to confirm.",
    )


def test_prompt_cursor_stays_on_previous_line_at_wrap_boundary(tmp_path):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))

    assert app._prompt_cursor_position("abcdefghij", cursor=10, width=10) == (0, 9)


def test_prompt_option_word_jumps_use_readline_style_boundaries(tmp_path):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))
    text = "alpha beta_gamma, delta"

    assert app._next_prompt_word(text, 0) == 5
    assert app._next_prompt_word(text, 5) == 16
    assert app._previous_prompt_word(text, 17) == 6
    assert app._previous_prompt_word(text, len(text)) == 18


def test_prompt_cursor_moves_between_wrapped_rows(tmp_path):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))
    text = "abcdefghijklmnopqrstuvwxyz"

    assert app._prompt_cursor_for_row_delta(text, cursor=15, width=10, direction=-1) == 5
    assert app._prompt_cursor_for_row_delta(text, cursor=5, width=10, direction=1) == 15
    assert app._prompt_cursor_for_row_delta(text, cursor=15, width=10, direction=1) == 25
    assert app._prompt_cursor_for_row_delta(text, cursor=5, width=10, direction=-1) == 5
    assert app._prompt_cursor_for_row_delta(text, cursor=25, width=10, direction=1) == 25


def test_running_arrow_keys_move_wrapped_prompt_rows_before_scrolling(tmp_path):
    class Window:
        def getmaxyx(self):
            return 40, 18

    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=repo)
    text = "abcdefghijklmno"

    moved_up = app._handle_running_key(
        Window(),
        session,
        curses.KEY_UP,
        text,
        15,
        "",
        0.0,
    )
    moved_down = app._handle_running_key(
        Window(),
        session,
        curses.KEY_DOWN,
        text,
        moved_up.cursor,
        "",
        0.0,
    )
    scrolled = app._handle_running_key(
        Window(),
        session,
        curses.KEY_UP,
        text,
        moved_up.cursor,
        "",
        0.0,
    )

    assert moved_up.cursor == 7
    assert moved_up.scroll_delta == 0
    assert moved_down.cursor == 15
    assert moved_down.scroll_delta == 0
    assert scrolled.cursor == 7
    assert scrolled.scroll_delta == 1


def test_running_option_arrows_jump_words(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=repo)
    text = "alpha beta"

    left = app._handle_running_key(object(), session, "\x1bb", text, len(text), "", 0.0)
    right = app._handle_running_key(object(), session, "\x1bf", text, 0, "", 0.0)

    assert left.cursor == 6
    assert right.cursor == 5


def test_prompt_reader_combines_meta_escape_prefix(tmp_path):
    class Window:
        def __init__(self):
            self._keys = iter(("b",))
            self.nodelay_calls = []

        def get_wch(self):
            return next(self._keys)

        def nodelay(self, flag):
            self.nodelay_calls.append(flag)

    window = Window()
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))

    assert app._complete_escape_key(window, "\x1b", restore_blocking=True) == "\x1bb"
    assert window.nodelay_calls == [True, False]


def test_prompt_reader_combines_raw_mouse_escape_sequence(tmp_path):
    class Window:
        def __init__(self):
            self._keys = iter("[<65;35;47M")
            self.nodelay_calls = []

        def get_wch(self):
            return next(self._keys)

        def nodelay(self, flag):
            self.nodelay_calls.append(flag)

    window = Window()
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))

    assert app._complete_escape_key(window, "\x1b", restore_blocking=True) == "\x1b[<65;35;47M"
    assert app._is_raw_mouse_fragment_key("<65;35;47M")
    assert window.nodelay_calls == [True, False]


def test_session_scroll_bounds_allow_bottom_slack(tmp_path):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))

    assert app._session_scroll_bounds(40, 10) == (-9, 30)
    assert app._session_view_start(-9, 40, 10) == 39
    assert app._session_view_start(30, 40, 10) == 0


def test_draw_session_clamps_overscroll_to_real_top(tmp_path):
    class Window:
        def erase(self):
            pass

        def getmaxyx(self):
            return 28, 80

        def addnstr(self, y, x, text, n, attr=0):
            del y, x, text, n, attr

        def refresh(self):
            pass

    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=repo, use_color=False)
    messages = [MessageLine("user", f"Line {index}") for index in range(40)]

    viewport = app._draw_session(Window(), session, messages, "", 0, 999)

    assert viewport.start == 0
    assert viewport.scroll == app._session_scroll_bounds(
        viewport.rendered_line_count,
        viewport.body_height,
    )[1]


def test_draw_session_allows_last_line_to_reach_top(tmp_path):
    class Window:
        def erase(self):
            pass

        def getmaxyx(self):
            return 28, 80

        def addnstr(self, y, x, text, n, attr=0):
            del y, x, text, n, attr

        def refresh(self):
            pass

    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=repo, use_color=False)
    messages = [MessageLine("user", f"Line {index}") for index in range(40)]

    viewport = app._draw_session(Window(), session, messages, "", 0, -999)

    assert viewport.start == viewport.rendered_line_count - 1
    assert viewport.scroll == app._session_scroll_bounds(
        viewport.rendered_line_count,
        viewport.body_height,
    )[0]


def test_draw_session_maps_anchor_to_equivalent_scroll(tmp_path):
    class Window:
        def erase(self):
            pass

        def getmaxyx(self):
            return 28, 80

        def addnstr(self, y, x, text, n, attr=0):
            del y, x, text, n, attr

        def refresh(self):
            pass

    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=repo, use_color=False)
    messages = [MessageLine("user", f"Line {index}") for index in range(40)]

    viewport = app._draw_session(Window(), session, messages, "", 0, 0, anchor_line=21)

    assert viewport.start == 21
    assert viewport.scroll == app._session_scroll_for_start(
        21,
        viewport.rendered_line_count,
        viewport.body_height,
    )


def test_draw_session_collapses_and_expands_sticky_user_anchor(tmp_path):
    class Window:
        def __init__(self):
            self.writes = []

        def erase(self):
            pass

        def getmaxyx(self):
            return 28, 80

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

        def refresh(self):
            pass

    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=repo, use_color=False)
    message = "Initial request " + ("with a lot of context " * 8) + "UNIQUE_TAIL_TOKEN"
    messages = [
        MessageLine("user", message, expansion_key="user:0"),
        MessageLine("agent", "Working on it"),
    ]
    window = Window()

    app._draw_session(
        window,
        session,
        messages,
        "",
        0,
        0,
        anchor_line=0,
        sticky_anchor=True,
    )

    pinned_actions = [
        action
        for actions in app._click_targets.values()
        for action in actions
        if action.kind == "toggle_pinned_user"
    ]
    assert len(pinned_actions) == 1
    assert any(
        x == 4 and text.startswith("Initial request") and text.endswith(" Expand")
        for _, x, text, _ in window.writes
    )
    assert not any("UNIQUE_TAIL_TOKEN" in text for _, _, text, _ in window.writes)

    app._toggle_pinned_user(pinned_actions[0].text)
    expanded_window = Window()
    app._draw_session(
        expanded_window,
        session,
        messages,
        "",
        0,
        0,
        anchor_line=0,
        sticky_anchor=True,
    )

    assert any("UNIQUE_TAIL_TOKEN" in text for _, _, text, _ in expanded_window.writes)


def test_draw_session_keeps_sticky_anchor_on_original_user_message(tmp_path):
    class Window:
        def __init__(self):
            self.writes = []

        def erase(self):
            pass

        def getmaxyx(self):
            return 28, 80

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

        def refresh(self):
            pass

    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=repo, use_color=False)
    window = Window()
    messages = [
        MessageLine("user", "Initial request " + ("context " * 20), expansion_key="user:0"),
        MessageLine("agent_intermediate", "Old response above the correction"),
        MessageLine(
            "user",
            "Intermediate user correction with FULL_LENGTH_TOKEN",
            expansion_key="user:2",
        ),
        MessageLine("agent_intermediate", "New response below the correction"),
    ]

    app._draw_session(
        window,
        session,
        messages,
        "",
        0,
        0,
        anchor_line=0,
        sticky_anchor=True,
    )

    pinned_y = next(
        y
        for y, actions in app._click_targets.items()
        if any(action.kind == "toggle_pinned_user" for action in actions)
    )
    pinned_writes = [text for y, x, text, _ in window.writes if y == pinned_y and x == 4]
    assert pinned_writes
    assert pinned_writes[0].startswith("Initial request")
    assert any(
        text.startswith("Intermediate user correction with FULL_LENGTH_TOKEN")
        and text.endswith(" Expand")
        for _, _, text, _ in window.writes
    )


def test_active_turn_anchor_recovers_initial_user_message(tmp_path):
    class Window:
        def getmaxyx(self):
            return 28, 80

    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    home.append_session_event(session.path, "user_message", {"message": "Initial request"})
    home.append_session_event(
        session.path,
        "agent_message",
        {"message": "Progress update", "turn_id": "turn-1", "intermediate": True},
    )
    home.append_session_event(
        session.path,
        "user_message",
        {
            "message": "Intermediate correction",
            "turn_id": "turn-1",
            "intermediate": True,
        },
    )
    app = AnomxCliApp(home=home, cwd=repo, use_color=False)
    initial_key = app._latest_root_user_expansion_key(session.path)
    worker = threading.Thread(target=lambda: None)
    worker.start()
    worker.join()
    turn = ui_module.ActiveSessionTurn(
        session=session,
        runtime=app.runtime,
        events=queue.SimpleQueue(),
        result={},
        turn_id="turn-1",
        started_at=time.monotonic(),
        worker=worker,
        mode=AgentMode.CONFIRM,
        anchor_expansion_key=initial_key,
    )

    anchor_line = app._active_turn_anchor_line(Window(), session, turn, None)
    rendered = app._session_rendered_lines(
        session,
        app._read_message_lines(session.path),
        width=72,
    )

    assert anchor_line is not None
    assert rendered[anchor_line].expansion_key == initial_key
    assert rendered[anchor_line].text.startswith("Initial request")


def test_panel_text_lines_sanitize_multiline_commands(tmp_path):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))

    lines = app._panel_text_lines("sh -lc 'cat <<EOF\nhello\nEOF'", width=80, max_lines=3)

    assert lines == ["sh -lc 'cat <<EOF / hello / EOF'"]


def test_bottom_panel_viewport_shows_hidden_session_counts(tmp_path):
    class Window:
        def getmaxyx(self):
            return 32, 80

    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)
    choices = tuple(
        MenuChoice(f"Session {index}", str(index), f"detail {index}") for index in range(18)
    )
    panel = BottomPanel("Open Session", "Choose a stored session", choices, selected=14)

    viewport = app._bottom_panel_viewport(Window(), panel)

    assert viewport.show_overflow_counts is True
    assert viewport.more_above == 4
    assert viewport.more_below == 3
    assert viewport.visible_indices == tuple(range(4, 15))


def test_openai_backend_requires_configured_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5")
    app = AnomxCliApp(home=home, cwd=repo)

    response = app._openai_response(session.path, "gpt-5")

    assert "OpenAI API key is not configured" in response


def test_anthropic_backend_requires_configured_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    config = home.load_config()
    config["provider"] = "anthropic"
    config["model"] = "claude-opus-4-8"
    home.save_config(config)
    session = home.create_session(repo, provider="anthropic", model="claude-opus-4-8")
    runtime = AgentRuntime(home, repo)

    response = runtime.backend_response(session.path)

    assert "Anthropic API key is not configured" in response


def test_desy_backend_requires_configured_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("DESY_ASSISTANT_API_KEY", raising=False)
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    config = home.load_config()
    config["provider"] = "desy"
    config["model"] = "desy-assistant"
    home.save_config(config)
    session = home.create_session(repo, provider="desy", model="desy-assistant")
    runtime = AgentRuntime(home, repo)

    response = runtime.backend_response(session.path)

    assert "DESY Assistant API key is not configured" in response


def test_openai_invalid_api_key_uses_specific_message(tmp_path):
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path)

    response = runtime._api_error(
        "openai",
        "OpenAI",
        "OPENAI_API_KEY",
        401,
        json.dumps(
            {
                "error": {
                    "type": "authentication_error",
                    "message": "Incorrect API key provided: sk-invalid",
                }
            }
        ),
    )

    assert "OpenAI credentials were rejected" in response
    assert "OPENAI_API_KEY" in response


def test_anthropic_invalid_api_key_uses_specific_message(tmp_path):
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path)

    response = runtime._api_error(
        "anthropic",
        "Anthropic",
        "ANTHROPIC_API_KEY",
        401,
        json.dumps(
            {
                "type": "error",
                "error": {
                    "type": "authentication_error",
                    "message": "The API key provided is invalid or has been revoked.",
                },
            }
        ),
    )

    assert "Anthropic credentials were rejected" in response
    assert "ANTHROPIC_API_KEY" in response


def test_desy_invalid_api_key_uses_specific_message(tmp_path):
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path)

    response = runtime._api_error(
        "desy",
        "DESY Assistant",
        "DESY_ASSISTANT_API_KEY",
        401,
        json.dumps({"detail": "Not authenticated"}),
    )

    assert "DESY Assistant credentials were rejected" in response
    assert "DESY_ASSISTANT_API_KEY" in response


def test_runtime_suggests_session_title_without_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    home.append_session_event(
        session.path,
        "user_message",
        {"message": "Find anomalies in repository data"},
    )
    runtime = AgentRuntime(home, repo)

    assert runtime.suggest_session_title(session.path) == "Find anomalies in repository data"


def test_runtime_suggests_session_continuation_without_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    home.append_session_event(
        session.path,
        "user_message",
        {"message": "Identify important data channels"},
    )
    runtime = AgentRuntime(home, repo)

    assert runtime.suggest_session_continuation(session.path, "repo") == (
        "Do you want to continue the previous session titled Identify important data channels?"
    )


def test_context_status_is_shown_after_first_message(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    home.append_session_event(session.path, "user_message", {"message": "Inspect this repo"})
    app = AnomxCliApp(home=home, cwd=repo)

    context_status = app._context_status(session, "gpt-5.5")
    assert context_status.endswith("% Context")
    assert app._session_header_lines(session, "gpt-5.5") == (str(repo.resolve()),)
    assert app._session_header_meta(session, "openai", "gpt-5.5").endswith(
        f" · {context_status}"
    )


def test_context_status_uses_backend_message_window_not_full_transcript(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    old_large_message = "old context " * 10_000
    for _ in range(5):
        home.append_session_event(session.path, "user_message", {"message": old_large_message})
        home.append_session_event(session.path, "agent_message", {"message": old_large_message})
    for index in range(10):
        home.append_session_event(session.path, "user_message", {"message": f"recent {index}"})
        home.append_session_event(session.path, "agent_message", {"message": f"done {index}"})
    app = AnomxCliApp(home=home, cwd=repo)

    assert len(app.runtime.conversation_messages(session.path)) == 20
    assert app._estimate_context_tokens(session.path) < 10_000


def test_session_header_lines_keep_location_as_subtitle(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=repo)

    assert app._session_header_lines(session, "gpt-5.5") == (
        str(repo.resolve()),
    )
    assert app._session_header_meta(session, "openai", "gpt-5.5") == (
        f"{session.session_id[:8]} · openai/gpt-5.5"
    )


def test_session_header_meta_shows_supported_thinking_intensity(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    config = home.load_config()
    config["thinking_intensity"] = "high"
    home.save_config(config)
    app = AnomxCliApp(home=home, cwd=repo)

    assert app._session_header_meta(session, "openai", "gpt-5.5") == (
        f"{session.session_id[:8]} · openai/gpt-5.5 (H)"
    )


def test_session_header_meta_omits_unsupported_thinking_intensity(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="desy", model="reasoning")
    config = home.load_config()
    config["thinking_intensity"] = "high"
    home.save_config(config)
    app = AnomxCliApp(home=home, cwd=repo)

    assert app._session_header_meta(session, "desy", "reasoning") == (
        f"{session.session_id[:8]} · desy/reasoning"
    )


def test_openai_text_extraction_prefers_output_text(tmp_path):
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path)

    assert runtime.extract_openai_text({"output_text": "  ready  "}) == "ready"


def test_openai_text_extraction_reads_output_content(tmp_path):
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path)

    response = runtime.extract_openai_text(
        {
            "output": [
                {
                    "content": [
                        {"type": "output_text", "text": "first"},
                        {"type": "output_text", "text": "second"},
                    ]
                }
            ]
        }
    )

    assert response == "first\nsecond"


def test_anthropic_text_extraction_reads_text_blocks(tmp_path):
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path)

    response = runtime.extract_anthropic_text(
        {
            "content": [
                {"type": "text", "text": "first"},
                {"type": "tool_use", "name": "run_command", "id": "toolu_1", "input": {}},
                {"type": "text", "text": "second"},
            ]
        }
    )

    assert response == "first\nsecond"


def test_openai_stream_reports_reasoning_summary_status(tmp_path, monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def __iter__(self):
            yield (
                b'data: {"type":"response.reasoning_summary_text.delta",'
                b'"delta":"Checking the repo"}\n'
            )
            yield b'data: {"type":"response.output_text.delta","delta":"done"}\n'
            yield b'data: {"type":"response.completed","response":{"id":"resp_1"}}\n'
            yield b"data: [DONE]\n"

    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path)
    statuses: list[str] = []
    deltas: list[str] = []

    monkeypatch.setattr(
        runtime_module.urllib.request,
        "urlopen",
        lambda *args, **kwargs: FakeResponse(),
    )

    response = runtime._stream_openai_response(
        "sk-test",
        {"model": "gpt-5.5", "input": [], "stream": True},
        deltas.append,
        statuses.append,
    )

    assert response.response_id == "resp_1"
    assert response.text == "done"
    assert statuses == ["Thinking"]
    assert deltas == ["done"]


def test_openai_stream_retries_transient_http_errors_before_success(
    tmp_path,
    monkeypatch,
):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def __iter__(self):
            yield b'data: {"type":"response.output_text.delta","delta":"done"}\n'
            yield b'data: {"type":"response.completed","response":{"id":"resp_1"}}\n'
            yield b"data: [DONE]\n"

    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path)
    attempts = 0
    statuses: list[str] = []

    def fake_urlopen(request, timeout=120):
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            raise HTTPError(
                request.full_url,
                503,
                "Service Unavailable",
                hdrs=None,
                fp=io.BytesIO(b'{"error":{"message":"temporary"}}'),
            )
        return FakeResponse()

    monkeypatch.setattr(runtime_module, "MODEL_REQUEST_RETRY_COUNT", 2)
    monkeypatch.setattr(runtime_module, "MODEL_REQUEST_RETRY_INITIAL_DELAY_SECONDS", 0.0)
    monkeypatch.setattr(runtime_module.urllib.request, "urlopen", fake_urlopen)

    response = runtime._stream_openai_response(
        "sk-test",
        {"model": "gpt-5.5", "input": [], "stream": True},
        None,
        statuses.append,
    )

    assert isinstance(response, runtime_module.OpenAIStreamResponse)
    assert response.text == "done"
    assert attempts == 3
    assert statuses == ["Reconnecting", "Reconnecting"]


def test_openai_stream_returns_http_error_after_retries_are_exhausted(
    tmp_path,
    monkeypatch,
):
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path)
    attempts = 0
    statuses: list[str] = []

    def fake_urlopen(request, timeout=120):
        nonlocal attempts
        attempts += 1
        raise HTTPError(
            request.full_url,
            503,
            "Service Unavailable",
            hdrs=None,
            fp=io.BytesIO(b'{"error":{"message":"temporary"}}'),
        )

    monkeypatch.setattr(runtime_module, "MODEL_REQUEST_RETRY_COUNT", 2)
    monkeypatch.setattr(runtime_module, "MODEL_REQUEST_RETRY_INITIAL_DELAY_SECONDS", 0.0)
    monkeypatch.setattr(runtime_module.urllib.request, "urlopen", fake_urlopen)

    response = runtime._stream_openai_response(
        "sk-test",
        {"model": "gpt-5.5", "input": [], "stream": True},
        None,
        statuses.append,
    )

    assert response == "OpenAI request failed (503): temporary"
    assert attempts == 3
    assert statuses == ["Reconnecting", "Reconnecting"]


def test_debug_normalizes_tool_blocks_as_message_keys():
    payload = {
        "system": "System instructions",
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Checking files"},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "run_command",
                        "input": {"command": "pwd"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "done",
                    }
                ],
            },
            {
                "role": "tool",
                "tool_name": "run_command",
                "content": "ollama result",
            },
        ],
        "input": [
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "openai result",
            }
        ],
    }

    messages = SessionDebugLogger.normalize_payload_messages(payload)

    assert messages[0] == {"role": "system", "content": "System instructions"}
    assert messages[1] == {
        "role": "assistant",
        "content": "Checking files",
        "tool_use": [
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "run_command",
                "input": {"command": "pwd"},
            }
        ],
    }
    assert messages[2] == {
        "role": "user",
        "content": "",
        "tool_result": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_1",
                "content": "done",
            }
        ],
    }
    assert messages[3] == {
        "role": "tool",
        "content": "",
        "tool_result": [
            {
                "type": "tool_result",
                "tool_name": "run_command",
                "content": "ollama result",
            }
        ],
    }
    assert messages[4] == {
        "role": "tool",
        "content": "",
        "tool_result": [
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "openai result",
            }
        ],
    }


def test_full_session_logs_write_each_backend_request_as_txt(tmp_path, monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def __iter__(self):
            yield b'data: {"type":"response.output_text.delta","delta":"done"}\n'
            yield b'data: {"type":"response.completed","response":{"id":"resp_1"}}\n'
            yield b"data: [DONE]\n"

    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    logs_dir = tmp_path / "request-logs"
    home.set_api_key("openai", "sk-test")
    config = home.load_config()
    config["provider"] = "openai"
    config["model"] = "gpt-5.5"
    config["debug_mode"] = True
    config["debug_full_session_logs"] = True
    config["debug_full_session_logs_path"] = str(logs_dir)
    home.save_config(config)
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    home.append_session_event(
        session.path,
        "user_message",
        {"message": "Inspect metrics"},
    )
    home.update_session_title(session.path, "Investigate Sensors")
    runtime = AgentRuntime(home, repo)

    monkeypatch.setattr(
        runtime_module.urllib.request,
        "urlopen",
        lambda *args, **kwargs: FakeResponse(),
    )

    first_response = runtime.backend_response(session.path)
    second_response = runtime.backend_response(session.path)

    assert first_response == "done"
    assert second_response == "done"
    session_logs_dir = logs_dir / f"{session.session_id} - Investigate Sensors"
    orchestrator_logs_dir = session_logs_dir / "Orchestrator"
    request_logs = sorted(path.name for path in orchestrator_logs_dir.glob("*.txt"))
    assert request_logs == ["req-0001.txt", "req-0002.txt"]
    assert not list(session_logs_dir.glob("req-*.txt"))
    text = (orchestrator_logs_dir / "req-0001.txt").read_text(encoding="utf-8")
    assert "=" * 88 in text
    assert "-" * 88 in text
    assert "Anomx AI backend request" in text
    assert "provider: openai" in text
    assert "Message 0001 | role: system" in text
    assert "Message 0002 | role: user" in text
    assert "Inspect metrics" in text
    assert "Available tools:" in text
    assert "run_command" in text
    assert "Raw request payload (JSON, exact backend input):" in text


def test_full_session_logs_route_worker_backend_requests_by_agent(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    logs_dir = tmp_path / "request-logs"
    config = home.load_config()
    config["debug_mode"] = True
    config["debug_full_session_logs"] = True
    config["debug_full_session_logs_path"] = str(logs_dir)
    home.save_config(config)
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    home.update_session_title(session.path, "Investigate Sensors")

    path = home.write_backend_request_log(
        provider="openai",
        payload={
            "messages": [{"role": "user", "content": "Inspect worker context"}],
            "tools": [],
        },
        session_path=session.path,
        actor="worker",
        worker_name="Engineer",
        worker_id="172543",
    )

    worker_logs_dir = (
        logs_dir
        / f"{session.session_id} - Investigate Sensors"
        / "Worker"
        / "Engineer (172543)"
    )
    assert path == worker_logs_dir / "req-0001.txt"
    assert path is not None
    assert path.exists()
    assert "Inspect worker context" in path.read_text(encoding="utf-8")


def test_full_session_logs_write_async_commands_and_processes(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    logs_dir = tmp_path / "request-logs"
    config = home.load_config()
    config["debug_mode"] = True
    config["debug_full_session_logs"] = True
    config["debug_full_session_logs_path"] = str(logs_dir)
    home.save_config(config)
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    home.update_session_title(session.path, "Investigate Sensors")
    payload = {
        "process_id": "cmd123",
        "command_id": "cmd123",
        "source": "command",
        "statement": "Run dev server",
        "command": "npm run dev",
        "status": "ended",
        "started_at": "2026-06-07T12:00:00Z",
        "finished_at": "2026-06-07T12:00:05Z",
        "exit_code": 0,
        "pid": 12345,
    }

    first_command = home.write_async_execution_log(
        session_path=session.path,
        kind="command",
        payload=payload,
        output="server started\\nready",
    )
    second_command = home.write_async_execution_log(
        session_path=session.path,
        kind="command",
        payload={**payload, "process_id": "cmd456", "command_id": "cmd456"},
        output="second",
    )
    process_log = home.write_async_execution_log(
        session_path=session.path,
        kind="process",
        payload={**payload, "source": "process", "process_id": "proc789"},
        output="process line",
    )

    session_logs_dir = logs_dir / f"{session.session_id} - Investigate Sensors"
    assert first_command == session_logs_dir / "Commands" / "0001.txt"
    assert second_command == session_logs_dir / "Commands" / "0002.txt"
    assert process_log == session_logs_dir / "Processes" / "0001.txt"
    assert first_command is not None
    text = first_command.read_text(encoding="utf-8")
    readable = text.split("Raw process payload", maxsplit=1)[0]
    assert "Anomx long-running command log" in readable
    assert "command_id: cmd123" in readable
    assert "server started\n" in readable
    assert "ready" in readable
    assert "server started\\nready" not in readable


def test_backend_request_log_prettifies_escaped_tool_output(tmp_path):
    home = AnomxHome(tmp_path / "home")
    logs_dir = tmp_path / "request-logs"
    config = home.load_config()
    config["debug_mode"] = True
    config["debug_full_session_logs"] = True
    config["debug_full_session_logs_path"] = str(logs_dir)
    home.save_config(config)
    payload = {
        "model": "gpt-5.5",
        "messages": [
            {
                "role": "user",
                "content": (
                    '{"approved": true, "output": "first line\\nsecond line", '
                    '"tool_use_id": "call_1", "type": "tool_result"}'
                ),
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "run_command",
                        "input": {
                            "statement": "Read generated app",
                            "command": "cat app.js",
                        },
                    }
                ],
            },
        ],
        "tools": [{"name": "run_command", "description": "Run a command"}],
    }

    path = home.write_backend_request_log(provider="openai", payload=payload)

    assert path is not None
    text = path.read_text(encoding="utf-8")
    readable = text.split("Raw request payload", maxsplit=1)[0]
    assert "Message 0001 | role: user" in readable
    assert "output:" in readable
    assert "first line\n" in readable
    assert "second line" in readable
    assert "first line\\nsecond line" not in readable
    assert "Tool 0001: run_command" in readable


def test_anthropic_stream_reports_thinking_status(tmp_path, monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def __iter__(self):
            yield (
                b'data: {"type":"content_block_start","index":0,'
                b'"content_block":{"type":"thinking","thinking":"","signature":""}}\n'
            )
            yield (
                b'data: {"type":"content_block_delta","index":0,'
                b'"delta":{"type":"thinking_delta","thinking":"Plan the tool call"}}\n'
            )
            yield (
                b'data: {"type":"content_block_delta","index":0,'
                b'"delta":{"type":"signature_delta","signature":"sig_1"}}\n'
            )
            yield b'data: {"type":"content_block_stop","index":0}\n'
            yield (
                b'data: {"type":"content_block_start","index":1,'
                b'"content_block":{"type":"text","text":""}}\n'
            )
            yield (
                b'data: {"type":"content_block_delta","index":1,'
                b'"delta":{"type":"text_delta","text":"done"}}\n'
            )
            yield b'data: {"type":"content_block_stop","index":1}\n'

    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path)
    statuses: list[str] = []
    deltas: list[str] = []

    monkeypatch.setattr(
        runtime_module.urllib.request,
        "urlopen",
        lambda *args, **kwargs: FakeResponse(),
    )

    response = runtime._stream_anthropic_response(
        "sk-ant-test",
        {"model": "claude-opus-4-8", "messages": [], "stream": True},
        deltas.append,
        statuses.append,
    )

    assert response.text == "done"
    assert statuses == ["Thinking"]
    assert deltas == ["done"]
    assert response.content[0]["type"] == "thinking"
    assert response.content[0]["signature"] == "sig_1"


def test_backend_response_applies_openai_thinking_intensity_config(
    tmp_path,
    monkeypatch,
):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    home.set_api_key("openai", "sk-test")
    config = home.load_config()
    config["provider"] = "openai"
    config["model"] = "gpt-5.5"
    config["thinking_intensity"] = "high"
    home.save_config(config)
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    runtime = AgentRuntime(home, repo)
    captured_payloads: list[dict[str, object]] = []

    def fake_stream(api_key, payload, delta_callback, status_callback):
        assert api_key == "sk-test"
        captured_payloads.append(payload)
        return runtime_module.OpenAIStreamResponse("resp_1", "done", ())

    monkeypatch.setattr(runtime, "_stream_openai_response", fake_stream)

    response = runtime.backend_response(session.path)

    assert response == "done"
    assert captured_payloads[0]["reasoning"] == {"summary": "auto", "effort": "high"}


def test_openai_auto_thinking_intensity_omits_effort(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    home.set_api_key("openai", "sk-test")
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    runtime = AgentRuntime(home, repo)
    captured_payloads: list[dict[str, object]] = []

    def fake_stream(_api_key, payload, _delta_callback, _status_callback):
        captured_payloads.append(payload)
        return runtime_module.OpenAIStreamResponse("resp_1", "done", ())

    monkeypatch.setattr(runtime, "_stream_openai_response", fake_stream)

    response = runtime.openai_response(
        session.path,
        "gpt-5.5",
        thinking_intensity="auto",
    )

    assert response == "done"
    assert captured_payloads[0]["reasoning"] == {"summary": "auto"}


def test_openai_response_suppresses_provisional_stream_deltas(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    home.set_api_key("openai", "sk-test")
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    runtime = AgentRuntime(home, repo)
    deltas: list[str] = []

    def fake_stream(_api_key, _payload, delta_callback, _status_callback):
        assert delta_callback is None
        return runtime_module.OpenAIStreamResponse("resp_1", "final answer", ())

    monkeypatch.setattr(runtime, "_stream_openai_response", fake_stream)

    response = runtime.openai_response(
        session.path,
        "gpt-5.5",
        RuntimeCallbacks(delta=deltas.append),
    )

    assert response == "final answer"
    assert deltas == []


def test_anthropic_response_applies_supported_thinking_intensity(
    tmp_path,
    monkeypatch,
):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    home.set_api_key("anthropic", "sk-ant-test")
    session = home.create_session(repo, provider="anthropic", model="claude-opus-4-8")
    runtime = AgentRuntime(home, repo)
    captured_payloads: list[dict[str, object]] = []

    def fake_stream(api_key, payload, delta_callback, status_callback):
        assert api_key == "sk-ant-test"
        captured_payloads.append(payload)
        return runtime_module.AnthropicStreamResponse(
            text="done",
            tool_calls=(),
            content=({"type": "text", "text": "done"},),
        )

    monkeypatch.setattr(runtime, "_stream_anthropic_response", fake_stream)

    response = runtime.anthropic_response(
        session.path,
        "claude-opus-4-8",
        thinking_intensity="xhigh",
    )

    assert response == "done"
    assert captured_payloads[0]["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert captured_payloads[0]["output_config"] == {"effort": "xhigh"}


def test_desy_response_omits_thinking_config(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    home.set_api_key("desy", "sk-desy-test")
    session = home.create_session(repo, provider="desy", model="desy-assistant")
    runtime = AgentRuntime(home, repo)
    captured_payloads: list[dict[str, object]] = []

    def fake_stream(api_key, payload, delta_callback, status_callback):
        assert api_key == "sk-desy-test"
        captured_payloads.append(payload)
        return runtime_module.AnthropicStreamResponse(
            text="done",
            tool_calls=(),
            content=({"type": "text", "text": "done"},),
        )

    monkeypatch.setattr(runtime, "_stream_desy_response", fake_stream)

    response = runtime.desy_response(session.path, "desy-assistant")

    assert response == "done"
    assert "thinking" not in captured_payloads[0]


def test_desy_payload_uses_configured_max_output_tokens(tmp_path, monkeypatch):
    from anomx.agent.backends.desy_assistant import DesyAssistantBackend

    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    home.set_api_key("desy", "sk-desy-test")
    session = home.create_session(repo, provider="desy", model="desy-assistant")
    runtime = AgentRuntime(home, repo)
    captured_payloads: list[dict[str, object]] = []

    def fake_stream(_self, _api_key, payload, _delta_callback, _status_callback):
        captured_payloads.append(payload)
        return runtime_module.AnthropicStreamResponse(
            text="done",
            tool_calls=(),
            content=({"type": "text", "text": "done"},),
        )

    monkeypatch.setattr(DesyAssistantBackend, "_stream_response", fake_stream)

    # desy-assistant has no metadata max_output_tokens, so the fallback applies.
    runtime.desy_response(session.path, "desy-assistant")
    assert captured_payloads[0]["max_tokens"] == 8192

    config = home.load_config()
    config["max_output_tokens"] = 20000
    home.save_config(config)
    captured_payloads.clear()
    runtime.desy_response(session.path, "desy-assistant")
    assert captured_payloads[0]["max_tokens"] == 20000


def test_desy_loop_continues_after_max_tokens_truncation(tmp_path, monkeypatch):
    from anomx.agent.backends.desy_assistant import DesyAssistantBackend

    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    home.set_api_key("desy", "sk-desy-test")
    session = home.create_session(repo, provider="desy", model="desy-assistant")
    runtime = AgentRuntime(home, repo)
    payloads: list[dict[str, object]] = []
    responses = [
        runtime_module.AnthropicStreamResponse(
            text="part one",
            tool_calls=(),
            content=({"type": "text", "text": "part one"},),
            stop_reason="max_tokens",
        ),
        runtime_module.AnthropicStreamResponse(
            text="part two",
            tool_calls=(),
            content=({"type": "text", "text": "part two"},),
            stop_reason="end_turn",
        ),
    ]

    def fake_stream(_self, _api_key, payload, _delta_callback, _status_callback):
        payloads.append(payload)
        return responses[len(payloads) - 1]

    monkeypatch.setattr(DesyAssistantBackend, "_stream_response", fake_stream)

    finals: list[str] = []
    response = runtime.desy_response(
        session.path,
        "desy-assistant",
        RuntimeCallbacks(finish=finals.append),
    )

    # The truncated first message must not end the turn; the loop continues and
    # the delivered text stitches the truncated prefix onto the continuation.
    assert len(payloads) == 2
    assert response == "part onepart two"
    assert finals == ["part onepart two"]
    second_messages = payloads[1]["messages"]
    assert any(
        isinstance(message.get("content"), str)
        and "truncated by the output token limit" in message["content"]
        for message in second_messages
    )


def test_desy_loop_stops_continuing_after_max_tokens_cap(tmp_path, monkeypatch):
    from anomx.agent.backends.desy_assistant import DesyAssistantBackend
    from anomx.agent.base.backends import MAX_TOKENS_CONTINUATIONS

    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    home.set_api_key("desy", "sk-desy-test")
    session = home.create_session(repo, provider="desy", model="desy-assistant")
    runtime = AgentRuntime(home, repo)
    call_count = {"n": 0}

    def fake_stream(_self, _api_key, _payload, _delta_callback, _status_callback):
        call_count["n"] += 1
        return runtime_module.AnthropicStreamResponse(
            text="still going",
            tool_calls=(),
            content=({"type": "text", "text": "still going"},),
            stop_reason="max_tokens",
        )

    monkeypatch.setattr(DesyAssistantBackend, "_stream_response", fake_stream)

    response = runtime.desy_response(session.path, "desy-assistant")

    # Continue up to the cap, then deliver the accumulated text instead of
    # looping forever.
    assert call_count["n"] == MAX_TOKENS_CONTINUATIONS + 1
    assert response == "still going" * (MAX_TOKENS_CONTINUATIONS + 1)


def test_desy_stream_uses_messages_endpoint_and_api_key_header(tmp_path, monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def __iter__(self):
            yield (
                b'data: {"type":"content_block_start","index":0,'
                b'"content_block":{"type":"text","text":""}}\n'
            )
            yield (
                b'data: {"type":"content_block_delta","index":0,'
                b'"delta":{"type":"text_delta","text":"OK"}}\n'
            )
            yield b'data: {"type":"content_block_stop","index":0}\n'

    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path)
    captured_request: dict[str, object] = {}

    def fake_urlopen(request, timeout=120):
        captured_request["url"] = request.full_url
        captured_request["headers"] = dict(request.header_items())
        return FakeResponse()

    monkeypatch.setattr(runtime_module.urllib.request, "urlopen", fake_urlopen)

    response = runtime._stream_desy_response(
        "sk-desy-test",
        {"model": "desy-assistant", "messages": [], "stream": True},
        None,
        None,
    )

    assert response.text == "OK"
    assert captured_request["url"] == "https://assistant.desy.de/api/v1/messages"
    assert captured_request["headers"]["X-api-key"] == "sk-desy-test"


def test_desy_stream_retries_transient_404_before_success(tmp_path, monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def __iter__(self):
            yield (
                b'data: {"type":"content_block_start","index":0,'
                b'"content_block":{"type":"text","text":""}}\n'
            )
            yield (
                b'data: {"type":"content_block_delta","index":0,'
                b'"delta":{"type":"text_delta","text":"OK"}}\n'
            )
            yield b'data: {"type":"content_block_stop","index":0}\n'

    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path)
    attempts = 0
    statuses: list[str] = []

    def fake_urlopen(request, timeout=120):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise HTTPError(
                request.full_url,
                404,
                "Not Found",
                hdrs=None,
                fp=io.BytesIO(b'{"detail":"route warming up"}'),
            )
        return FakeResponse()

    monkeypatch.setattr(runtime_module, "MODEL_REQUEST_RETRY_COUNT", 1)
    monkeypatch.setattr(runtime_module, "MODEL_REQUEST_RETRY_INITIAL_DELAY_SECONDS", 0.0)
    monkeypatch.setattr(runtime_module.urllib.request, "urlopen", fake_urlopen)

    response = runtime._stream_desy_response(
        "sk-desy-test",
        {"model": "desy-assistant", "messages": [], "stream": True},
        None,
        statuses.append,
    )

    assert isinstance(response, runtime_module.AnthropicStreamResponse)
    assert response.text == "OK"
    assert attempts == 2
    assert statuses == ["Reconnecting"]


def test_ollama_response_reports_loading_model_status(tmp_path, monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def __iter__(self):
            yield b'{"message":{"content":"hello"}}\n'

    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="ollama", model="qwen3.6")
    runtime = AgentRuntime(home, repo)
    statuses: list[str] = []
    deltas: list[str] = []

    monkeypatch.setattr(
        runtime_module.urllib.request,
        "urlopen",
        lambda *args, **kwargs: FakeResponse(),
    )

    response = runtime.ollama_response(
        session.path,
            "qwen3.6",
            RuntimeCallbacks(status=statuses.append, delta=deltas.append),
        )

    assert statuses == ["Loading model", "Thinking"]
    assert response == "hello"
    assert deltas == []


def test_ollama_response_executes_streamed_tool_calls(tmp_path, monkeypatch):
    class FakeResponse:
        def __init__(self, lines):
            self._lines = lines

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def __iter__(self):
            return iter(self._lines)

    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="ollama", model="qwen3.6")
    runtime = AgentRuntime(home, repo)
    captured_payloads: list[dict[str, object]] = []
    executed: list[tuple[str, dict[str, object]]] = []

    responses = iter(
        [
            FakeResponse(
                [
                    (
                        b'{"message":{"thinking":"Looking for files","tool_calls":['
                        b'{"function":{"name":"run_command","arguments":'
                        b'{"statement":"List files","command":"ls"}}}]}}\n'
                    )
                ]
            ),
            FakeResponse([b'{"message":{"content":"done"}}\n']),
        ]
    )

    def fake_urlopen(request, timeout=120):
        captured_payloads.append(json.loads(request.data.decode("utf-8")))
        return next(responses)

    def fake_execute_tool(name, arguments, callbacks, session_path=None):
        executed.append((name, arguments))
        return json.dumps({"approved": True, "output": "README.md"})

    monkeypatch.setattr(runtime_module.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(runtime, "_execute_tool", fake_execute_tool)

    response = runtime.ollama_response(session.path, "qwen3.6")

    assert response == "done"
    assert executed == [("run_command", {"statement": "List files", "command": "ls"})]
    assert captured_payloads[0]["tools"]
    assert captured_payloads[0]["think"] is True
    second_messages = captured_payloads[1]["messages"]
    assert second_messages[-2]["role"] == "assistant"
    assert second_messages[-2]["tool_calls"][0]["function"]["name"] == "run_command"
    assert second_messages[-1] == {
        "role": "tool",
        "tool_name": "run_command",
        "content": '{"approved": true, "output": "README.md"}',
    }


def test_runtime_status_events_keep_runtime_text(tmp_path):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))
    home = AnomxHome(tmp_path / "home")
    session = home.create_session(tmp_path, provider="ollama", model="qwen3.6")
    events: queue.SimpleQueue[RuntimeUiEvent] = queue.SimpleQueue()
    events.put(RuntimeUiEvent("status", "Loading model"))

    working_text, working_deadline, final_text, work_count = app._process_runtime_events(
        object(),
        session,
        events,
        "Thinking",
        None,
        "",
        "turn-1",
        0,
        None,
    )

    assert working_text == "Loading model"
    assert working_deadline is None
    assert final_text == ""
    assert work_count == 0


def test_runtime_status_events_keep_thinking_live_only(tmp_path):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))
    home = AnomxHome(tmp_path / "home")
    session = home.create_session(tmp_path, provider="ollama", model="qwen3.6")
    events: queue.SimpleQueue[RuntimeUiEvent] = queue.SimpleQueue()
    events.put(RuntimeUiEvent("status", "Thinking"))

    working_text, working_deadline, final_text, work_count = app._process_runtime_events(
        object(),
        session,
        events,
        None,
        None,
        "",
        "turn-1",
        0,
        None,
    )

    assert working_text == "Thinking"
    assert working_deadline is None
    assert final_text == ""
    assert work_count == 0


def test_concrete_status_events_persist_as_work_statements(tmp_path):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))
    home = AnomxHome(tmp_path / "home")
    session = home.create_session(tmp_path, provider="ollama", model="qwen3.6")
    events: queue.SimpleQueue[RuntimeUiEvent] = queue.SimpleQueue()
    events.put(RuntimeUiEvent("status", "Checking README"))
    events.put(RuntimeUiEvent("status", "Checking package.json"))

    working_text, working_deadline, final_text, work_count = app._process_runtime_events(
        object(),
        session,
        events,
        "Thinking",
        None,
        "",
        "turn-1",
        0,
        None,
    )

    assert working_text == "Checking package.json"
    assert working_deadline is None
    assert final_text == ""
    assert work_count == 2
    lines = app._read_message_lines(session.path)
    assert lines == [
        MessageLine("tool", "Checking README", "turn-1"),
        MessageLine("tool", "Checking package.json", "turn-1"),
    ]
    assert app._render_messages(lines, 80) == lines


def test_wait_returns_immediately_without_running_workers(tmp_path):
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path)
    messages: list[str] = []
    statuses: list[str] = []

    output = runtime._wait_tool(
        {},
        RuntimeCallbacks(tool_message=messages.append, status=statuses.append),
    )

    payload = json.loads(output)
    assert payload["waited_seconds"] == 0.0
    assert messages == []
    assert statuses == []


def test_wait_defaults_to_sixty_seconds_and_interrupts_when_workers_finish(
    tmp_path,
    monkeypatch,
):
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path)
    statuses: list[str] = []
    calls = 0

    def running_workers():
        nonlocal calls
        calls += 1
        return ("running",) if calls == 1 else ()

    monkeypatch.setattr(runtime, "_running_worker_states", running_workers)

    output = runtime._wait_tool(
        {},
        RuntimeCallbacks(status=statuses.append),
    )

    payload = json.loads(output)
    assert 0.0 <= payload["waited_seconds"] < 1.0
    assert statuses == ["Waiting:60.0"]


def test_waiting_status_renders_animated_base_text(tmp_path):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))
    home = AnomxHome(tmp_path / "home")
    session = home.create_session(tmp_path, provider="ollama", model="qwen3.6")
    events: queue.SimpleQueue[RuntimeUiEvent] = queue.SimpleQueue()
    events.put(RuntimeUiEvent("status", "Waiting:65"))

    working_text, working_deadline, final_text, work_count = app._process_runtime_events(
        object(),
        session,
        events,
        "Thinking",
        None,
        "",
        "turn-1",
        0,
        None,
    )

    assert working_text == "Waiting"
    assert working_deadline is not None
    assert app._working_status_text("Waiting", 165.0, now=100.0) == "Waiting"
    assert final_text == ""
    assert work_count == 0


def test_waiting_working_line_animates_dots(tmp_path):
    class Window:
        def __init__(self):
            self.writes = []

        def getmaxyx(self):
            return 20, 80

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)
    window = Window()

    app._draw_working_line(window, 3, 4, "Waiting", 40, frame=0)
    app._draw_working_line(window, 4, 4, "Waiting", 40, frame=4)
    app._draw_working_line(window, 5, 4, "Waiting", 40, frame=8)
    app._draw_working_line(window, 6, 4, "Thinking", 40, frame=12)
    app._draw_working_line(
        window,
        7,
        4,
        "waiting for long-running command",
        40,
        frame=12,
    )

    assert window.writes[0][2] == "Waiting."
    assert window.writes[1][2] == "Waiting.."
    assert window.writes[2][2] == "Waiting..."
    assert window.writes[3][2] == "Thinking..."
    assert window.writes[4][2] == "waiting for long-running command..."


def test_platform_connect_loading_renders_connecting_status(tmp_path):
    class Window:
        def __init__(self):
            self.writes = []

        def erase(self):
            pass

        def getmaxyx(self):
            return 32, 100

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

        def refresh(self):
            pass

    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)
    window = Window()

    app._draw_platform_connect_loading(window, frame=12)

    assert any(text == "Connect Platform" for _, _, text, _ in window.writes)
    assert any(
        y == 18 and x == 4 and text == "Connecting..." and attr == 0
        for y, x, text, attr in window.writes
    )


def test_platform_connection_form_highlights_selected_row(tmp_path):
    class Window:
        def __init__(self):
            self.writes = []
            self.cursor = None

        def erase(self):
            pass

        def getmaxyx(self):
            return 32, 100

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

        def move(self, y, x):
            self.cursor = (y, x)

        def refresh(self):
            pass

    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)
    app._colors = {"accent": 10, "light": 20, "bold": 30, "selected": 40, "ok": 50}
    window = Window()

    app._draw_platform_connection_form(
        window,
        PlatformConnectionDraft(
            url="anomalies.desy.de",
            email="theo.rieken@desy.de",
            password="secret",
        ),
        selected=2,
    )

    assert any(
        x == 4 and text == "Domain:" and attr == 20
        for _, x, text, attr in window.writes
    )
    assert any(
        x == 4 and text == "Email:" and attr == 20
        for _, x, text, attr in window.writes
    )
    assert any(
        x == 4 and text == "Password:" and attr == 10
        for _, x, text, attr in window.writes
    )
    assert any(
        x == 16 and text == "anomalies.desy.de" and attr == 0
        for _, x, text, attr in window.writes
    )
    assert any(
        x == 16 and text == "theo.rieken@desy.de" and attr == 0
        for _, x, text, attr in window.writes
    )
    assert any(
        x == 16 and text == "******" and attr == 0
        for _, x, text, attr in window.writes
    )
    assert any(
        text == "Esc Cancel · ↑↓ Navigate · Enter for Login"
        for _, _, text, _ in window.writes
    )

    window.writes.clear()
    app._draw_platform_connection_form(
        window,
        PlatformConnectionDraft(
            url="anomalies.desy.de",
            email="theo.rieken@desy.de",
            password="secret",
        ),
        selected=2,
        status="Connection alive.",
        status_role="ok",
    )

    assert any(
        x == 4 and text == "Connection alive." and attr == 50
        for _, x, text, attr in window.writes
    )

    window.writes.clear()
    app._draw_platform_connection_form(
        window,
        PlatformConnectionDraft(
            url="anomalies.desy.de",
            email="theo.rieken@desy.de",
            password="*****",
        ),
        selected=2,
        status="Checking connection...",
        editable=False,
    )

    assert any(text == "Password:" and attr == 20 for _, _, text, attr in window.writes)
    assert any(
        x == 4 and text == "Checking connection..." and attr == 0
        for _, x, text, attr in window.writes
    )


def test_platform_connection_form_returns_to_password_after_login_error(
    tmp_path,
    monkeypatch,
):
    class Window:
        def __init__(self):
            self.writes = []
            self._keys = iter((*"wrong-password", "\n", "\x1b"))

        def erase(self):
            pass

        def getmaxyx(self):
            return 32, 100

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

        def get_wch(self):
            return next(self._keys)

        def move(self, y, x):
            pass

        def refresh(self):
            pass

    home = AnomxHome(tmp_path / "home")
    home.set_platform_form_defaults(
        url="anomalies.desy.de",
        email="theo.rieken@desy.de",
    )
    app = AnomxCliApp(home=home, use_color=False)
    app._colors = {"accent": 10, "light": 20, "bold": 30, "danger": 40, "selected": 50}
    monkeypatch.setattr(
        app,
        "_connect_platform_with_loading",
        lambda *_args: (_ for _ in ()).throw(
            platform_client_module.PlatformClientError("Invalid credentials.")
        ),
    )
    window = Window()

    assert app._run_platform_connection_form(window) is None
    assert any(
        x == 4 and text == "Invalid credentials." and attr == 40
        for _, x, text, attr in window.writes
    )
    assert any(text == "Password:" and attr == 10 for _, _, text, attr in window.writes)
    config = home.load_config()
    assert config["platform_last_url"] == "anomalies.desy.de"
    assert config["platform_last_email"] == "theo.rieken@desy.de"


def test_manage_platform_logout_requires_confirmation(tmp_path, monkeypatch):
    class Window:
        def __init__(self):
            self.writes = []
            self._keys = iter(("\x04", "\x1b", "\x04", "\n", "\n"))

        def erase(self):
            pass

        def getmaxyx(self):
            return 32, 100

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

        def get_wch(self):
            return next(self._keys)

        def move(self, y, x):
            pass

        def refresh(self):
            pass

        def nodelay(self, flag):
            pass

    home = AnomxHome(tmp_path / "home")
    home.set_platform_connection(
        url="https://anomalies.msktools.desy.de/api",
        token="platform-token",
        user_email="theo.rieken@desy.de",
    )
    app = AnomxCliApp(home=home, use_color=False)
    monkeypatch.setattr(platform_client_module, "heartbeat_platform_connection", lambda _home: True)
    window = Window()

    assert (
        app._run_platform_management_form(
            window,
            home.platform_connection(),
            initial_status="Connection alive.",
            initial_status_role="ok",
            check_connection=False,
        )
        is True
    )

    assert any(text == "Logout this CLI agent?" for _, _, text, _ in window.writes)
    assert home.platform_connection() is None


def test_approval_events_persist_command_decision(tmp_path, monkeypatch):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))
    home = AnomxHome(tmp_path / "home")
    session = home.create_session(tmp_path, provider="ollama", model="qwen3.6")
    response_queue: queue.SimpleQueue[ApprovalChoice] = queue.SimpleQueue()
    events: queue.SimpleQueue[RuntimeUiEvent] = queue.SimpleQueue()
    events.put(
        RuntimeUiEvent(
            "approval",
            approval_request=CommandApprovalRequest(
                command="cat README.md",
                statement="Read project overview",
                reason="Needs approval",
                canonical_command="cat README.md",
                allowance_subject="cat",
            ),
            approval_response=response_queue,
        )
    )
    captured: dict[str, object] = {}

    def fake_request_command_approval(*_args, **kwargs):
        captured.update(kwargs)
        return ApprovalChoice.ALLOW

    monkeypatch.setattr(app, "_request_command_approval", fake_request_command_approval)

    working_text, working_deadline, final_text, work_count = app._process_runtime_events(
        object(),
        session,
        events,
        "Thinking",
        None,
        "",
        "turn-1",
        0,
        11,
        scroll=4,
    )

    assert response_queue.get_nowait() == ApprovalChoice.ALLOW
    assert working_text == "Thinking"
    assert working_deadline is None
    assert final_text == ""
    assert work_count == 0
    assert captured == {"scroll": 4, "anchor_line": 11}
    assert app._read_message_lines(session.path) == []


def test_rejected_approval_persists_blocked_statement(tmp_path, monkeypatch):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))
    home = AnomxHome(tmp_path / "home")
    session = home.create_session(tmp_path, provider="ollama", model="qwen3.6")
    response_queue: queue.SimpleQueue[ApprovalChoice] = queue.SimpleQueue()
    events: queue.SimpleQueue[RuntimeUiEvent] = queue.SimpleQueue()
    events.put(
        RuntimeUiEvent(
            "approval",
            approval_request=CommandApprovalRequest(
                command="rm README.md",
                statement="Removing generated README",
                reason="Needs approval",
                canonical_command="rm README.md",
                allowance_subject="rm",
            ),
            approval_response=response_queue,
        )
    )

    monkeypatch.setattr(
        app,
        "_request_command_approval",
        lambda *_args, **_kwargs: ApprovalChoice.REJECT,
    )

    *_unused, work_count = app._process_runtime_events(
        object(),
        session,
        events,
        "Thinking",
        None,
        "",
        "turn-1",
        0,
        None,
    )

    assert response_queue.get_nowait() == ApprovalChoice.REJECT
    assert work_count == 1
    lines = app._read_message_lines(session.path)
    assert lines == [
        MessageLine(
            "forbidden",
            "Blocked: Removing generated README",
            "turn-1",
            detail_title="Reason: Rejected by user.",
            detail_body="rm README.md",
        )
    ]
    assert lines[0].detail_title == "Reason: Rejected by user."
    assert lines[0].detail_body == "rm README.md"


def test_question_events_return_answer_and_persist_hidden_context(tmp_path, monkeypatch):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))
    home = AnomxHome(tmp_path / "home")
    session = home.create_session(tmp_path, provider="ollama", model="qwen3.6")
    response_queue: queue.SimpleQueue[QuestionResponse] = queue.SimpleQueue()
    question = QuestionRequest(
        question="Which stack should I use?",
        kind="select",
    )
    events: queue.SimpleQueue[RuntimeUiEvent] = queue.SimpleQueue()
    events.put(
        RuntimeUiEvent(
            "question",
            question_request=question,
            question_response=response_queue,
        )
    )
    captured: dict[str, object] = {}

    def fake_request_question(*_args, **kwargs):
        captured.update(kwargs)
        return QuestionResponse(
            answered=True,
            answer="next",
            selected_label="Next.js",
            kind="select",
        )

    monkeypatch.setattr(app, "_request_question", fake_request_question)

    working_text, working_deadline, final_text, work_count = app._process_runtime_events(
        object(),
        session,
        events,
        "Thinking",
        None,
        "",
        "turn-1",
        0,
        11,
        scroll=4,
    )

    answer = response_queue.get_nowait()
    raw_events = home.read_session_events(session.path)

    assert answer.answer == "next"
    assert working_text == "Thinking"
    assert working_deadline is None
    assert final_text == ""
    assert work_count == 0
    assert captured == {"scroll": 4, "anchor_line": 11}
    assert app._read_message_lines(session.path) == []
    assert any(
        event_payload_type(event) == "system_message"
        and event_payload(event).get("role") == "question"
        and "Which stack should I use?" in str(event_payload(event).get("message"))
        for event in raw_events
    )


def test_output_message_events_persist_as_agent_messages(tmp_path, monkeypatch):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))
    home = AnomxHome(tmp_path / "home")
    session = home.create_session(tmp_path, provider="ollama", model="qwen3.6")
    events: queue.SimpleQueue[RuntimeUiEvent] = queue.SimpleQueue()
    events.put(RuntimeUiEvent("message", "I am checking the repository."))
    monkeypatch.setattr(app, "_fake_type_message", lambda *args, **kwargs: None)

    working_text, working_deadline, final_text, work_count = app._process_runtime_events(
        object(),
        session,
        events,
        "Thinking",
        None,
        "",
        "turn-1",
        0,
        None,
    )

    assert working_text is None
    assert working_deadline is None
    assert final_text == ""
    assert work_count == 1
    assert app._read_message_lines(session.path) == [
        MessageLine("agent_intermediate", "I am checking the repository.", "turn-1")
    ]


def test_intermediate_message_clears_streamed_final_buffer(tmp_path, monkeypatch):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))
    home = AnomxHome(tmp_path / "home")
    session = home.create_session(tmp_path, provider="ollama", model="qwen3.6")
    events: queue.SimpleQueue[RuntimeUiEvent] = queue.SimpleQueue()
    events.put(RuntimeUiEvent("delta", "This looked final."))
    events.put(RuntimeUiEvent("message", "This is actually a progress update."))
    monkeypatch.setattr(app, "_fake_type_message", lambda *args, **kwargs: None)

    working_text, working_deadline, final_text, work_count = app._process_runtime_events(
        object(),
        session,
        events,
        "Thinking",
        None,
        "",
        "turn-1",
        0,
        None,
    )

    assert working_text is None
    assert working_deadline is None
    assert final_text == ""
    assert work_count == 1
    assert app._read_message_lines(session.path) == [
        MessageLine("agent_intermediate", "This is actually a progress update.", "turn-1")
    ]


def test_tool_message_events_persist_as_work_messages(tmp_path, monkeypatch):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))
    home = AnomxHome(tmp_path / "home")
    session = home.create_session(tmp_path, provider="ollama", model="qwen3.6")
    events: queue.SimpleQueue[RuntimeUiEvent] = queue.SimpleQueue()
    events.put(RuntimeUiEvent("tool_message", "Starting Engineer Worker"))
    monkeypatch.setattr(app, "_fake_type_message", lambda *args, **kwargs: None)

    working_text, working_deadline, final_text, work_count = app._process_runtime_events(
        object(),
        session,
        events,
        "Thinking",
        None,
        "",
        "turn-1",
        0,
        None,
    )

    assert working_text is None
    assert working_deadline is None
    assert final_text == ""
    assert work_count == 1
    assert app._read_message_lines(session.path) == [
        MessageLine("tool", "Starting Engineer Worker", "turn-1")
    ]


def test_command_events_persist_statement_with_hidden_command(tmp_path, monkeypatch):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))
    home = AnomxHome(tmp_path / "home")
    session = home.create_session(tmp_path, provider="ollama", model="qwen3.6")
    events: queue.SimpleQueue[RuntimeUiEvent] = queue.SimpleQueue()
    events.put(RuntimeUiEvent("command", "Reading project overview", command="cat README.md"))
    monkeypatch.setattr(app, "_fake_type_message", lambda *args, **kwargs: None)

    *_unused, work_count = app._process_runtime_events(
        object(),
        session,
        events,
        "Thinking",
        None,
        "",
        "turn-1",
        0,
        None,
    )

    lines = app._read_message_lines(session.path)

    assert work_count == 1
    assert lines == [
        MessageLine(
            "tool",
            "Reading project overview",
            "turn-1",
            detail_body="cat README.md",
        )
    ]
    assert lines[0].detail_body == "cat README.md"
    rendered = app._render_messages(lines, 80)
    assert any(line.text == "Reading project overview" for line in rendered)
    assert not any("cat README.md" in line.text for line in rendered)
    app._toggle_work_line(lines[0].expansion_key)
    expanded = app._render_messages(lines, 80)
    assert any("cat README.md" in line.text for line in expanded)


def test_operator_tool_statements_include_hidden_tool_overview(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    runtime = AgentRuntime(home, repo)
    records: list[tuple[str, str, str]] = []

    def record_command(statement: str, command: str, output: str) -> None:
        records.append((statement, command, output))

    runtime._execute_tool(
        "create_plan",
        {
            "statement": "Planning the doodle-like polling app",
            "steps": [
                {
                    "title": "Scaffold project",
                    "description": "Create the starter app.",
                    "is_done": False,
                }
            ],
        },
        RuntimeCallbacks(command=record_command),
        session.path,
    )

    assert len(records) == 1
    statement, detail_body, output = records[0]
    assert statement == "Planning the doodle-like polling app"
    assert "Tool: create_plan" in detail_body
    assert "Parameters:" in detail_body
    assert '"steps"' in detail_body
    assert "statement" not in detail_body
    assert output == ""


def test_wait_schema_is_openai_strict_compatible(tmp_path):
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path, role=AgentRole.OPERATOR)
    runtime._running_worker_states = lambda: ("running",)  # type: ignore[method-assign]
    wait_tool = next(tool for tool in runtime._openai_tools() if tool["name"] == "wait")

    assert wait_tool["parameters"]["properties"] == {}
    assert wait_tool["parameters"]["required"] == []


def test_markdown_is_rendered_as_terminal_lines():
    lines = markdown_to_terminal_lines(
        "## Finding\n\n- **Signal** changed in `sensor_a`\n```text\nraw value\n```",
        width=80,
    )

    assert lines == ["Finding", "", "- Signal changed in sensor_a", "  raw value"]


def test_markdown_pipe_tables_render_as_aligned_terminal_tables():
    rendered = markdown_to_terminal_rendered_lines(
        "\n".join(
            [
                "Data channels",
                "",
                "| Channel | Description |",
                "| --- | --- |",
                "| LOCK_STATUS.VALUE.RD | Lock status |",
                "| RF_FUND_MAG.SPEC | RF magnitude spectrum |",
            ]
        ),
        width=80,
    )
    texts = [line.text for line in rendered]
    table_lines = texts[2:]

    assert texts[:2] == ["Data channels", ""]
    assert table_lines[0].startswith("┌")
    assert table_lines[-1].startswith("└")
    assert "Channel" in table_lines[1]
    assert "Description" in table_lines[1]
    assert rendered[2].style == "table_border"
    assert rendered[3].style == "table_header"
    assert rendered[5].style == "table_row"
    assert len({len(line) for line in table_lines}) == 1
    assert not any("---" in line for line in table_lines)


def test_markdown_pipe_tables_wrap_to_terminal_width():
    lines = markdown_to_terminal_lines(
        "\n".join(
            [
                "| Channel | Description |",
                "| --- | --- |",
                "| VERY_LONG_SIGNAL_IDENTIFIER_WITHOUT_SPACES | "
                "A long description that should wrap cleanly inside the cell |",
            ]
        ),
        width=42,
    )

    assert all(len(line) <= 42 for line in lines)
    assert any("VERY_LONG_SIGNAL" in line for line in lines)
    assert any("cleanly inside" in line for line in lines)


def test_table_row_sides_use_border_attribute(tmp_path):
    class Window:
        def __init__(self):
            self.calls = []

        def getmaxyx(self):
            return 5, 80

        def addnstr(self, y, x, text, n, attr=0):
            self.calls.append((y, x, text, n, attr))

    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)
    app._colors = {
        "table_border": 10,
        "table_header": 20,
    }

    header_window = Window()
    app._draw_table_line(header_window, 0, 0, "│ Header │", 80, "table_header")

    assert [call[2] for call in header_window.calls] == ["│", " Header ", "│"]
    assert [call[4] for call in header_window.calls] == [10, 20, 10]

    body_window = Window()
    app._draw_table_line(body_window, 0, 0, "│ Value │", 80, "table_row")

    assert [call[2] for call in body_window.calls] == ["│", " Value ", "│"]
    assert [call[4] for call in body_window.calls] == [10, curses.A_NORMAL, 10]


def test_transcript_rendering_does_not_prefix_roles(tmp_path):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))

    rendered = app._render_messages(
        [
            MessageLine("user", "Hi there"),
            MessageLine("agent", "**Hello** from Anomx"),
        ],
        width=80,
    )

    assert rendered[0].role == "user"
    assert rendered[0].text.startswith("Hi there")
    assert rendered[0].text.endswith(" Expand")
    assert rendered[2] == MessageLine("agent", "Hello from Anomx")


def test_file_reference_suggestions_find_workspace_files_and_folders(tmp_path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / "src" / "anomx" / "agent").mkdir(parents=True)
    (repo / "src" / "anomx" / "agent" / "ui.py").write_text("", encoding="utf-8")
    (repo / "src" / "anomx" / "agent" / "tools").mkdir()
    (repo / "src" / "anomx" / "agent" / "tools" / "grep.py").write_text("", encoding="utf-8")
    (repo / "anomx-website" / "app" / "admin").mkdir(parents=True)
    (repo / "anomx-website" / "app" / "page.tsx").write_text("", encoding="utf-8")
    (repo / "node_modules").mkdir()
    (repo / "node_modules" / "ui.js").write_text("", encoding="utf-8")
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), cwd=repo)
    app._refresh_file_reference_index()

    assert app._active_file_reference_token("Read @ui", 8) == (5, 8, "ui")

    choices = app._filtered_file_references("ui")

    assert choices == [
        MenuChoice(
            "src/anomx/agent/ui.py",
            "src/anomx/agent/ui.py",
            "",
            "ui",
            ((16, 18),),
        ),
    ]
    folder_choices = app._filtered_file_references("agent")

    assert folder_choices[0] == MenuChoice(
        "src/anomx/agent/",
        "src/anomx/agent/",
        "",
        "agent",
        ((10, 15),),
    )

    nested_choices = app._filtered_file_references("anomx/agent")

    assert nested_choices[0] == MenuChoice(
        "src/anomx/agent/",
        "src/anomx/agent/",
        "",
        "anomx/agent",
        ((4, 15),),
    )
    assert all(choice.label != "src/anomx/agent/ui.py" for choice in nested_choices)

    assert app._filtered_file_references("anomy/agent") == []

    website_choices = app._filtered_file_references("anomx-website/ap")

    assert website_choices[0] == MenuChoice(
        "anomx-website/app/",
        "anomx-website/app/",
        "",
        "anomx-website/ap",
        ((0, 16),),
    )

    child_choices = app._filtered_file_references("anomx-website/app/")

    assert [choice.label for choice in child_choices] == [
        "anomx-website/app/admin/",
        "anomx-website/app/page.tsx",
    ]
    assert child_choices[0].highlight_spans == ((0, 18),)


def test_file_reference_index_persists_and_returns_ten_ranked_results(tmp_path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / "deep").mkdir()
    (repo / "target.py").write_text("", encoding="utf-8")
    for index in range(12):
        (repo / "deep" / f"alpha-target-{index}.py").write_text("", encoding="utf-8")
    home = AnomxHome(tmp_path / "home")
    app = AnomxCliApp(home=home, cwd=repo)

    app._refresh_file_reference_index()

    cache_path = app._file_reference_index_path()
    assert cache_path.parent == home.search_dir
    assert cache_path.exists()

    choices = app._filtered_file_references("target")

    assert len(choices) == 10
    assert choices[0].label == "target.py"
    assert any(choice.label.startswith("deep/alpha-target-") for choice in choices[1:])


def test_bottom_panel_choice_highlights_file_reference_match(tmp_path):
    class Window:
        def __init__(self):
            self.writes = []

        def getmaxyx(self):
            return 20, 80

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)
    app._colors = {
        "bold": 10,
        "accent": 20,
        "selected": 30,
    }
    window = Window()

    app._draw_bottom_panel_choice_label(
        window,
        4,
        2,
        "› src/anomx/agent/ui.py",
        40,
        app._attr("bold"),
        "",
        True,
        ((4, 9), (10, 15)),
        label_offset=2,
        highlight_attr="accent",
        selected_highlight_attr="accent",
    )

    assert ("› src/", 10) in [(text, attr) for _, _, text, attr in window.writes]
    assert ("anomx", 20) in [(text, attr) for _, _, text, attr in window.writes]
    assert ("agent", 20) in [(text, attr) for _, _, text, attr in window.writes]
    assert all(attr != 30 for _, _, _text, attr in window.writes)


def test_file_reference_insert_and_backend_message(tmp_path):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))
    file_references: dict[str, str] = {}

    input_text, cursor = app._insert_file_reference(
        "Read @ui now",
        8,
        (5, 8, "ui"),
        MenuChoice("src/anomx/agent/ui.py", "src/anomx/agent/ui.py", "", "ui"),
        file_references,
    )

    assert input_text == "Read src/anomx/agent/ui.py now"
    assert cursor == len("Read src/anomx/agent/ui.py")
    assert file_references == {"src/anomx/agent/ui.py": "src/anomx/agent/ui.py"}
    assert (
        app._backend_message_for_prompt(input_text, file_references)
        == "Read src/anomx/agent/ui.py now"
    )
    assert (
        app._backend_message_for_prompt("Read myui.py too", file_references)
        == "Read myui.py too"
    )

    input_text, cursor = app._insert_file_reference(
        "Inspect @agent",
        len("Inspect @agent"),
        (8, len("Inspect @agent"), "agent"),
        MenuChoice("src/anomx/agent/", "src/anomx/agent/", "", "agent"),
        file_references,
    )

    assert input_text == "Inspect src/anomx/agent/ "
    assert cursor == len("Inspect src/anomx/agent/ ")
    assert file_references["src/anomx/agent/"] == "src/anomx/agent/"
    assert (
        app._backend_message_for_prompt(input_text.strip(), file_references)
        == "Inspect src/anomx/agent/"
    )


def test_file_references_are_visible_in_thread_but_expanded_for_runtime(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    home.append_session_event(
        session.path,
        "user_message",
        {
            "message": "Read ui.py",
            "backend_message": "Read ui.py [src/anomx/agent/ui.py]",
            "file_references": {"ui.py": "src/anomx/agent/ui.py"},
        },
    )
    app = AnomxCliApp(home=home, cwd=repo)
    runtime = AgentRuntime(home, repo)

    assert app._read_message_lines(session.path) == [MessageLine("user", "Read ui.py")]
    assert runtime.conversation_messages(session.path) == [
        {"role": "user", "content": "Read ui.py [src/anomx/agent/ui.py]"},
    ]


def test_dropped_image_path_is_tokenized_and_attached(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    image = repo / "plot image.png"
    image.write_bytes(b"png-bytes")
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), cwd=repo)
    attachments: dict[str, dict[str, str]] = {}
    input_text = f"Explain {image}"

    updated, cursor, added = app._consume_dropped_images(
        input_text,
        len(input_text),
        attachments,
    )

    assert updated == "Explain [image: plot image.png] "
    assert cursor == len(updated)
    assert tuple(attachments) == ("[image: plot image.png]",)
    assert added == (
        {
            "label": "plot image.png",
            "token": "[image: plot image.png]",
            "path": image.resolve().as_posix(),
            "mime_type": "image/png",
        },
    )
    assert app._active_image_attachments(updated, attachments) == attachments


def test_dropped_file_url_image_is_tokenized(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    image = repo / "trend.webp"
    image.write_bytes(b"webp-bytes")
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), cwd=repo)
    attachments: dict[str, dict[str, str]] = {}
    input_text = f"Inspect file://{image}"

    updated, _cursor, added = app._consume_dropped_images(
        input_text,
        len(input_text),
        attachments,
    )

    assert updated == "Inspect [image: trend.webp] "
    assert added[0]["mime_type"] == "image/webp"


def test_dropped_image_warns_when_backend_does_not_support_images(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="desy", model="coding")
    config = home.load_config()
    config.update({"provider": "desy", "model": "coding"})
    home.save_config(config)
    app = AnomxCliApp(home=home, cwd=repo)

    app._append_unsupported_image_notice(
        session,
        (
            {
                "label": "plot.png",
                "token": "[image: plot.png]",
                "path": (repo / "plot.png").as_posix(),
                "mime_type": "image/png",
            },
        ),
    )

    lines = app._read_message_lines(session.path)
    assert lines == [
        MessageLine(
            "warning",
            (
                "DESY Assistant/coding does not support image input. Dropped image "
                "attachments will be kept in the prompt but sent to this backend as "
                "text only: plot.png."
            ),
        )
    ]


def test_image_attachments_are_converted_for_supported_backends(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    image = repo / "plot.png"
    image_bytes = b"png-bytes"
    image.write_bytes(image_bytes)
    encoded = base64.b64encode(image_bytes).decode("ascii")
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    payload = {
        "label": "plot.png",
        "token": "[image: plot.png]",
        "path": image.as_posix(),
        "mime_type": "image/png",
    }
    home.append_session_event(
        session.path,
        "user_message",
        {
            "message": "Explain [image: plot.png]",
            "backend_message": "Explain [image: plot.png]",
            "image_attachments": [payload],
        },
    )
    runtime = AgentRuntime(home, repo)
    messages = runtime.conversation_messages(session.path)

    assert messages == [
        {
            "role": "user",
            "content": "Explain [image: plot.png]",
            "images": [payload],
        },
    ]

    openai_message = runtime._openai_messages(messages, "gpt-5.5")[0]
    assert openai_message["content"][0] == {
        "type": "input_text",
        "text": "Explain [image: plot.png]\n\nAttached images:\n- plot.png",
    }
    assert openai_message["content"][1] == {
        "type": "input_image",
        "image_url": f"data:image/png;base64,{encoded}",
    }

    anthropic_message = runtime._anthropic_messages(
        messages,
        "anthropic",
        "claude-sonnet-4-6",
    )[0]
    assert anthropic_message["content"][1] == {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": encoded,
        },
    }

    ollama_message = runtime._ollama_messages(messages, "llava:latest")[0]
    assert ollama_message["images"] == [encoded]


def test_unsupported_backends_keep_image_tokens_as_text(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    image = repo / "plot.png"
    image.write_bytes(b"png-bytes")
    runtime = AgentRuntime(home, repo)
    messages = [
        {
            "role": "user",
            "content": "Explain [image: plot.png]",
            "images": [
                {
                    "label": "plot.png",
                    "token": "[image: plot.png]",
                    "path": image.as_posix(),
                    "mime_type": "image/png",
                },
            ],
        },
    ]

    assert not backend_supports_image_input("desy", "coding")
    assert not backend_supports_image_input("ollama", "qwen3.6")
    assert runtime._anthropic_messages(messages, "desy", "coding") == [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Explain [image: plot.png]"}],
        },
    ]
    assert runtime._ollama_messages(messages, "qwen3.6") == [
        {"role": "user", "content": "Explain [image: plot.png]"}
    ]


def test_work_messages_render_without_blank_gaps(tmp_path):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))

    rendered = app._render_messages(
        [
            MessageLine("tool", "Thinking..."),
            MessageLine("tool", "Checking repo..."),
            MessageLine("tool", "Started Engineer Worker"),
        ],
        width=80,
    )

    assert rendered == [
        MessageLine("tool", "Thinking..."),
        MessageLine("tool", "Checking repo..."),
        MessageLine("tool", "Started Engineer Worker"),
    ]


def test_long_work_messages_render_as_single_ellipsized_statement_line(tmp_path):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))
    message = "Writing generated type definitions " * 4

    rendered = app._render_messages(
        [MessageLine("tool", message, expansion_key="line-1")],
        width=56,
    )

    assert len(rendered) == 1
    assert rendered[0].role == "tool"
    assert rendered[0].expansion_key == "line-1"
    assert rendered[0].text.endswith("...")
    assert "click to expand" not in rendered[0].text
    assert len(rendered[0].text) <= 56


def test_short_work_messages_are_clickable_statement_rows(tmp_path):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))

    rendered = app._render_messages(
        [
            MessageLine(
                "tool",
                "Checking repository",
                expansion_key="line-1",
            ),
            MessageLine(
                "forbidden",
                "Blocked: Removing generated README",
                expansion_key="line-2",
                detail_title="Reason: Rejected by user.",
            ),
        ],
        width=120,
    )

    assert rendered == [
        MessageLine("tool", "Checking repository", expansion_key="line-1"),
        MessageLine(
            "forbidden",
            "Blocked: Removing generated README",
            expansion_key="line-2",
            detail_title="Reason: Rejected by user.",
        ),
    ]


def test_expanded_work_message_renders_inside_box(tmp_path):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))
    app._toggle_work_line("line-1")

    rendered = app._render_messages(
        [
            MessageLine(
                "tool",
                "Writing generated page",
                expansion_key="line-1",
                detail_body="printf 'hello'\ncp /tmp/page.tsx src/app/page.tsx",
            )
        ],
        width=48,
    )

    assert [line.role for line in rendered] == ["work_box"] * len(rendered)
    assert rendered[0].text.startswith("╭")
    assert rendered[-1].text.startswith("╰")
    assert "Writing generated page" in rendered[1].text
    assert rendered[2].text.strip("│ ") == ""
    assert any("printf 'hello'" in line.text for line in rendered)
    assert any("cp /tmp/page.tsx" in line.text for line in rendered)
    assert all(line.expansion_key == "line-1" for line in rendered)


def test_expanded_blocked_work_message_renders_reason_title(tmp_path):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))
    app._toggle_work_line("line-1")

    rendered = app._render_messages(
        [
            MessageLine(
                "forbidden",
                "Blocked: Reading outside workspace",
                expansion_key="line-1",
                detail_title="Reason: Path is outside the trusted workspace.",
                detail_body="cat ../outside.txt",
            )
        ],
        width=64,
    )

    assert [line.role for line in rendered] == ["work_box"] * len(rendered)
    assert "Blocked: Reading outside workspace" in rendered[1].text
    assert "Reason: Path is outside the trusted workspace." in rendered[2].text
    assert rendered[3].text.strip("│ ") == ""
    assert "cat ../outside.txt" in rendered[4].text
    assert not any(line.text.startswith("├") for line in rendered)


def test_rendering_inserts_blank_lines_when_message_kind_changes(tmp_path):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))

    rendered = app._render_messages(
        [
            MessageLine("user", "Build this"),
            MessageLine("tool", "Checking repo"),
            MessageLine("approved", "Approved command: cat README.md"),
            MessageLine("agent", "Done"),
        ],
        width=80,
    )

    assert rendered[0].role == "user"
    assert rendered[0].text.startswith("Build this")
    assert rendered[0].text.endswith(" Expand")
    assert rendered[1:] == [
        MessageLine("meta", ""),
        MessageLine("tool", "Checking repo"),
        MessageLine("approved", "Approved command: cat README.md"),
        MessageLine("meta", ""),
        MessageLine("agent", "Done"),
    ]


def test_intermediate_agent_messages_have_single_kind_change_gaps(tmp_path):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))

    rendered = app._render_messages(
        [
            MessageLine("tool", "Checking README"),
            MessageLine("agent_intermediate", "I found the README."),
            MessageLine("tool", "Checking package.json"),
        ],
        width=80,
    )

    assert rendered == [
        MessageLine("tool", "Checking README"),
        MessageLine("meta", ""),
        MessageLine("agent_intermediate", "I found the README."),
        MessageLine("meta", ""),
        MessageLine("tool", "Checking package.json"),
    ]


def test_live_working_status_gets_spacing_after_user_message(tmp_path):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))
    session = AnomxHome(tmp_path / "home").create_session(
        tmp_path,
        provider="ollama",
        model="qwen3.6",
    )

    rendered = app._session_rendered_lines(
        session,
        [MessageLine("user", "Checkout this repo")],
        width=80,
        working_text="Loading model",
    )

    assert rendered == [
        MessageLine(
            "user",
            "Checkout this repo                                                        Expand",
        ),
        MessageLine("meta", ""),
        MessageLine("working", "Loading model"),
        MessageLine("meta", ""),
    ]


def test_work_summary_collapses_turn_local_agent_and_work_messages(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    home.append_session_event(
        session.path,
        "agent_message",
        {"message": "I am checking the repository.", "turn_id": "turn-1"},
    )
    home.append_session_event(
        session.path,
        "work_message",
        {"message": "Reading README", "role": "tool", "turn_id": "turn-1"},
    )
    home.append_session_event(
        session.path,
        "work_summary",
        {"message": "Worked for 00:01 min", "turn_id": "turn-1"},
    )
    home.append_session_event(
        session.path,
        "agent_message",
        {"message": "Final response"},
    )
    app = AnomxCliApp(home=home, cwd=repo)

    assert app._read_message_lines(session.path) == [
        MessageLine("work_summary", "Worked for 00:01 min · expand", "turn-1"),
        MessageLine("agent", "Final response"),
    ]


def test_active_turn_keeps_statements_and_intermediate_messages_in_order(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    home.append_session_event(session.path, "user_message", {"message": "Inspect this repo"})
    home.append_session_event(
        session.path,
        "work_message",
        {"message": "Reading README", "role": "tool", "turn_id": "turn-1"},
    )
    home.append_session_event(
        session.path,
        "agent_message",
        {
            "message": "The README points to a poll app.",
            "turn_id": "turn-1",
            "intermediate": True,
        },
    )
    home.append_session_event(
        session.path,
        "work_message",
        {"message": "Reading package.json", "role": "tool", "turn_id": "turn-1"},
    )
    app = AnomxCliApp(home=home, cwd=repo)

    assert app._read_message_lines(session.path) == [
        MessageLine("user", "Inspect this repo"),
        MessageLine("tool", "Reading README", "turn-1"),
        MessageLine(
            "agent_intermediate",
            "The README points to a poll app.",
            "turn-1",
        ),
        MessageLine("tool", "Reading package.json", "turn-1"),
    ]


def test_work_summary_collapses_requeued_turn_to_original_prompt(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    home.append_session_event(session.path, "user_message", {"message": "What is this repo?"})
    home.append_session_event(
        session.path,
        "agent_message",
        {
            "message": "Let me explore the repository structure.",
            "turn_id": "turn-1",
            "intermediate": True,
        },
    )
    home.append_session_event(
        session.path,
        "work_message",
        {"message": "List root directory", "role": "tool", "turn_id": "turn-1"},
    )
    home.append_session_event(
        session.path,
        "user_message",
        {"message": "Lets go deep!", "turn_id": "turn-1", "intermediate": True},
    )
    home.append_session_event(
        session.path,
        "work_message",
        {"message": "Inspect package internals", "role": "tool", "turn_id": "turn-1"},
    )
    home.append_session_event(
        session.path,
        "work_summary",
        {"message": "Interrupted after 00:05", "turn_id": "turn-1"},
    )
    home.append_session_event(
        session.path,
        "agent_message",
        {"message": MANUAL_INTERRUPT_MESSAGE},
    )
    app = AnomxCliApp(home=home, cwd=repo)

    assert app._read_message_lines(session.path) == [
        MessageLine("user", "What is this repo?"),
        MessageLine("work_summary", "Interrupted after 00:05 · expand", "turn-1"),
        MessageLine("agent", MANUAL_INTERRUPT_MESSAGE),
    ]

    app._toggle_work_turn("turn-1")

    assert app._read_message_lines(session.path) == [
        MessageLine("user", "What is this repo?"),
        MessageLine(
            "agent_intermediate",
            "Let me explore the repository structure.",
            "turn-1",
        ),
        MessageLine("tool", "List root directory", "turn-1"),
        MessageLine("user", "Lets go deep!", "turn-1"),
        MessageLine("tool", "Inspect package internals", "turn-1"),
        MessageLine("work_summary", "Interrupted after 00:05 · collapse", "turn-1"),
        MessageLine("agent", MANUAL_INTERRUPT_MESSAGE),
    ]


def test_work_summary_absorbs_late_turn_local_messages(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    home.append_session_event(session.path, "user_message", {"message": "Inspect this repo"})
    home.append_session_event(
        session.path,
        "work_message",
        {"message": "Reading README", "role": "tool", "turn_id": "turn-1"},
    )
    home.append_session_event(
        session.path,
        "work_summary",
        {"message": "Worked for 00:02 min", "turn_id": "turn-1"},
    )
    home.append_session_event(
        session.path,
        "agent_message",
        {
            "message": "I found the main mismatch.",
            "turn_id": "turn-1",
            "intermediate": True,
        },
    )
    home.append_session_event(
        session.path,
        "work_message",
        {"message": "Reading route file", "role": "tool", "turn_id": "turn-1"},
    )
    home.append_session_event(
        session.path,
        "agent_message",
        {"message": "Final response"},
    )
    app = AnomxCliApp(home=home, cwd=repo)

    assert app._read_message_lines(session.path) == [
        MessageLine("user", "Inspect this repo"),
        MessageLine("work_summary", "Worked for 00:02 min · expand", "turn-1"),
        MessageLine("agent", "Final response"),
    ]

    app._toggle_work_turn("turn-1")

    assert app._read_message_lines(session.path) == [
        MessageLine("user", "Inspect this repo"),
        MessageLine("tool", "Reading README", "turn-1"),
        MessageLine("agent_intermediate", "I found the main mismatch.", "turn-1"),
        MessageLine("tool", "Reading route file", "turn-1"),
        MessageLine("work_summary", "Worked for 00:02 min · collapse", "turn-1"),
        MessageLine("agent", "Final response"),
    ]


def test_session_rendering_starts_with_messages_not_status_metadata(tmp_path):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"))
    session = AnomxHome(tmp_path / "home").create_session(
        tmp_path,
        provider="openai",
        model="gpt-5.5",
    )

    rendered = app._session_rendered_lines(
        session,
        [MessageLine("user", "Hi")],
        width=80,
    )

    assert rendered[0].role == "user"
    assert rendered[0].text.startswith("Hi")
    assert rendered[0].text.endswith(" Expand")


def test_system_messages_render_with_persisted_roles(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    home.append_session_event(
        session.path,
        "system_message",
        {"message": "Blocked command: reboot", "role": "forbidden"},
    )
    app = AnomxCliApp(home=home, cwd=repo)

    lines = app._read_message_lines(session.path)
    assert lines == [
        MessageLine("forbidden", "Blocked: reboot")
    ]
    assert lines[0].detail_title == ""


def test_forbidden_messages_use_light_attr(tmp_path):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)
    app._colors = {
        "light": curses.A_DIM,
        "danger": curses.A_BOLD,
    }

    assert app._line_attr("forbidden") == curses.A_DIM
    assert app._line_attr("notice") == curses.A_DIM


def test_runtime_asks_before_cli_paths_outside_workspace(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), repo)

    output = runtime._execute_tool(
        "run_command",
        {"statement": "I am checking a file", "command": "cat ../secret.txt"},
        RuntimeCallbacks(),
    )

    payload = json.loads(output)
    assert payload["approved"] is False
    assert payload["output"] == "Command requires approval."


def test_runtime_allows_nested_launch_to_edit_discovered_workspace_root(tmp_path):
    repo = tmp_path / "repo"
    nested = repo / "src" / "anomx" / "agent"
    nested.mkdir(parents=True)
    (repo / ".git").mkdir()
    (repo / "source.txt").write_text("hello", encoding="utf-8")
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), nested)

    output = runtime._execute_tool(
        "run_command",
        {
            "statement": "Writing file from nested launch directory",
            "command": "cat ../../../source.txt > ../../../target.txt",
        },
        RuntimeCallbacks(approval=lambda _request: ApprovalChoice.ALLOW),
    )

    payload = json.loads(output)
    assert payload["approved"] is True
    assert (repo / "target.txt").read_text(encoding="utf-8") == "hello"


def test_runtime_still_asks_for_paths_outside_discovered_workspace_root(tmp_path):
    repo = tmp_path / "repo"
    nested = repo / "src" / "anomx" / "agent"
    nested.mkdir(parents=True)
    (repo / ".git").mkdir()
    (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), nested)

    output = runtime._execute_tool(
        "run_command",
        {"statement": "Reading outside file", "command": "cat ../../../../secret.txt"},
        RuntimeCallbacks(),
    )

    payload = json.loads(output)
    assert payload["approved"] is False
    assert payload["output"] == "Command requires approval."


def test_tool_manager_terminates_running_command_when_cancelled(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    cancel_event = runtime_module.threading.Event()
    manager = CliToolManager(repo, mode=AgentMode.AUTONOMOUS, cancel_event=cancel_event)
    result: dict[str, str] = {}

    def run_command():
        result["output"] = manager._execute_shell_command("sleep 10; echo done")

    thread = runtime_module.threading.Thread(target=run_command)
    thread.start()
    runtime_module.time.sleep(0.1)

    cancel_event.set()
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert result["output"] == "Command stopped because the agent was interrupted."


def test_runtime_requests_approval_for_command_paths_outside_workspace(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), repo)
    events: list[tuple[str, str]] = []
    approval_reasons: list[str] = []

    def reject_request(request: CommandApprovalRequest) -> ApprovalChoice:
        approval_reasons.append(request.reason)
        return ApprovalChoice.REJECT

    runtime._execute_tool(
        "run_command",
        {"statement": "Reading outside workspace", "command": f"cat {outside}"},
        RuntimeCallbacks(
            approval=reject_request,
            system_message=lambda role, message: events.append((role, message)),
        ),
    )

    assert approval_reasons == [f"Path is outside the trusted workspace: {outside}"]
    assert events == []


def test_strict_workspace_allows_agent_skill_and_response_roots(tmp_path):
    workspace = tmp_path / "chat-workspace"
    agent_home = tmp_path / "agent-home"
    responses = agent_home / ".anomx" / "responses"
    skills = agent_home / ".anomx" / "skills"
    workspace.mkdir()
    responses.mkdir(parents=True)
    skills.mkdir(parents=True)
    response_path = responses / "anomx-api.json"
    response_path.write_text('{"ok": true}', encoding="utf-8")
    manager = CliToolManager(
        workspace,
        mode=AgentMode.SANDBOX,
        current_dir=workspace,
        subprocess_env={"HOME": str(agent_home)},
        strict_workspace=True,
        trusted_roots=(responses, skills),
    )

    absolute_policy = manager.classify(f"cat {response_path}")
    home_policy = manager.classify(
        "cat ~/.anomx/responses/anomx-api.json | python3 -m json.tool"
    )
    private_config_policy = manager.classify("cat ~/.anomx/config.toml")
    prompt = "\n".join(manager.workspace_prompt_lines())

    assert absolute_policy.safety == CommandSafety.ALLOW
    assert home_policy.safety == CommandSafety.ALLOW
    assert private_config_policy.safety == CommandSafety.FORBIDDEN
    assert str(responses) in prompt
    assert str(skills) in prompt


def test_local_sandbox_allows_agent_response_root_reads_and_copies(tmp_path):
    workspace = tmp_path / "chat-workspace"
    agent_home = tmp_path / "agent-home"
    responses = agent_home / ".anomx" / "responses"
    workspace.mkdir()
    responses.mkdir(parents=True)
    response_path = responses / "anomx-api.json"
    response_path.write_text('{"ok": true}', encoding="utf-8")
    session = LocalSandboxSession(
        LocalSandboxConfig(
            root=workspace,
            home=agent_home,
            current_dir=workspace,
            trusted_roots=(responses,),
        )
    )

    read_output = session.exec_command("cat ~/.anomx/responses/anomx-api.json")
    copy_output = session.exec_command(
        "cp ~/.anomx/responses/anomx-api.json anomx-folders.json"
    )
    private_config_output = session.exec_command("cat ~/.anomx/config.toml")

    assert read_output == '{"ok": true}'
    assert copy_output == "Paths copied."
    assert (workspace / "anomx-folders.json").read_text(encoding="utf-8") == '{"ok": true}'
    assert "outside the sandbox root" in private_config_output


def test_runtime_file_tools_allow_agent_response_root(tmp_path):
    workspace = tmp_path / "chat-workspace"
    agent_home = tmp_path / "agent-home"
    anomx_home = AnomxHome(agent_home / ".anomx")
    responses = anomx_home.responses_dir
    workspace.mkdir()
    responses.mkdir(parents=True)
    response_path = responses / "anomx-api.json"
    response_path.write_text('{"ok": true}', encoding="utf-8")
    runtime = AgentRuntime(
        anomx_home,
        workspace,
        mode=AgentMode.SANDBOX,
        local_sandbox_enabled=True,
    )

    output = ReadFileTool(statement_description="Read a file.").execute(
        {
            "statement": "Reading API response",
            "path": "~/.anomx/responses/anomx-api.json",
            "start_line": 1,
            "max_lines": 20,
        },
        ToolExecutionContext(runtime=runtime, callbacks=RuntimeCallbacks()),
    )

    payload = json.loads(output)
    assert payload["path"] == str(response_path)
    assert payload["content"] == '{"ok": true}'


def test_runtime_includes_current_mode_in_system_prompt(tmp_path):
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path, mode=AgentMode.CONFIRM)

    assert "Current mode: Standard Mode." in runtime._instructions()

    runtime.set_mode(AgentMode.AUTO)

    assert "Current mode: Automatic Mode." in runtime._instructions()

    runtime.set_mode(AgentMode.AUTONOMOUS)
    instructions = runtime._instructions()

    assert "Current mode: Autonomous Mode." in instructions
    assert "inside or outside the trusted workspace root" not in instructions


def test_runtime_includes_workspace_access_in_system_prompt(tmp_path):
    repo = tmp_path / "repo"
    nested = repo / "src" / "anomx" / "agent"
    nested.mkdir(parents=True)
    (repo / ".git").mkdir()
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), nested)

    instructions = runtime._instructions()

    assert f"Trusted workspace root: {repo}" in instructions
    assert f"Shell starts in: {nested}" in instructions


def test_runtime_includes_onboarding_user_name_in_system_prompt(tmp_path):
    home = AnomxHome(tmp_path / "home")
    config = home.load_config()
    config["user_name"] = "Ada"
    home.save_config(config)
    runtime = AgentRuntime(home, tmp_path)

    assert "User profile:\n- Name: Ada" in runtime._instructions()


def test_runtime_memory_prompt_uses_metadata_and_points_to_brain_dir(tmp_path):
    home = AnomxHome(tmp_path / "home")
    write_memory(
        home.brain_dir,
        create_memory_record(
            title="Avoid Global Installs",
            summary="Do not run global package installs without explicit approval.",
            kind=MemoryKind.MANUAL,
            context={"source": "test"},
            content="Full private memory content should stay out of the prompt.",
        ),
    )
    runtime = AgentRuntime(home, tmp_path)

    instructions = runtime._instructions()

    assert "## Memories" in instructions
    assert "~/.anomx/brain" in instructions
    assert str(home.brain_dir) in instructions
    assert "inspect that folder yourself" in instructions
    assert "Avoid Global Installs" in instructions
    assert "Do not run global package installs without explicit approval." in instructions
    assert "Full private memory content should stay out of the prompt." not in instructions


def test_operator_prompt_pushes_execution_after_planning(tmp_path):
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path, mode=AgentMode.CONFIRM)
    instructions = runtime._instructions()

    assert "A plan is not a stopping point." in instructions
    assert "# Anomx Operator Agent" in instructions
    assert "## Delegation" in instructions
    assert "Favor working with subagents." in instructions
    assert "start at least one Worker agent" in instructions
    assert "Do not ask for that approval in prose before calling tools." in instructions


def test_runtime_includes_session_command_policy_in_operator_and_worker_prompts(tmp_path):
    allowed = {"cmd:python"}
    rejected = {"cmd:curl"}

    operator = AgentRuntime(
        AnomxHome(tmp_path / "home"),
        tmp_path,
        allowed,
        rejected,
        role=AgentRole.OPERATOR,
    )
    worker = AgentRuntime(
        AnomxHome(tmp_path / "home"),
        tmp_path,
        allowed,
        rejected,
        role=AgentRole.WORKER,
    )

    operator_instructions = operator._instructions()
    worker_instructions = worker._instructions()

    assert "# Anomx Worker Agent" in worker_instructions
    assert "## Commands" in worker_instructions
    assert "Already approved for this session: python." in operator_instructions
    assert "Never call run_command with these command families in this session: curl." in (
        operator_instructions
    )
    assert "Never call run_command with these command families in this session: curl." in (
        worker_instructions
    )


def test_runtime_tool_schemas_are_role_specific(tmp_path):
    operator = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path)
    worker = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path, role=AgentRole.WORKER)

    operator_tools = operator._openai_tools()
    operator_names = [tool["name"] for tool in operator_tools]
    operator_run_command = next(tool for tool in operator_tools if tool["name"] == "run_command")
    worker_tools = worker._openai_tools()

    assert "run_command" in operator_names
    assert "run_cli_command" not in operator_names
    assert "statement" in operator_run_command["parameters"]["properties"]
    assert operator_run_command["parameters"]["required"] == ["statement", "command"]
    expected_operator_tools = {
        "start_process",
        "end_process",
        "ask_question",
        "memorize",
        "create_plan",
        "update_plan",
        "remove_plan",
        "finish_anyways",
        "start_subagent",
        "prompt_subagent",
        "remove_subagent",
        "get_subagent_info",
    }
    assert expected_operator_tools.issubset(set(operator_names))
    assert "check_agent" not in operator_names
    assert "stop_agent" not in operator_names
    assert "wait" not in operator_names
    for tool in operator_tools:
        if tool["name"] in {
            "create_plan",
            "get_subagent_info",
            "output_message",
            "update_plan",
            "wait",
        }:
            continue
        assert "statement" in tool["parameters"]["properties"]
        assert "statement" in tool["parameters"]["required"]
    assert [tool["name"] for tool in worker_tools] == [
        "run_command",
        "start_process",
        "end_process",
        "web_search",
        "web_fetch",
    ]
    assert worker_tools[0]["parameters"]["required"] == ["statement", "command"]


def test_wait_tool_is_only_advertised_with_running_workers(tmp_path):
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path)

    assert "wait" not in [tool["name"] for tool in runtime._openai_tools()]

    runtime._running_worker_states = lambda: ("running",)  # type: ignore[method-assign]

    wait_tool = next(tool for tool in runtime._openai_tools() if tool["name"] == "wait")
    assert wait_tool["parameters"] == {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }


def test_ask_question_tool_returns_interactive_answer(tmp_path):
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path)
    seen_requests: list[QuestionRequest] = []

    def answer_question(request: QuestionRequest) -> QuestionResponse:
        seen_requests.append(request)
        return QuestionResponse(
            answered=True,
            answer="next",
            selected_label="Next.js",
            kind=request.kind,
        )

    output = runtime._execute_tool(
        "ask_question",
        {
            "statement": "Choosing framework",
            "question": "Which framework should I use?",
            "kind": "select",
            "options": [
                {
                    "label": "Next.js",
                    "value": "next",
                    "description": "Use the existing website stack.",
                }
            ],
            "placeholder": None,
            "default": None,
            "allow_custom": False,
        },
        RuntimeCallbacks(question=answer_question),
    )

    payload = json.loads(output)

    assert payload == {
        "answer": "next",
        "answered": True,
        "cancelled": False,
        "kind": "select",
        "selected_label": "Next.js",
    }
    assert seen_requests[0].options[0].label == "Next.js"


def test_memorize_tool_writes_agent_memory(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    runtime = AgentRuntime(home, tmp_path)
    monkeypatch.setattr(
        runtime,
        "suggest_memory_metadata",
        lambda **_kwargs: MemoryMetadata(
            "Prefer Pytest",
            "Use pytest for package validation.",
        ),
    )

    output = runtime._execute_tool(
        "memorize",
        {
            "statement": "Saving preference",
            "content": "Use pytest for package validation unless the user asks otherwise.",
            "context": {"source": "test"},
            "title": None,
            "summary": None,
        },
        RuntimeCallbacks(),
    )

    payload = json.loads(output)
    memories = load_memories(home.brain_dir)

    assert payload["created"] is True
    assert payload["title"] == "Prefer Pytest"
    assert memories[0].kind == MemoryKind.AGENT
    assert (
        memories[0].content
        == "Use pytest for package validation unless the user asks otherwise."
    )


def test_ask_question_rejects_select_without_options(tmp_path):
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path)

    output = runtime._execute_tool(
        "ask_question",
        {
            "statement": "Choosing framework",
            "question": "Which framework should I use?",
            "kind": "select",
            "options": [],
            "placeholder": None,
            "default": None,
            "allow_custom": False,
        },
        RuntimeCallbacks(question=lambda request: QuestionResponse(True, "unused")),
    )

    payload = json.loads(output)

    assert payload["answered"] is False
    assert "select questions require options" in payload["error"]


def test_ollama_provisional_text_stays_private_while_worker_runs(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    session = home.create_session(tmp_path, provider="ollama", model="qwen3.6")
    runtime = AgentRuntime(home, tmp_path)
    progress_messages: list[str] = []
    captured_messages: list[list[dict[str, object]]] = []
    worker_running = True

    responses = iter(
        (
            runtime_module.OllamaStreamResponse(
                "The Engineer is working. I will wait.",
                "",
                (),
                {
                    "role": "assistant",
                    "content": "The Engineer is working. I will wait.",
                },
            ),
            runtime_module.OllamaStreamResponse(
                "The work is complete.",
                "",
                (),
                {"role": "assistant", "content": "The work is complete."},
            ),
        )
    )

    def fake_running_workers():
        return ("running",) if worker_running else ()

    def fake_wait(arguments, callbacks=None):
        nonlocal worker_running
        assert arguments == {}
        worker_running = False
        return '{"waited_seconds": 0.0, "workers": []}'

    def fake_stream(model, messages, callbacks):
        assert model == "qwen3.6"
        captured_messages.append([dict(message) for message in messages])
        return next(responses)

    monkeypatch.setattr(runtime, "_running_worker_states", fake_running_workers)
    monkeypatch.setattr(runtime, "_wait_tool", fake_wait)
    monkeypatch.setattr(runtime, "_stream_ollama_response", fake_stream)

    response = runtime.ollama_response(
        session.path,
        "qwen3.6",
        RuntimeCallbacks(message=progress_messages.append),
    )

    assert response == "The work is complete."
    assert progress_messages == []
    assert len(captured_messages) == 2
    assert captured_messages[1][-1]["role"] == "user"
    assert "not delivered to the user" in str(captured_messages[1][-1]["content"])
    assert "not a final answer" in str(captured_messages[1][-1]["content"])


def test_operator_run_command_persists_statement_as_work_message(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("hello", encoding="utf-8")
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), repo)
    messages: list[str] = []

    output = runtime._execute_tool(
        "run_command",
        {"statement": "Reading project overview", "command": "cat README.md"},
        RuntimeCallbacks(tool_message=messages.append),
    )

    payload = json.loads(output)
    assert payload["approved"] is True
    assert payload["output"] == "hello"
    assert messages == ["Reading project overview"]


def test_pending_approval_uses_menu_even_in_autonomous(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=repo, use_color=False)
    app.agent_mode = AgentMode.AUTONOMOUS
    monkeypatch.setattr(
        app,
        "_bottom_menu",
        lambda *_args, **_kwargs: ApprovalChoice.REJECT.value,
    )

    decision = app._request_command_approval(
        object(),
        session,
        CommandApprovalRequest(
            command="cat source.txt > target.txt",
            statement="Writing file",
            reason="Shell operators require explicit approval.",
            canonical_command="cat source.txt > target.txt",
            allowance_label="cat commands",
            allowance_subject="cat",
        ),
    )

    assert decision == ApprovalChoice.REJECT


def test_approval_menu_describes_command_family_allowance(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=repo, use_color=False)
    captured: dict[str, object] = {}

    def fake_bottom_menu(
        _stdscr,
        _session,
        title,
        subtitle,
        choices,
        **kwargs,
    ):
        captured["title"] = title
        captured["subtitle"] = subtitle
        captured["choices"] = choices
        captured["autonomous_value"] = kwargs.get("autonomous_value")
        captured["scroll"] = kwargs.get("scroll")
        captured["anchor_line"] = kwargs.get("anchor_line")
        return ApprovalChoice.ALWAYS_ALLOW.value

    monkeypatch.setattr(app, "_bottom_menu", fake_bottom_menu)

    decision = app._request_command_approval(
        object(),
        session,
        CommandApprovalRequest(
            command="cat source.txt > target.txt",
            statement="Writing file",
            reason="Shell operators require explicit approval.",
            canonical_command="cat source.txt > target.txt",
            allowance_label="cat commands",
            allowance_subject="cat",
        ),
        scroll=6,
        anchor_line=3,
    )

    choices = captured["choices"]
    assert decision == ApprovalChoice.ALWAYS_ALLOW
    assert captured["title"] == "Shell operators require explicit approval."
    assert captured["subtitle"] == "cat source.txt > target.txt"
    assert captured["autonomous_value"] == ApprovalChoice.ALLOW.value
    assert captured["scroll"] == 6
    assert captured["anchor_line"] == 3
    assert choices[0].label == "Approve"
    assert choices[2].label == "Approve always"
    assert choices[2].detail == "Trust cat commands globally for cat"
    assert choices[3].label == "Reject always, because ..."


def test_reject_always_captures_memory_reason(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=repo, use_color=False)

    monkeypatch.setattr(
        app,
        "_bottom_menu",
        lambda *_args, **_kwargs: ApprovalChoice.ALWAYS_REJECT.value,
    )
    monkeypatch.setattr(
        app,
        "_prompt_multiline_text",
        lambda *_args, **_kwargs: "Do not run package install commands globally.",
    )

    decision = app._request_command_approval(
        object(),
        session,
        CommandApprovalRequest(
            command="npm install",
            statement="Installing dependencies",
            reason="npm may read, compute, install, or modify files.",
            canonical_command="npm install",
            allowance_label="npm commands",
            allowance_subject="npm",
        ),
    )

    assert decision == ApprovalChoice.ALWAYS_REJECT
    assert app._approval_memory_reason == "Do not run package install commands globally."


def test_commands_config_panel_displays_saved_parameters(tmp_path):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)
    app.session_allowed_commands.add("cmd:rm -rf")

    assert app._command_panel_label("rm -rf") == "rm"
    assert app._command_panel_detail("rm -rf") == "Approved · Parameters: -rf"


def test_evaluated_approval_panel_uses_risk_and_description(tmp_path):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)
    request = CommandApprovalRequest(
        command="rm -rf build",
        statement="Remove build output",
        reason="rm may read, compute, install, or modify files.",
        canonical_command="rm -rf build",
        allowance_label="rm -rf commands",
        allowance_subject="rm -rf",
        evaluation=CommandRiskEvaluation(
            risk="high",
            description="Deletes the build directory recursively and cannot be undone.",
        ),
    )
    panel = app._command_approval_panel(
        request,
        app._command_approval_choices(request),
        0,
        show_command=False,
        command_scroll=0,
    )
    command_panel = app._command_approval_panel(
        request,
        app._command_approval_choices(request),
        0,
        show_command=True,
        command_scroll=2,
    )

    assert panel.title_prefix == "High Risk"
    assert panel.title_prefix_attr == "danger"
    assert panel.title_attr == "bold"
    assert app._command_risk_label("low") == "Low Risk"
    assert app._command_risk_attr("low") == "ok"
    assert panel.subtitle == "Deletes the build directory recursively and cannot be undone."
    assert command_panel.subtitle == "rm -rf build"
    assert command_panel.subtitle_max_lines == 5
    assert command_panel.subtitle_scroll == 2


def test_extract_json_object_handles_fenced_model_text():
    text = 'Here is the result:\n```json\n{"risk":"medium","description":"Writes a file."}\n```'

    assert extract_json_object(text) == {
        "risk": "medium",
        "description": "Writes a file.",
    }


def test_runtime_enriches_command_approval_request(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    home.append_session_event(
        session.path,
        "user_message",
        {"message": "Clean generated assets"},
    )
    runtime = AgentRuntime(home, repo)
    request = CommandApprovalRequest(
        command="rm -rf build",
        statement="Remove generated build output",
        reason="rm may read, compute, install, or modify files.",
        canonical_command="rm -rf build",
    )

    def fake_evaluate(*, command, statement, user_message, model):
        assert command == "rm -rf build"
        assert statement == "Remove generated build output"
        assert user_message == "Clean generated assets"
        assert model == "gpt-5.5"
        return CommandRiskEvaluation("high", "Deletes generated assets recursively.")

    class FakeBackend:
        evaluate_command_request = staticmethod(fake_evaluate)

    monkeypatch.setattr(
        runtime_module,
        "backend_for_provider",
        lambda _provider, _runtime: FakeBackend(),
    )
    evaluation = runtime.evaluate_command_request(session.path, request)

    assert evaluation == CommandRiskEvaluation("high", "Deletes generated assets recursively.")


def test_openai_command_evaluation_uses_structured_output(tmp_path, monkeypatch):
    import anomx.agent.backends.openai as openai_module
    from anomx.agent.backends.openai import OpenAIBackend

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "output_text": json.dumps(
                        {
                            "risk": "medium",
                            "description": "Runs a package build script.",
                        }
                    )
                }
            ).encode("utf-8")

    home = AnomxHome(tmp_path / "home")
    auth = home.load_auth()
    auth["api_keys"] = {"openai": "sk-test"}
    home.save_auth(auth)
    runtime = AgentRuntime(home, tmp_path)
    captured_payload: dict[str, object] = {}

    def fake_urlopen(request, timeout):
        captured_payload.update(json.loads(request.data.decode("utf-8")))
        assert timeout == 8
        return FakeResponse()

    monkeypatch.setattr(openai_module.urllib.request, "urlopen", fake_urlopen)

    evaluation = OpenAIBackend(runtime).evaluate_command_request(
        command="npm run build",
        statement="Build the project",
        user_message="Check whether the project builds",
        model="gpt-5.5",
    )

    assert evaluation == CommandRiskEvaluation("medium", "Runs a package build script.")
    assert captured_payload["text"] == {
        "format": {
            "type": "json_schema",
            "name": "command_risk_evaluation",
            "schema": {
                "type": "object",
                "properties": {
                    "risk": {"type": "string", "enum": ["low", "medium", "high"]},
                    "description": {"type": "string"},
                },
                "required": ["risk", "description"],
                "additionalProperties": False,
            },
            "strict": True,
        }
    }


def test_approval_menu_can_always_reject_command_family(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=repo, use_color=False)

    monkeypatch.setattr(
        app,
        "_bottom_menu",
        lambda *_args, **_kwargs: ApprovalChoice.ALWAYS_REJECT.value,
    )

    decision = app._request_command_approval(
        object(),
        session,
        CommandApprovalRequest(
            command="curl -s https://example.com",
            statement="Fetching URL",
            reason="curl may read, compute, install, or modify files.",
            canonical_command="curl -s https://example.com",
            allowance_label="curl commands",
            allowance_subject="curl",
        ),
    )

    assert decision == ApprovalChoice.ALWAYS_REJECT


def test_request_question_select_uses_bottom_menu(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=repo, use_color=False)
    captured: dict[str, object] = {}

    def fake_bottom_menu(_stdscr, _session, title, subtitle, choices, **kwargs):
        captured["title"] = title
        captured["subtitle"] = subtitle
        captured["choices"] = choices
        captured["scroll"] = kwargs.get("scroll")
        captured["anchor_line"] = kwargs.get("anchor_line")
        return "react"

    monkeypatch.setattr(app, "_bottom_menu", fake_bottom_menu)

    answer = app._request_question(
        object(),
        session,
        QuestionRequest(
            question="Which UI stack?",
            kind="select",
            options=(
                runtime_module.QuestionOption("React", "react", "Use React."),
                runtime_module.QuestionOption("Vue", "vue", "Use Vue."),
            ),
        ),
        scroll=6,
        anchor_line=3,
    )

    assert answer == QuestionResponse(
        answered=True,
        answer="react",
        selected_label="React",
        kind="select",
    )
    assert captured["title"] == "Question"
    assert captured["subtitle"] == "Which UI stack?"
    assert captured["choices"][0].label == "React"
    assert captured["scroll"] == 6
    assert captured["anchor_line"] == 3


def test_request_question_confirm_adds_yes_no_options(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=repo, use_color=False)
    captured: dict[str, object] = {}

    def fake_bottom_menu(_stdscr, _session, _title, _subtitle, choices, **_kwargs):
        captured["choices"] = choices
        return "yes"

    monkeypatch.setattr(app, "_bottom_menu", fake_bottom_menu)

    answer = app._request_question(
        object(),
        session,
        QuestionRequest(question="Continue?", kind="confirm"),
    )

    assert answer.answer == "yes"
    assert answer.selected_label == "Yes"
    assert [choice.value for choice in captured["choices"]] == ["yes", "no"]


def test_request_question_text_preserves_viewport(tmp_path, monkeypatch):
    class Window:
        def get_wch(self):
            return "\n"

        def nodelay(self, _flag):
            pass

    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=repo, use_color=False)
    captured: list[dict[str, object]] = []

    def fake_draw_session(
        _stdscr,
        _session,
        _messages,
        _input_text,
        _cursor,
        scroll,
        **kwargs,
    ):
        captured.append(
            {
                "scroll": scroll,
                "anchor_line": kwargs.get("anchor_line"),
            }
        )
        return None

    monkeypatch.setattr(app, "_draw_session", fake_draw_session)

    answer = app._request_question(
        Window(),
        session,
        QuestionRequest(question="What is your name?", kind="text", default="Ada"),
        scroll=9,
        anchor_line=2,
    )

    assert answer.answer == "Ada"
    assert captured == [{"scroll": 9, "anchor_line": 2}]


def test_create_and_update_plan_are_persisted(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    runtime = AgentRuntime(home, repo)

    runtime._execute_tool(
        "create_plan",
        {
            "steps": [
                {
                    "title": "Inspect",
                    "description": "Read the relevant files.",
                    "is_done": False,
                },
                {
                    "title": "Validate",
                    "description": "Run focused checks.",
                    "is_done": False,
                },
            ]
        },
        RuntimeCallbacks(),
        session.path,
    )
    runtime._execute_tool(
        "update_plan",
        {
            "steps": [
                {
                    "position": 1,
                    "title": "Inspect",
                    "description": "Read the relevant files.",
                    "is_done": True,
                },
                {
                    "position": 2,
                    "title": "Validate",
                    "description": "Run focused checks.",
                    "is_done": False,
                },
            ]
        },
        RuntimeCallbacks(),
        session.path,
    )

    steps = latest_plan_steps(home.read_session_events(session.path))
    instructions = runtime._instructions(session.path)

    assert [(step.position, step.title, step.is_done) for step in steps] == [
        (1, "Inspect", True),
        (2, "Validate", False),
    ]
    assert "1. [done] Inspect" in instructions
    assert "keep the current plan up to date" in instructions
    assert "every plan step should be marked done before ending operations" in instructions


def test_remove_plan_clears_current_plan(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    runtime = AgentRuntime(home, repo)
    messages: list[str] = []

    runtime._execute_tool(
        "create_plan",
        {
            "steps": [
                {
                    "title": "Inspect",
                    "description": "Read files.",
                    "is_done": False,
                }
            ]
        },
        RuntimeCallbacks(),
        session.path,
    )
    output = runtime._execute_tool(
        "remove_plan",
        {},
        RuntimeCallbacks(tool_message=messages.append),
        session.path,
    )

    assert json.loads(output)["removed"] is True
    assert latest_plan_steps(home.read_session_events(session.path)) == ()
    assert messages == ["Removed plan"]


def test_finish_anyways_clears_current_plan(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    runtime = AgentRuntime(home, repo)
    messages: list[str] = []

    runtime._execute_tool(
        "create_plan",
        {
            "steps": [
                {
                    "title": "Inspect",
                    "description": "Read files.",
                    "is_done": False,
                }
            ]
        },
        RuntimeCallbacks(),
        session.path,
    )
    output = runtime._execute_tool(
        "finish_anyways",
        {"statement": "Finishing despite stale plan"},
        RuntimeCallbacks(tool_message=messages.append),
        session.path,
    )

    assert json.loads(output) == {"finish_anyways": True, "removed_plan": True}
    assert latest_plan_steps(home.read_session_events(session.path)) == ()
    assert messages == ["Finishing despite stale plan"]


def test_question_message_is_delivered_despite_open_plan_steps(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    runtime = AgentRuntime(home, repo)

    runtime._execute_tool(
        "create_plan",
        {
            "steps": [
                {
                    "title": "Phase 7",
                    "description": "Continue the refactor.",
                    "is_done": False,
                },
            ]
        },
        RuntimeCallbacks(),
        session.path,
    )

    # The agent hands control back with a question while a plan step is open.
    # It must be delivered and the turn must end (wait for the user), not be
    # suppressed by the plan-finish guard.
    prompt, used_guard = runtime._continuation_prompt_after_text(
        "Ich warte auf deine Antwort: Soll ich mit Phase 7 weitermachen?",
        RuntimeCallbacks(),
        session.path,
        0,
    )

    assert prompt is None
    assert used_guard is False
    work_messages = [
        event_payload(event)
        for event in home.read_session_events(session.path)
        if event_payload_type(event) == "work_message"
    ]
    assert work_messages == []


def test_unfinished_plan_blocks_final_answer_with_work_reprompt(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    runtime = AgentRuntime(home, repo)

    runtime._execute_tool(
        "create_plan",
        {
            "steps": [
                {
                    "title": "Inspect",
                    "description": "Read files.",
                    "is_done": True,
                },
                {
                    "title": "Validate",
                    "description": "Run checks.",
                    "is_done": False,
                },
            ]
        },
        RuntimeCallbacks(),
        session.path,
    )

    prompt, used_guard = runtime._continuation_prompt_after_text(
        "Done.",
        RuntimeCallbacks(),
        session.path,
        0,
    )

    events = home.read_session_events(session.path)
    work_messages = [
        event_payload(event)
        for event in events
        if event_payload_type(event) == "work_message"
    ]

    assert used_guard is True
    assert prompt is not None
    assert "unfinished steps" in prompt
    assert "2. [open] Validate" in prompt
    assert "finish_anyways(statement)" in prompt
    assert work_messages == [
        {
            "type": "work_message",
            "message": "Validating whether the plan is finished",
            "role": "tool",
        }
    ]

    prompt, used_guard = runtime._continuation_prompt_after_text(
        "Done.",
        RuntimeCallbacks(),
        session.path,
        MAX_PLAN_FINISH_REPROMPTS,
    )

    assert prompt is None
    assert used_guard is False


def test_plan_validation_work_is_turn_scoped_and_collapsible(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    runtime = AgentRuntime(home, repo)
    runtime._execute_tool(
        "create_plan",
        {
            "steps": [
                {
                    "title": "Inspect",
                    "description": "Read files.",
                    "is_done": False,
                }
            ]
        },
        RuntimeCallbacks(),
        session.path,
    )
    events: queue.SimpleQueue[RuntimeUiEvent] = queue.SimpleQueue()

    prompt, used_guard = runtime._continuation_prompt_after_text(
        "This provisional text should not render.",
        RuntimeCallbacks(
            message=lambda message: events.put(RuntimeUiEvent("message", message)),
            tool_message=lambda message: events.put(RuntimeUiEvent("tool_message", message)),
        ),
        session.path,
        0,
    )
    app = AnomxCliApp(home=home, cwd=repo, use_color=False)
    monkeypatch.setattr(app, "_fake_type_message", lambda *args, **kwargs: None)

    *_unused, work_count = app._process_runtime_events(
        object(),
        session,
        events,
        "Thinking",
        None,
        "",
        "turn-1",
        0,
        None,
    )

    assert used_guard is True
    assert prompt is not None
    assert work_count == 1
    assert app._read_message_lines(session.path) == [
        MessageLine("tool", "Validating whether the plan is finished", "turn-1")
    ]

    home.append_session_event(
        session.path,
        "work_summary",
        {"message": "Worked for 00:04 min", "turn_id": "turn-1"},
    )
    home.append_session_event(session.path, "agent_message", {"message": "Final answer"})

    assert app._read_message_lines(session.path) == [
        MessageLine("work_summary", "Worked for 00:04 min · expand", "turn-1"),
        MessageLine("agent", "Final answer"),
    ]


def test_continuation_guard_does_not_emit_provisional_agent_message(
    tmp_path, monkeypatch
):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    runtime = AgentRuntime(home, repo)
    delivered: list[str] = []
    monkeypatch.setattr(runtime, "_running_worker_states", lambda: ("worker-1",))
    monkeypatch.setattr(runtime, "_wait_tool", lambda _arguments, _callbacks: "{}")

    prompt, used_guard = runtime._continuation_prompt_after_text(
        "This is not an output_message update.",
        RuntimeCallbacks(message=delivered.append),
        session.path,
        0,
    )

    assert prompt is not None
    assert used_guard is False
    assert delivered == []


def test_completed_plan_is_removed_automatically_on_final_answer(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    runtime = AgentRuntime(home, repo)

    runtime._execute_tool(
        "create_plan",
        {
            "steps": [
                {
                    "title": "Inspect",
                    "description": "Read files.",
                    "is_done": True,
                },
                {
                    "title": "Validate",
                    "description": "Run checks.",
                    "is_done": True,
                },
            ]
        },
        RuntimeCallbacks(),
        session.path,
    )

    prompt, used_guard = runtime._continuation_prompt_after_text(
        "Done.",
        RuntimeCallbacks(),
        session.path,
        0,
    )

    assert prompt is None
    assert used_guard is False
    assert latest_plan_steps(home.read_session_events(session.path)) == ()


def test_worker_prompt_sessions_are_stored_under_parent_session_folder(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    runtime = AgentRuntime(
        home,
        repo,
        role=AgentRole.WORKER,
        process_owner_id="worker-1",
        process_owner_name="Engineer",
    )

    response = runtime.backend_response_for_prompt(
        "Inspect files",
        parent_session_path=session.path,
    )

    worker_dir = session.path.parent / "worker-Engineer-worker-1"
    worker_session = worker_dir / "session.jsonl"
    assert "OpenAI API key is not configured" in response
    assert worker_session.exists()
    events = _read_jsonl(worker_session)
    assert events[0]["type"] == "worker_session_meta"
    assert events[0]["payload"]["parent_session_id"] == session.session_id
    assert events[0]["payload"]["parent_session_path"] == str(session.path)
    assert events[0]["payload"]["worker_name"] == "Engineer"
    assert events[0]["payload"]["worker_id"] == "worker-1"
    assert events[1]["payload"]["type"] == "user_message"
    assert events[1]["payload"]["message"] == "Inspect files"
    assert [record.session_id for record in home.list_sessions(limit=None)] == [
        session.session_id
    ]

    runtime.backend_response_for_prompt(
        "Inspect tests",
        parent_session_path=session.path,
    )

    events = _read_jsonl(worker_session)
    worker_prompts = [
        event["payload"]["message"]
        for event in events
        if event["type"] == "event_msg"
        and event["payload"].get("type") == "user_message"
    ]
    assert worker_prompts == ["Inspect files", "Inspect tests"]

    assert home.delete_session(session.path) is True
    assert not worker_dir.exists()


def test_worker_lifecycle_events_are_persisted(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    runtime = AgentRuntime(home, repo)

    def fake_worker_response(self, prompt, callbacks=None, **_kwargs):
        assert prompt == "Inspect files"
        if callbacks is not None and callbacks.tool_message is not None:
            callbacks.tool_message("Listing files")
        if callbacks is not None and callbacks.command is not None:
            callbacks.command("Listing files", "ls", "README.md")
        if callbacks is not None and callbacks.status is not None:
            callbacks.status("Thinking")
        return "Worker report"

    monkeypatch.setattr(AgentRuntime, "backend_response_for_prompt", fake_worker_response)

    started = json.loads(
        runtime._execute_tool(
            "start_agent",
            {"name": "Engineer", "prompt": "Inspect files"},
            RuntimeCallbacks(),
            session.path,
        )
    )
    runtime._wait_tool({"seconds": 1})

    events = home.read_session_events(session.path)
    snapshots = worker_snapshots(events)
    messages = AnomxCliApp(home=home, cwd=repo)._read_message_lines(session.path)
    checked = json.loads(
        runtime._execute_tool(
            "check_agent",
            {"agent_id": started["agent_id"]},
            RuntimeCallbacks(),
            session.path,
        )
    )

    assert started["name"] == "Engineer"
    assert snapshots[-1].status == "ready"
    assert snapshots[-1].statement == ""
    assert snapshots[-1].context_tokens > 0
    assert snapshots[-1].context_percent > 0
    assert running_worker_snapshots(events) == ()
    assert not any(line.role == "worker" and "Worker report" in line.text for line in messages)
    assert snapshots[-1].command_history == (
        {
            "statement": "Listing files",
            "command": "ls",
            "output": "README.md",
        },
    )
    assert checked["commands"] == [
        {
            "statement": "Listing files",
            "command": "ls",
            "output": "README.md",
        }
    ]
    running_statements = [
        str(event["payload"].get("statement"))
        for event in events
        if event_payload_type(event) == "worker_event"
        and event_payload(event).get("status") == "working"
    ]
    assert running_statements[-1] == "Listing files"
    assert "Thinking" not in running_statements[1:]


def test_subagent_tool_schemas_are_role_specific(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()

    build_tools = {
        tool["name"] for tool in AgentRuntime(home, repo, role=AgentRole.BUILD)._tool_definitions()
    }
    general_tools = {
        tool["name"]
        for tool in AgentRuntime(home, repo, role=AgentRole.GENERAL)._tool_definitions()
    }
    explore_tools = [
        tool["name"]
        for tool in AgentRuntime(home, repo, role=AgentRole.EXPLORE)._tool_definitions()
    ]
    scout_tools = [
        tool["name"]
        for tool in AgentRuntime(home, repo, role=AgentRole.SCOUT)._tool_definitions()
    ]

    assert {
        "start_subagent",
        "prompt_subagent",
        "remove_subagent",
        "get_subagent_info",
    }.issubset(build_tools)
    assert "run_command" in general_tools
    assert explore_tools == ["bash", "read", "list", "glob", "grep", "web_search", "web_fetch"]
    assert scout_tools == ["bash", "read", "list", "glob", "grep", "web_search", "web_fetch"]


def test_read_only_subagent_denies_write_commands(tmp_path):
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path, role=AgentRole.EXPLORE)

    output = runtime._execute_tool(
        "bash",
        {"statement": "Trying write", "command": "touch should-not-exist"},
        RuntimeCallbacks(),
    )

    payload = json.loads(output)
    assert payload["approved"] is False
    assert payload["safety"] == "forbidden"
    assert "read-only" in payload["output"]


def test_build_runtime_starts_subagent_and_persists_events(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    runtime = AgentRuntime(home, repo)

    def fake_backend_response(self, session_path, callbacks=None, **_kwargs):
        active_callbacks = RuntimeCallbacks() if callbacks is None else callbacks
        if active_callbacks.status is not None:
            active_callbacks.status("Reading files")
        if active_callbacks.tool_message is not None:
            active_callbacks.tool_message("Listing files")
        if active_callbacks.command is not None:
            active_callbacks.command("Listing files", "ls", "README.md")
        return "Subagent report"

    monkeypatch.setattr(AgentRuntime, "backend_response", fake_backend_response)

    started = json.loads(
        runtime._execute_tool(
            "start_subagent",
            {
                "statement": "Starting explorer",
                "agent_kind": "explore",
                "name": "Explorer",
                "prompt": "Inspect files",
            },
            RuntimeCallbacks(),
            session.path,
        )
    )
    deadline = time.monotonic() + 2
    while runtime._running_subagent_states() and time.monotonic() < deadline:
        time.sleep(0.01)

    events = home.read_session_events(session.path)
    snapshots = subagent_snapshots(events)
    checked = json.loads(
        runtime._execute_tool(
            "get_subagent_info",
            {"agent_id": started["agent_id"]},
            RuntimeCallbacks(),
            session.path,
        )
    )

    assert started["started"] is True
    assert snapshots[-1].name == "Explorer"
    assert snapshots[-1].kind == "explore"
    assert snapshots[-1].status == "ready"
    assert snapshots[-1].response == "Subagent report"
    assert snapshots[-1].command_history == (
        {
            "statement": "Listing files",
            "command": "ls",
            "output": "README.md",
        },
    )
    assert checked["latest_outputs"][-1]["text"] == "Subagent report"
    assert checked["commands"] == [
        {
            "statement": "Listing files",
            "command": "ls",
            "output": "README.md",
        }
    ]


def test_async_process_lifecycle_is_persisted_and_contextualized(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    runtime = AgentRuntime(home, repo)

    started = json.loads(
        runtime._execute_tool(
            "start_process",
            {"statement": "Starting dev server", "command": "sleep 10"},
            RuntimeCallbacks(approval=lambda _request: ApprovalChoice.ALLOW),
            session.path,
        )
    )

    process_id = started["process_id"]
    events = home.read_session_events(session.path)
    assert started["started"] is True
    assert running_process_snapshots(events)[0].process_id == process_id
    assert process_id in runtime._instructions(session.path)
    assert "sleep 10" in runtime._instructions(session.path)

    ended = json.loads(runtime.end_process(process_id, session.path))

    assert ended["ended"] is True
    assert running_process_snapshots(home.read_session_events(session.path)) == ()


def test_runtime_writes_finished_command_and_process_full_logs(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(tool_manager_module, "LONG_RUNNING_COMMAND_SECONDS", 0.01)
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    logs_dir = tmp_path / "request-logs"
    config = home.load_config()
    config["debug_mode"] = True
    config["debug_full_session_logs"] = True
    config["debug_full_session_logs_path"] = str(logs_dir)
    home.save_config(config)
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    home.update_session_title(session.path, "Runtime Logs")
    runtime = AgentRuntime(home, repo)
    script = repo / "slow_command.py"
    script.write_text(
        "import time\nprint('cmd-ready', flush=True)\ntime.sleep(0.2)\nprint('cmd-done')\n",
        encoding="utf-8",
    )

    command_output = json.loads(
        runtime._execute_tool(
            "run_command",
            {
                "statement": "Run long command",
                "command": f"{sys.executable} {script}",
            },
            RuntimeCallbacks(approval=lambda _request: ApprovalChoice.ALLOW),
            session.path,
        )
    )
    process_output = json.loads(
        runtime._execute_tool(
            "start_process",
            {
                "statement": "Run async process",
                "command": f"{sys.executable} -c \"print('proc-ready')\"",
            },
            RuntimeCallbacks(approval=lambda _request: ApprovalChoice.ALLOW),
            session.path,
        )
    )

    session_logs_dir = logs_dir / f"{session.session_id} - Runtime Logs"
    command_log = session_logs_dir / "Commands" / "0001.txt"
    process_log = session_logs_dir / "Processes" / "0001.txt"
    for _ in range(40):
        if command_log.exists() and process_log.exists():
            break
        time.sleep(0.05)

    assert command_output["status"] == "ended"
    assert process_output["started"] is True
    assert command_log.exists()
    assert process_log.exists()
    assert "cmd-ready" in command_log.read_text(encoding="utf-8")
    assert "cmd-done" in command_log.read_text(encoding="utf-8")
    assert "proc-ready" in process_log.read_text(encoding="utf-8")


def test_operator_long_running_command_is_temporarily_promoted_to_bottom_panel(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(tool_manager_module, "LONG_RUNNING_COMMAND_SECONDS", 0.01)
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    runtime = AgentRuntime(home, repo)
    statuses: list[str] = []

    output = runtime._execute_tool(
        "run_command",
        {
            "statement": "Installing dependencies",
            "command": "sleep 0.2; echo done",
        },
        RuntimeCallbacks(
            approval=lambda _request: ApprovalChoice.ALLOW,
            status=statuses.append,
        ),
        session.path,
    )

    payload = json.loads(output)
    events = home.read_session_events(session.path)
    process_events = [
        event_payload(event)
        for event in events
        if event_payload_type(event) == "process_event"
    ]

    assert payload["approved"] is True
    assert payload["status"] == "ended"
    assert payload["command_id"]
    assert payload["output"] == "done"
    assert "Waiting:60.0" in statuses
    assert statuses[-1] == "Thinking"
    assert any(
        event.get("source") == "command" and event.get("status") == "running"
        for event in process_events
    )
    assert any(
        event.get("source") == "command" and event.get("status") == "ended"
        for event in process_events
    )
    assert running_process_snapshots(events) == ()


def test_operator_long_running_command_status_and_kill_are_scoped(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(tool_manager_module, "LONG_RUNNING_COMMAND_SECONDS", 0.01)
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    script = repo / "long_command.py"
    script.write_text(
        "import time\nprint('ready', flush=True)\ntime.sleep(10)\n",
        encoding="utf-8",
    )
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    runtime = AgentRuntime(home, repo)

    def no_wait(process_state, callbacks=None):
        del callbacks
        payload = runtime._command_state_payload(process_state)
        payload["waited_seconds"] = 0.0
        return payload

    monkeypatch.setattr(runtime, "_wait_for_command_state", no_wait)

    output = runtime._execute_tool(
        "run_command",
        {
            "statement": "Installing dependencies",
            "command": f"{sys.executable} {script}",
        },
        RuntimeCallbacks(approval=lambda _request: ApprovalChoice.ALLOW),
        session.path,
    )
    payload = json.loads(output)
    command_id = payload["command_id"]

    for _ in range(20):
        status = json.loads(
            runtime._execute_tool(
                "check_command_status",
                {"command_id": command_id},
                RuntimeCallbacks(),
                session.path,
            )
        )
        if "ready" in status.get("output", ""):
            break
        time.sleep(0.05)

    tool_names = {tool["name"] for tool in runtime._tool_definitions()}
    process = running_process_snapshots(home.read_session_events(session.path))[0]

    assert payload["status"] == "running"
    assert {"check_command_status", "kill_command", "wait"} <= tool_names
    assert status["command_id"] == command_id
    assert status["status"] == "running"
    assert "ready" in status["output"]
    assert "ready" in process.output
    killed = json.loads(
        runtime._execute_tool(
            "kill_command",
            {"command_id": command_id},
            RuntimeCallbacks(),
            session.path,
        )
    )
    assert killed["ended"] is True
    assert running_process_snapshots(home.read_session_events(session.path)) == ()


def test_long_running_command_process_callback_receives_live_output(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(tool_manager_module, "LONG_RUNNING_COMMAND_SECONDS", 0.01)
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    script = repo / "live_output.py"
    script.write_text(
        "import time\nprint('live-ready', flush=True)\ntime.sleep(10)\n",
        encoding="utf-8",
    )
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    runtime = AgentRuntime(home, repo)
    process_updates: list[tuple[str, str]] = []

    def no_wait(process_state, callbacks=None):
        del callbacks
        payload = runtime._command_state_payload(process_state)
        payload["waited_seconds"] = 0.0
        return payload

    def record_process(process_state):
        process_updates.append((process_state.status, process_state.output))

    monkeypatch.setattr(runtime, "_wait_for_command_state", no_wait)

    output = runtime._execute_tool(
        "run_command",
        {
            "statement": "Run live process",
            "command": f"{sys.executable} {script}",
        },
        RuntimeCallbacks(
            approval=lambda _request: ApprovalChoice.ALLOW,
            process=record_process,
        ),
        session.path,
    )
    command_id = json.loads(output)["command_id"]
    try:
        for _ in range(30):
            if any("live-ready" in update_output for _status, update_output in process_updates):
                break
            time.sleep(0.05)
        assert any("live-ready" in update_output for _status, update_output in process_updates)
    finally:
        runtime._execute_tool(
            "kill_command",
            {"command_id": command_id},
            RuntimeCallbacks(process=record_process),
            session.path,
        )


def test_worker_long_running_command_tools_do_not_leak_to_operator(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(tool_manager_module, "LONG_RUNNING_COMMAND_SECONDS", 0.01)
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    script = repo / "worker_long_command.py"
    script.write_text(
        "import time\nprint('worker-ready', flush=True)\ntime.sleep(10)\n",
        encoding="utf-8",
    )
    operator_session = home.create_session(repo, provider="openai", model="gpt-5.5")
    operator_runtime = AgentRuntime(home, repo)
    worker_runtime = AgentRuntime(
        home,
        repo,
        role=AgentRole.WORKER,
        process_owner_id="worker123",
        process_owner_name="Engineer",
    )

    def no_wait(process_state, callbacks=None):
        del callbacks
        payload = worker_runtime._command_state_payload(process_state)
        payload["waited_seconds"] = 0.0
        return payload

    def mirror_process(process_state):
        with operator_runtime._process_lock:
            operator_runtime._processes[process_state.process_id] = process_state
        operator_runtime._append_process_event(operator_session.path, process_state)

    monkeypatch.setattr(worker_runtime, "_wait_for_command_state", no_wait)

    output = worker_runtime._execute_tool(
        "run_command",
        {
            "statement": "Installing dependencies",
            "command": f"{sys.executable} {script}",
        },
        RuntimeCallbacks(
            approval=lambda _request: ApprovalChoice.ALLOW,
            process=mirror_process,
        ),
        operator_session.path,
    )
    payload = json.loads(output)
    command_id = payload["command_id"]

    for _ in range(20):
        status = json.loads(
            worker_runtime._execute_tool(
                "check_command_status",
                {"command_id": command_id},
                RuntimeCallbacks(),
                operator_session.path,
            )
        )
        if "worker-ready" in status.get("output", ""):
            break
        time.sleep(0.05)

    worker_tool_names = {tool["name"] for tool in worker_runtime._tool_definitions()}
    operator_tool_names = {tool["name"] for tool in operator_runtime._tool_definitions()}
    worker_instructions = worker_runtime._instructions(operator_session.path)
    operator_instructions = operator_runtime._instructions(operator_session.path)
    operator_status = json.loads(
        operator_runtime._execute_tool(
            "check_command_status",
            {"command_id": command_id},
            RuntimeCallbacks(),
            operator_session.path,
        )
    )
    process = running_process_snapshots(home.read_session_events(operator_session.path))[0]
    killed = json.loads(
        worker_runtime._execute_tool(
            "kill_command",
            {"command_id": command_id},
            RuntimeCallbacks(process=mirror_process),
            operator_session.path,
        )
    )

    assert payload["source"] == "worker_command"
    assert process.owner_id == "worker123"
    assert process.owner_name == "Engineer"
    assert "worker-ready" in process.output
    assert {"check_command_status", "kill_command", "wait"} <= worker_tool_names
    assert "check_command_status" not in operator_tool_names
    assert "kill_command" not in operator_tool_names
    assert "check_command_status" in worker_instructions
    assert "kill_command" in worker_instructions
    assert "check_command_status" not in operator_instructions
    assert "kill_command" not in operator_instructions
    assert operator_status["error"] == "Unknown command id."
    assert "worker-ready" in status["output"]
    assert killed["ended"] is True
    assert running_process_snapshots(home.read_session_events(operator_session.path)) == ()


def test_header_box_draws_plan_steps(tmp_path):
    class Window:
        def __init__(self):
            self.writes = []

        def getmaxyx(self):
            return 24, 120

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)
    window = Window()
    steps = (
        PlanStep(1, "Inspect", "Read files", False),
        PlanStep(2, "Validate", "Run checks", True),
    )

    app._draw_header_box(
        window,
        "Session",
        "/repo",
        steps,
        header_meta="abc123 · openai/gpt-5.5",
        plan_expanded=True,
    )

    assert any(
        text == f"abc123 · openai/gpt-5.5 · v{__version__}"
        for _, _, text, _ in window.writes
    )
    assert any(text == "Anomx" for _, _, text, _ in window.writes)
    assert any(text == "." for _, _, text, _ in window.writes)
    assert any(
        text == "Data Analysis and Anomaly Detection Agent"
        for _, _, text, _ in window.writes
    )
    assert any("☐ Inspect" in text for _, _, text, _ in window.writes)
    assert any("☑" in text and "V\u0336" in text for _, _, text, _ in window.writes)
    assert app._session_body_top(steps, plan_expanded=True) == 10
    assert app._session_body_top(steps, subtitle_line_count=2, plan_expanded=True) == 11


def test_header_box_collapses_plan_to_current_step(tmp_path):
    class Window:
        def __init__(self):
            self.writes = []

        def getmaxyx(self):
            return 24, 120

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)
    window = Window()
    steps = (
        PlanStep(1, "Inspect", "Read files", True),
        PlanStep(2, "Validate", "Run checks", False),
    )

    app._draw_header_box(window, "Session", "/repo", steps, title_suffix="00:02")

    assert any("Session › Validate 00:02" in text for _, _, text, _ in window.writes)
    assert not any("expand" in text for _, _, text, _ in window.writes)
    assert not any("☐ Validate" in text for _, _, text, _ in window.writes)
    assert app._session_body_top(steps) == 7
    title_actions = [
        action
        for actions in app._click_targets.values()
        for action in actions
        if action.kind == "toggle_plan"
    ]
    assert title_actions
    assert title_actions[0].x_start == 4
    assert title_actions[0].x_end >= len("Session › Validate 00:02")


def test_startup_brand_draws_randomized_dot_cell(tmp_path):
    class Window:
        def __init__(self):
            self.writes = []

        def getmaxyx(self):
            return 24, 80

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)
    app._colors = {"brand_dot": 77, "matrix_brand": 55}
    window = Window()

    app._draw_startup_brand(
        window,
        24,
        80,
        0,
        reveal_progress=1.0,
        removal_progress=0.0,
    )

    dot_writes = [text for _, _, text, attr in window.writes if attr == 77]
    assert dot_writes
    assert all(text.isalnum() for text in dot_writes)
    assert not any(text == "." for text in dot_writes)


def test_new_plan_reveals_header_steps_incrementally(tmp_path):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)
    steps_payload = [
        {"position": 1, "title": "Inspect", "description": "", "is_done": False},
        {"position": 2, "title": "Build", "description": "", "is_done": False},
        {"position": 3, "title": "Validate", "description": "", "is_done": False},
    ]
    recent_create_event = {
        "type": "event_msg",
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "payload": {
            "type": "plan_update",
            "action": "create",
            "steps": steps_payload,
        },
    }
    recent_create_events = [recent_create_event]
    older_create_events = [
        {
            **recent_create_event,
            "timestamp": (datetime.now(tz=UTC) - timedelta(seconds=1.2)).isoformat(),
        }
    ]
    update_events = [
        {
            **recent_create_event,
            "payload": {
                "type": "plan_update",
                "action": "update",
                "steps": steps_payload,
            },
        }
    ]

    assert [
        step.title
        for step in app._visible_plan_steps(
            recent_create_events,
            latest_plan_steps(recent_create_events),
        )
    ] == ["Inspect"]
    assert [
        step.title
        for step in app._visible_plan_steps(
            older_create_events,
            latest_plan_steps(older_create_events),
        )
    ] == ["Inspect", "Build"]
    assert [
        step.title
        for step in app._visible_plan_steps(update_events, latest_plan_steps(update_events))
    ] == ["Inspect", "Build", "Validate"]


def test_approved_command_rows_show_session_allowances(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home)

    assert app._approved_command_rows(session) == (
        InfoRow("Session", "No commands approved yet"),
    )

    app.session_allowed_commands.update({"cmd:python", "cmd:curl"})
    home.append_session_event(
        session.path,
        "work_message",
        {"message": "Approved command: python -V", "role": "tool"},
    )

    assert app._approved_command_rows(session) == (
        InfoRow("Always approved", "curl"),
        InfoRow("Always approved", "python"),
        InfoRow("Approved", "python -V"),
    )


def test_draw_session_registers_click_target_for_collapsed_work_line(tmp_path):
    class Window:
        def __init__(self):
            self.writes = []

        def erase(self):
            pass

        def getmaxyx(self):
            return 28, 80

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

        def refresh(self):
            pass

    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=repo, use_color=False)
    window = Window()
    message = "Approved command: " + "python -c 'print(\"long command\")' " * 4

    app._draw_session(
        window,
        session,
        [MessageLine("approved", message, expansion_key="line-1")],
        "",
        0,
        0,
    )

    assert any(text.endswith("...") for _, _, text, _ in window.writes)
    assert not any("click to expand" in text for _, _, text, _ in window.writes)
    assert any(
        action.kind == "toggle_work_line" and action.text == "line-1"
        for actions in app._click_targets.values()
        for action in actions
    )


def test_draw_session_registers_click_target_for_expanded_work_box(tmp_path):
    class Window:
        def __init__(self):
            self.writes = []

        def erase(self):
            pass

        def getmaxyx(self):
            return 28, 80

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

        def refresh(self):
            pass

    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=repo, use_color=False)
    app._toggle_work_line("line-1")

    app._draw_session(
        Window(),
        session,
        [
            MessageLine(
                "approved",
                "Approved command: python script.py",
                expansion_key="line-1",
                detail_body="python script.py",
            )
        ],
        "",
        0,
        0,
    )

    assert any(
        action.kind == "toggle_work_line" and action.text == "line-1"
        for actions in app._click_targets.values()
        for action in actions
    )


def test_session_mouse_release_does_not_toggle_work_line(tmp_path, monkeypatch):
    class Window:
        def getmaxyx(self):
            return 30, 80

    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)
    app._click_targets = {
        5: [SessionMouseAction("toggle_work_line", 0, "line-1")],
    }

    monkeypatch.setattr(
        curses,
        "getmouse",
        lambda: (0, 4, 5, 0, curses.BUTTON1_RELEASED),
    )

    assert app._session_mouse_action(Window(), "", []) is None

    monkeypatch.setattr(
        curses,
        "getmouse",
        lambda: (0, 4, 5, 0, curses.BUTTON1_PRESSED),
    )

    action = app._session_mouse_action(Window(), "", [])

    assert action is not None
    assert action.kind == "toggle_work_line"
    assert action.text == "line-1"


def test_draw_session_renders_slash_commands_in_bottom_panel(tmp_path):
    class Window:
        def __init__(self):
            self.writes = []

        def erase(self):
            pass

        def getmaxyx(self):
            return 28, 80

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

        def refresh(self):
            pass

    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=repo, use_color=False)
    window = Window()
    suggestions = app._filtered_commands("/")

    app._draw_session(
        window,
        session,
        [],
        "/",
        1,
        0,
        command_suggestions=suggestions,
        command_selected=4,
    )

    assert any(text == "Commands" for _, _, text, _ in window.writes)
    assert any(text == "Choose a command to run" for _, _, text, _ in window.writes)
    assert any(text == "› /config" for _, _, text, _ in window.writes)


def test_draw_empty_session_renders_starter_skill_hints(tmp_path):
    class Window:
        def __init__(self):
            self.writes = []

        def erase(self):
            pass

        def getmaxyx(self):
            return 36, 120

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

        def refresh(self):
            pass

    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=repo, use_color=False)
    window = Window()

    app._draw_session(window, session, [], "", 0, 0, start_hint_reveal_progress=1.0)

    assert any(text == "Map the folder" for _, _, text, _ in window.writes)
    assert any(text == "Find issues" for _, _, text, _ in window.writes)
    assert any(text == "Make a report" for _, _, text, _ in window.writes)


def test_draw_session_includes_back_to_project_link(tmp_path):
    class Window:
        def __init__(self):
            self.writes = []

        def erase(self):
            pass

        def getmaxyx(self):
            return 28, 100

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

        def refresh(self):
            pass

    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    home.save_project(repo, "Signal Lab")
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=repo, use_color=False)
    window = Window()

    app._draw_session(window, session, [], "", 0, 0)

    assert any(text == "Signal Lab › New session" for _, _, text, _ in window.writes)
    assert any(text == "Back to Project" for _, _, text, _ in window.writes)
    assert any(text == "esc" for _, _, text, _ in window.writes)
    assert any(
        action.kind == "back_project"
        for actions in app._click_targets.values()
        for action in actions
    )


def test_draw_session_moves_running_counter_to_header_title(tmp_path):
    class Window:
        def __init__(self):
            self.writes = []

        def erase(self):
            pass

        def getmaxyx(self):
            return 28, 120

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

        def refresh(self):
            pass

    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    home.save_project(repo, "Signal Lab")
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    messages = [MessageLine("user", "Build the app")]
    app = AnomxCliApp(home=home, cwd=repo, use_color=False)
    window = Window()

    app._draw_session(window, session, messages, "", 0, 0, active_turn_elapsed=2.4)

    assert any("Signal Lab › New session 00:02" in text for _, _, text, _ in window.writes)
    assert not any("Build the app · 00:02" in text for _, _, text, _ in window.writes)


def test_draw_project_lists_sessions_with_activity_age(tmp_path):
    class Window:
        def __init__(self):
            self.writes = []

        def erase(self):
            pass

        def getmaxyx(self):
            return 28, 110

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

        def refresh(self):
            pass

    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    project = home.save_project(repo, "Signal Lab")
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    home.update_session_title(session.path, "Map Datasets")
    session = home.list_sessions(limit=None)[0]
    app = AnomxCliApp(home=home, cwd=repo, use_color=False)
    session = replace(
        session,
        updated_at=(datetime.now(tz=UTC) - timedelta(minutes=2)).isoformat(
            timespec="seconds"
        ),
    )
    window = Window()

    app._draw_project(window, project, [session], 0, 0, "", 0, "", "light", 0)

    assert any(text == "Signal Lab" for _, _, text, _ in window.writes)
    assert any(text == "Map Datasets" for _, _, text, _ in window.writes)
    assert any("2min" in text for _, _, text, _ in window.writes)
    assert not any("expand" in text for _, _, text, _ in window.writes)
    assert any(
        action.kind == "open_project_session"
        for actions in app._click_targets.values()
        for action in actions
    )


def test_project_sessions_order_by_last_user_message_not_background_activity(
    tmp_path,
    monkeypatch,
):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    app = AnomxCliApp(home=home, cwd=repo, use_color=False)

    monkeypatch.setattr(store_module, "utc_now_iso", lambda: "2026-01-01T00:00:00Z")
    first = home.create_session(repo, provider="openai", model="gpt-5.5")
    home.update_session_title(first.path, "First Session")
    monkeypatch.setattr(store_module, "utc_now_iso", lambda: "2026-01-01T00:01:00Z")
    second = home.create_session(repo, provider="openai", model="gpt-5.5")
    home.update_session_title(second.path, "Second Session")

    monkeypatch.setattr(store_module, "utc_now_iso", lambda: "2026-01-01T00:02:00Z")
    home.append_session_event(first.path, "user_message", {"message": "first prompt"})
    monkeypatch.setattr(store_module, "utc_now_iso", lambda: "2026-01-01T00:03:00Z")
    home.append_session_event(second.path, "user_message", {"message": "second prompt"})
    monkeypatch.setattr(store_module, "utc_now_iso", lambda: "2026-01-01T00:04:00Z")
    home.append_session_event(first.path, "agent_message", {"message": "background reply"})

    sessions = app._project_sessions(repo)

    assert [session.title for session in sessions[:2]] == [
        "Second Session",
        "First Session",
    ]
    assert sessions[1].updated_at == "2026-01-01T00:04:00Z"


def test_draw_project_selected_session_shows_delete_affordance(tmp_path):
    class Window:
        def __init__(self):
            self.writes = []

        def erase(self):
            pass

        def getmaxyx(self):
            return 28, 120

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

        def refresh(self):
            pass

    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    project = home.save_project(repo, "Signal Lab")
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=repo, use_color=False)

    window = Window()
    app._draw_project(window, project, [session], 0, 0, "", 0, "", "light", 0)
    assert any(
        text == "›" and attr == app._attr("accent")
        for _, _, text, attr in window.writes
    )
    assert any(
        text == "New session" and attr == app._attr("light")
        for _, _, text, attr in window.writes
    )
    assert any(
        "ctrl+d Delete" in text and attr == app._attr("light")
        for _, _, text, attr in window.writes
    )

    confirm_window = Window()
    app._draw_project(
        confirm_window,
        project,
        [session],
        0,
        0,
        "",
        0,
        "",
        "light",
        0,
        delete_pending_index=0,
    )
    assert any(
        "Enter to confirm" in text and attr == app._attr("light")
        for _, _, text, attr in confirm_window.writes
    )


def test_draw_project_running_session_shows_statement_and_timer(tmp_path, monkeypatch):
    class Window:
        def __init__(self):
            self.writes = []

        def erase(self):
            pass

        def getmaxyx(self):
            return 28, 170

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

        def refresh(self):
            pass

    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    project = home.save_project(repo, "Signal Lab")
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=repo, use_color=False)
    turn = ui_module.ActiveSessionTurn(
        session=session,
        runtime=app.runtime,
        events=queue.SimpleQueue(),
        result={},
        turn_id="turn-1",
        started_at=2.0,
        worker=threading.Thread(target=lambda: None),
        mode=AgentMode.CONFIRM,
        working_text="Reading project overview",
    )
    app._active_session_turns[app._session_turn_key(session)] = turn
    monkeypatch.setattr(ui_module.time, "monotonic", lambda: 12.0)
    window = Window()

    app._draw_project(window, project, [session], 0, 0, "", 0, "", "light", 0)

    assert any(text == "›" and attr == app._attr("accent") for _, _, text, attr in window.writes)
    assert any(
        text == "New session" and attr == app._attr("accent")
        for _, _, text, attr in window.writes
    )
    assert any("Reading project overview" in text for _, _, text, _ in window.writes)
    assert any(text == "00:10" for _, _, text, _ in window.writes)
    assert not any("expand" in text for _, _, text, _ in window.writes)


def test_draw_project_unread_session_shows_primary_dot(tmp_path):
    class Window:
        def __init__(self):
            self.writes = []

        def erase(self):
            pass

        def getmaxyx(self):
            return 28, 120

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

        def refresh(self):
            pass

    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    project = home.save_project(repo, "Signal Lab")
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    session = replace(session, unread=True)
    app = AnomxCliApp(home=home, cwd=repo, use_color=False)
    window = Window()

    app._draw_project(window, project, [session], 0, 0, "", 0, "", "light", 0)

    assert not any("ctrl+d · " in text for _, _, text, _ in window.writes)
    assert any("ctrl+d Delete" in text for _, _, text, _ in window.writes)
    assert any(text == "•" and attr == app._attr("accent") for _, _, text, attr in window.writes)


def test_project_view_defers_pending_approval_and_shows_confirmation(tmp_path, monkeypatch):
    class Window:
        def __init__(self):
            self.writes = []

        def erase(self):
            pass

        def getmaxyx(self):
            return 28, 140

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

        def refresh(self):
            pass

    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    project = home.save_project(repo, "Signal Lab")
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=repo, use_color=False)
    released = threading.Event()
    worker = threading.Thread(target=lambda: released.wait(2.0))
    worker.start()
    events: queue.SimpleQueue[RuntimeUiEvent] = queue.SimpleQueue()
    events.put(
        RuntimeUiEvent(
            "approval",
            approval_request=CommandApprovalRequest(
                command="python build.py",
                statement="Build artifacts",
                reason="Needs approval",
                canonical_command="python build.py",
                allowance_subject="python",
            ),
            approval_response=queue.SimpleQueue(),
        )
    )
    turn = ui_module.ActiveSessionTurn(
        session=session,
        runtime=app.runtime,
        events=events,
        result={},
        turn_id="turn-1",
        started_at=time.monotonic(),
        worker=worker,
        mode=AgentMode.CONFIRM,
    )
    app._active_session_turns[app._session_turn_key(session)] = turn
    monkeypatch.setattr(
        app,
        "_request_command_approval",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected prompt")),
    )

    app._drain_active_session_turns(Window())
    window = Window()
    app._draw_project(window, project, [session], 0, 0, "", 0, "", "light", 0)

    assert turn.pending_events and turn.pending_events[0].kind == "approval"
    assert any(
        "Confirmation needed" in text and attr == app._attr("warning_badge")
        for _, _, text, attr in window.writes
    )

    released.set()
    worker.join(timeout=1.0)


def test_draw_project_displays_command_controls_for_slash(tmp_path):
    class Window:
        def __init__(self):
            self.writes = []

        def erase(self):
            pass

        def getmaxyx(self):
            return 32, 120

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

        def refresh(self):
            pass

    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    project = home.save_project(repo, "Signal Lab")
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=repo, use_color=False)
    suggestions = app._filtered_commands("/")
    window = Window()

    app._draw_project(
        window,
        project,
        [session],
        0,
        0,
        "/",
        1,
        "",
        "light",
        0,
        command_suggestions=suggestions,
        command_selected=0,
    )

    assert any(text == "Commands" for _, _, text, _ in window.writes)
    assert any("/new" in text for _, _, text, _ in window.writes)


def test_project_background_completion_marks_session_unread(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=repo, use_color=False)
    worker = threading.Thread(target=lambda: None)
    worker.start()
    worker.join()
    turn = ui_module.ActiveSessionTurn(
        session=session,
        runtime=app.runtime,
        events=queue.SimpleQueue(),
        result={"response": "Final answer"},
        turn_id="turn-1",
        started_at=time.monotonic(),
        worker=worker,
        mode=AgentMode.CONFIRM,
    )
    app._active_session_turns[app._session_turn_key(session)] = turn

    app._drain_active_session_turns(object())

    assert home.list_sessions()[0].unread is True


def test_draw_project_keeps_completed_sessions_to_one_line(tmp_path):
    class Window:
        def __init__(self):
            self.writes = []

        def erase(self):
            pass

        def getmaxyx(self):
            return 28, 120

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

        def refresh(self):
            pass

    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    project = home.save_project(repo, "Signal Lab")
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    home.append_session_event(session.path, "agent_message", {"message": "Long final answer"})
    session = home.list_sessions(limit=None)[0]
    app = AnomxCliApp(home=home, cwd=repo, use_color=False)
    window = Window()

    app._draw_project(window, project, [session], 0, 0, "", 0, "", "light", 0)

    rows_with_session_text = [
        y
        for y, _x, text, _attr in window.writes
        if "New session" in text or "Long final answer" in text
    ]
    assert rows_with_session_text == [min(rows_with_session_text)]


def test_transient_session_messages_show_streaming_text_without_elapsed_time(tmp_path):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)
    messages = [MessageLine("user", "Build the app")]

    rendered = app._messages_with_transient_state(messages, 2.4, "Working on it")

    assert rendered[0] == MessageLine("user", "Build the app")
    assert rendered[1] == MessageLine("agent", "Working on it")
    assert messages == [MessageLine("user", "Build the app")]


def test_streaming_text_suppresses_live_thinking_status(tmp_path):
    class Window:
        def __init__(self):
            self.writes = []

        def erase(self):
            pass

        def getmaxyx(self):
            return 28, 120

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

        def refresh(self):
            pass

    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    home.save_project(repo, "Signal Lab")
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    home.append_session_event(session.path, "user_message", {"message": "Inspect"})
    app = AnomxCliApp(home=home, cwd=repo, use_color=False)
    window = Window()

    app._draw_session(
        window,
        session,
        app._read_message_lines(session.path),
        "",
        0,
        0,
        working_text="Thinking",
        streaming_text="Final answer is streaming",
    )

    assert any("Final answer is streaming" in text for _, _, text, _ in window.writes)
    assert not any("Thinking" in text for _, _, text, _ in window.writes)


def test_streaming_delta_waits_to_collapse_until_turn_completion(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=repo, use_color=False)
    events: queue.SimpleQueue[RuntimeUiEvent] = queue.SimpleQueue()
    events.put(RuntimeUiEvent("tool_message", "Checking README"))
    events.put(RuntimeUiEvent("delta", "Final answer"))
    worker = threading.Thread(target=lambda: None)
    worker.start()
    worker.join()
    turn = ui_module.ActiveSessionTurn(
        session=session,
        runtime=app.runtime,
        events=events,
        result={},
        turn_id="turn-1",
        started_at=100.0,
        worker=worker,
        mode=AgentMode.CONFIRM,
    )
    monkeypatch.setattr(time, "monotonic", lambda: 103.0)

    app._drain_session_turn_events(object(), turn, render_events=False)

    assert turn.final_text == "Final answer"
    assert turn.work_summary_appended is False
    assert app._read_message_lines(session.path) == [
        MessageLine("tool", "Checking README", "turn-1")
    ]

    captured_final_render: dict[str, object] = {}

    def fake_type_message(*args, **kwargs):
        captured_final_render["anchor_line"] = kwargs.get("anchor_line")
        captured_final_render["scroll"] = kwargs.get("scroll")

    monkeypatch.setattr(app, "_fake_type_message", fake_type_message)
    app._complete_session_turn(object(), turn, anchor_line=7, scroll=3, render_final=True)

    assert turn.work_summary_appended is True
    assert captured_final_render == {"anchor_line": 7, "scroll": 3}
    assert app._read_message_lines(session.path) == [
        MessageLine("work_summary", "Worked for 00:03 · expand", "turn-1"),
        MessageLine("agent", "Final answer"),
    ]


def test_fake_type_scroll_preserves_running_sticky_anchor(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=repo, use_color=False)

    assert app._handle_fake_type_key(
        object(),
        session,
        curses.KEY_UP,
        "",
        0,
        7,
        0,
        True,
    ) == (7, 1, True)
    assert app._handle_fake_type_mouse_action(
        session,
        SessionMouseAction("scroll", 1),
        7,
        0,
        True,
    ) == (7, 1, True)
    assert app._handle_fake_type_key(
        object(),
        session,
        curses.KEY_UP,
        "",
        0,
        7,
        0,
        False,
    ) == (None, 1, False)


def test_start_hints_remain_visible_while_typing(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=repo, use_color=False)

    assert app._start_hints_visible_for_prompt(session.path, [], "hello", [], [])


def test_start_hint_outline_is_not_randomized_during_reveal(tmp_path):
    class Window:
        def __init__(self):
            self.writes = []

        def getmaxyx(self):
            return 36, 120

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)
    skill = app._starter_skills()[0]
    window = Window()

    app._draw_start_hint_card(
        window,
        1,
        2,
        24,
        7,
        skill,
        frame=7,
        reveal_progress=0.0,
        removal_progress=0.0,
    )

    assert any(text == "╭" + ("─" * 22) + "╮" for _, _, text, _ in window.writes)
    assert any(text == "╰" + ("─" * 22) + "╯" for _, _, text, _ in window.writes)


def test_session_mouse_action_maps_starter_hint_click_to_skill(tmp_path, monkeypatch):
    class Window:
        def __init__(self):
            self.writes = []

        def erase(self):
            pass

        def getmaxyx(self):
            return 36, 120

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

        def refresh(self):
            pass

    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=repo, use_color=False)
    window = Window()

    app._draw_session(window, session, [], "", 0, 0)
    target_y, target_action = next(
        (y, action)
        for y, actions in app._click_targets.items()
        for action in actions
        if action.kind == "skill" and action.text == "map-folder"
    )
    monkeypatch.setattr(
        curses,
        "getmouse",
        lambda: (0, target_action.x_start, target_y, 0, curses.BUTTON1_CLICKED),
    )

    action = app._session_mouse_action(window, "", [])

    assert action is not None
    assert action.kind == "skill"
    assert action.text == "map-folder"


def test_session_mouse_drag_copies_selected_transcript_text(tmp_path, monkeypatch):
    class Window:
        def __init__(self):
            self.writes = []

        def erase(self):
            pass

        def getmaxyx(self):
            return 28, 80

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

        def refresh(self):
            pass

    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=repo, use_color=False)
    window = Window()
    messages = [
        MessageLine("agent", "hello world"),
        MessageLine("agent", "second line"),
    ]
    copied: list[str] = []

    app._draw_session(window, session, messages, "", 0, 0)
    rows = sorted(app._session_text_rows.values(), key=lambda row: row.line_index)
    monkeypatch.setattr(app, "_copy_to_clipboard", lambda text: copied.append(text) or True)

    monkeypatch.setattr(
        curses,
        "getmouse",
        lambda: (0, rows[0].x + 6, rows[0].y, 0, curses.BUTTON1_PRESSED),
    )
    start = app._session_mouse_action(window, "", [])

    monkeypatch.setattr(
        curses,
        "getmouse",
        lambda: (0, rows[1].x + 6, rows[1].y, 0, curses.BUTTON1_RELEASED),
    )
    done = app._session_mouse_action(window, "", [])

    assert start is not None
    assert start.kind == "selection"
    assert done is not None
    assert done.kind == "selection"
    assert done.value == len("world\nsecond")
    assert copied == ["world\nsecond"]

    window = Window()
    app._draw_session(window, session, messages, "", 0, 0)
    assert any(
        text == "world" and attr == app._attr("selected")
        for _, _, text, attr in window.writes
    )
    assert any(
        text == "second" and attr == app._attr("selected")
        for _, _, text, attr in window.writes
    )


def test_run_session_invokes_clicked_starter_skill(tmp_path, monkeypatch):
    class Window:
        def __init__(self):
            self._keys = iter((curses.KEY_MOUSE,))

        def get_wch(self):
            return next(self._keys)

    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=repo, use_color=False)
    invoked: list[tuple[str, str]] = []

    monkeypatch.setattr(app, "_draw_session", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        app,
        "_session_mouse_action",
        lambda *_args, **_kwargs: SessionMouseAction("skill", 0, "map-folder"),
    )

    def record_skill(_stdscr, _session, skill, submitted):
        invoked.append((skill.command, submitted))
        return "exit"

    monkeypatch.setattr(app, "_invoke_skill", record_skill)

    assert app._run_session(Window(), session) == 0
    assert invoked == [("map-folder", "/map-folder")]


def test_session_mouse_action_maps_bottom_panel_command_click(tmp_path, monkeypatch):
    class Window:
        def getmaxyx(self):
            return 28, 80

    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)
    suggestions = app._filtered_commands("/")
    panel = app._command_bottom_panel(suggestions, selected=0)
    assert panel is not None

    viewport = app._bottom_panel_viewport(Window(), panel, input_text="/")
    choice_y = viewport.choice_y + (1 if viewport.show_overflow_counts else 0)
    monkeypatch.setattr(
        curses,
        "getmouse",
        lambda: (0, 4, choice_y + 1, 0, curses.BUTTON1_CLICKED),
    )

    action = app._session_mouse_action(Window(), "/", suggestions)

    assert action is not None
    assert action.kind == "command"
    assert action.value == 1


def test_running_workers_render_at_bottom(tmp_path):
    class Window:
        def __init__(self):
            self.writes = []

        def getmaxyx(self):
            return 24, 80

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n]))

    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)
    window = Window()
    events = [
        {
            "type": "event_msg",
            "payload": {
                "type": "worker_event",
                "worker_id": "abc123",
                "name": "Engineer",
                "status": "running",
                "statement": "Reading files",
                "started_at": runtime_module.utc_now_iso(),
                "context_percent": 63,
            },
        }
    ]
    workers = worker_snapshots(events)

    app._draw_activity_panel(window, app._activity_items(workers, (), events, frame=12), 20)

    assert any(text == "▶" for _, _, text in window.writes)
    assert any(text == "Engineer · Reading files..." for _, _, text in window.writes)
    assert not any("abc123" in text for _, _, text in window.writes)
    assert any(
        x > 35 and text.startswith("Expand · 63% Context · ") and text.count(":") == 1
        for _, x, text in window.writes
    )


def test_running_worker_initial_thinking_renders_with_activity_dots(tmp_path):
    class Window:
        def __init__(self):
            self.writes = []

        def getmaxyx(self):
            return 24, 80

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n]))

    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)
    window = Window()
    events = [
        {
            "type": "event_msg",
            "payload": {
                "type": "worker_event",
                "worker_id": "abc123",
                "name": "Engineer",
                "status": "running",
                "statement": "thinking",
                "started_at": runtime_module.utc_now_iso(),
                "context_percent": 8,
            },
        }
    ]
    workers = worker_snapshots(events)

    app._draw_activity_panel(window, app._activity_items(workers, (), events, frame=12), 20)

    assert any("Engineer · Thinking..." in text for _, _, text in window.writes)


def test_ready_and_interrupted_workers_render_state_without_activity(tmp_path):
    class Window:
        def __init__(self):
            self.writes = []

        def getmaxyx(self):
            return 24, 80

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n]))

    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)
    window = Window()
    events = [
        {
            "type": "event_msg",
            "payload": {
                "type": "worker_event",
                "worker_id": "ready1",
                "name": "Engineer",
                "status": "ready",
                "statement": "Hidden work title",
                "context_percent": 98,
            },
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "worker_event",
                "worker_id": "stop1",
                "name": "Reviewer",
                "status": "interrupted",
                "statement": "Also hidden",
                "context_percent": 12,
            },
        },
    ]
    workers = worker_snapshots(events)

    app._draw_activity_panel(window, app._activity_items(workers, (), events, frame=12), 20)

    assert any(text == "Engineer" for _, _, text in window.writes)
    assert any(text == "Reviewer" for _, _, text in window.writes)
    assert not any("ready1" in text or "stop1" in text for _, _, text in window.writes)
    assert not any(
        "Hidden work title" in text or "Also hidden" in text
        for _, _, text in window.writes
    )
    assert any(
        x > 35 and text == "Expand · 98% Context · Ready"
        for _, x, text in window.writes
    )
    assert any(
        x > 35 and text == "Expand · 12% Context · Interrupted"
        for _, x, text in window.writes
    )


def test_expanded_worker_activity_renders_plain_statement_lines(tmp_path):
    class Window:
        def __init__(self):
            self.writes = []

        def getmaxyx(self):
            return 32, 80

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)
    window = Window()
    events = [
        {
            "type": "event_msg",
            "payload": {
                "type": "worker_event",
                "worker_id": "abc123",
                "name": "Engineer",
                "status": "working",
                "statement": "Thinking",
            },
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "worker_event",
                "worker_id": "abc123",
                "name": "Engineer",
                "status": "working",
                "statement": "Reading files",
            },
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "worker_event",
                "worker_id": "abc123",
                "name": "Engineer",
                "status": "ready",
                "statement": "",
                "response": "Implemented the fix.",
                "commands": [
                    {
                        "statement": "Listing files",
                        "command": "ls",
                        "output": "README.md",
                    }
                ],
            },
        },
    ]
    items = app._activity_items(worker_snapshots(events), (), events, frame=0)
    app._toggle_activity_item("worker:abc123")

    app._draw_activity_panel(window, items, 20)

    texts = [text for _, _, text, _ in window.writes]
    assert "Statements:" not in texts
    assert not any(text.startswith("- ") for text in texts)
    assert "Thinking" not in texts
    assert "Reading files" in texts
    assert "Listing files" in texts
    assert "Listing files · ls" not in texts
    assert "Agent is done" in texts
    assert "Implemented the fix." not in texts
    command_entry = next(entry for entry in items[0].details if entry.text == "Listing files")
    assert command_entry.detail_body == "ls"


def test_process_activity_renders_state_and_expandable_logs(tmp_path):
    class Window:
        def __init__(self):
            self.writes = []

        def getmaxyx(self):
            return 32, 80

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)
    window = Window()
    events = [
        {
            "type": "event_msg",
            "payload": {
                "type": "process_event",
                "process_id": "proc123",
                "command": "npm run dev",
                "statement": "Starting dev server",
                "status": "running",
                "started_at": runtime_module.utc_now_iso(),
                "output": "ready on http://localhost:3000",
            },
        }
    ]
    process = process_snapshots(events)[0]
    items = app._activity_items((), (process,), events, frame=12)
    app._toggle_activity_item("process:proc123")
    app._draw_activity_panel(window, items, 20)

    assert any(text == "▶" for _, _, text, _ in window.writes)
    assert any("Process Starting dev server..." in text for _, _, text, _ in window.writes)
    assert any(
        x > 50 and text.startswith("Collapse · ") and text.count(":") == 1
        for _, x, text, _ in window.writes
    )
    assert any("ready on http://localhost:3000" in text for _, _, text, _ in window.writes)
    assert not any(text == "Command:" or text == "Logs:" for _, _, text, _ in window.writes)
    assert not any("Click to kill" in text for _, _, text, _ in window.writes)
    assert any(
        action.kind == "toggle_activity_item"
        and action.text == "process:proc123"
        for actions in app._click_targets.values()
        for action in actions
    )


def test_activity_marker_shows_running_and_idle_states(tmp_path):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)

    assert app._activity_marker(active=True, frame=0) == "▶"
    assert app._activity_marker(active=True, frame=5) == "▶"
    assert app._activity_marker(active=False, frame=5) == "⏸"


def test_session_activity_panel_connects_directly_to_prompt(tmp_path):
    class Window:
        def __init__(self):
            self.writes = []

        def erase(self):
            pass

        def getmaxyx(self):
            return 32, 100

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

        def refresh(self):
            pass

    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    home.append_session_event(
        session.path,
        "process_event",
        {
            "process_id": "proc123",
            "command": "npm install",
            "statement": "Installing dependencies",
            "status": "running",
            "started_at": runtime_module.utc_now_iso(),
        },
    )
    app = AnomxCliApp(home=home, cwd=repo, use_color=False)
    window = Window()

    app._draw_session(window, session, [], "", 0, 0)
    layout = app._prompt_layout(window, "")

    assert any(y == layout.top_line and x == 2 for y, x, text, _ in window.writes if "─" in text)
    assert any(
        y == layout.top_line - 1 and "Process Installing dependencies" in text
        for y, _, text, _ in window.writes
    )
    assert any(
        y == layout.prompt_line and x == 4 and text == "›"
        for y, x, text, _ in window.writes
    )


def test_activity_detail_limits_to_ten_lines_and_scrolls(tmp_path):
    class Window:
        def __init__(self):
            self.writes = []

        def getmaxyx(self):
            return 32, 80

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)
    events = [
        {
            "type": "event_msg",
            "payload": {
                "type": "process_event",
                "process_id": "proc123",
                "command": "tail -f app.log",
                "statement": "Watching logs",
                "status": "running",
                "started_at": runtime_module.utc_now_iso(),
                "output": "\n".join(f"log {index}" for index in range(12)),
            },
        }
    ]
    item = app._activity_items((), process_snapshots(events), events, frame=0)[0]
    app._toggle_activity_item("process:proc123")

    first_window = Window()
    app._draw_activity_panel(first_window, (item,), 20)
    first_texts = [text for _, _, text, _ in first_window.writes]

    assert all(f"log {index}" in first_texts for index in range(10))
    assert "log 10" not in first_texts

    app._scroll_activity_item("process:proc123", -2)
    second_window = Window()
    app._draw_activity_panel(second_window, (item,), 20)
    second_texts = [text for _, _, text, _ in second_window.writes]

    assert "log 0" not in second_texts
    assert all(f"log {index}" in second_texts for index in range(2, 12))


def test_activity_detail_statement_click_expands_box_without_collapsing(tmp_path):
    class Window:
        def __init__(self):
            self.writes = []

        def getmaxyx(self):
            return 32, 80

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)
    events = [
        {
            "type": "event_msg",
            "payload": {
                "type": "worker_event",
                "worker_id": "abc123",
                "name": "Engineer",
                "status": "working",
                "statement": "Reading files",
            },
        }
    ]
    item = app._activity_items(worker_snapshots(events), (), events, frame=0)[0]
    app._toggle_activity_item("worker:abc123")
    first_window = Window()
    app._draw_activity_panel(first_window, (item,), 20)
    entry_action = next(
        action
        for actions in app._click_targets.values()
        for action in actions
        if action.kind == "toggle_activity_entry"
    )

    app._toggle_activity_entry(entry_action.text)
    second_window = Window()
    app._draw_activity_panel(second_window, (item,), 20)

    assert "worker:abc123" in app._expanded_activity_items
    assert any(text.startswith("╭") for _, _, text, _ in second_window.writes)
    assert any("Reading files" in text for _, _, text, _ in second_window.writes)
    statement_y = next(y for y, _, text, _ in second_window.writes if text == "Reading files")
    assert not any(
        action.kind == "toggle_activity_item" and action.text == "worker:abc123"
        for actions in (app._click_targets.get(statement_y, ()),)
        for action in actions
    )


def test_activity_mouse_wheel_scrolls_open_item_before_session(tmp_path, monkeypatch):
    class Window:
        def getmaxyx(self):
            return 32, 80

    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)
    app._click_targets = {
        22: [SessionMouseAction("scroll_activity_item", 0, "process:proc123")]
    }
    monkeypatch.setattr(
        curses,
        "getmouse",
        lambda: (0, 6, 22, 0, curses.BUTTON5_PRESSED),
    )

    action = app._session_mouse_action(Window(), "", [])

    assert action == SessionMouseAction("scroll_activity_item", -1, "process:proc123")


def test_activity_title_click_toggles_instead_of_scrolls(tmp_path, monkeypatch):
    class Window:
        def getmaxyx(self):
            return 32, 80

    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)
    app._click_targets = {
        22: [
            SessionMouseAction("toggle_activity_item", 0, "process:proc123"),
            SessionMouseAction("scroll_activity_item", 0, "process:proc123"),
        ]
    }
    monkeypatch.setattr(
        curses,
        "getmouse",
        lambda: (0, 6, 22, 0, curses.BUTTON1_PRESSED),
    )

    action = app._session_mouse_action(Window(), "", [])

    assert action == SessionMouseAction("toggle_activity_item", 0, "process:proc123")


def test_activity_expansion_is_exclusive(tmp_path):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)
    app._toggle_activity_item("worker:abc123")
    app._toggle_activity_entry("activity:abc123:statement:0:hash")

    app._toggle_activity_item("process:proc123")

    assert app._expanded_activity_items == {"process:proc123"}
    assert app._expanded_activity_entries == set()

    app._toggle_activity_item("process:proc123")

    assert app._expanded_activity_items == set()


def test_raw_mouse_sequence_toggles_activity_without_prompt_text(tmp_path):
    class Window:
        def getmaxyx(self):
            return 32, 80

    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)
    app._click_targets = {
        22: [
            SessionMouseAction("toggle_activity_item", 0, "process:proc123"),
            SessionMouseAction("scroll_activity_item", 0, "process:proc123"),
        ]
    }

    action = app._raw_mouse_action("\x1b[<0;7;23M", Window(), "", [])
    running = app._handle_running_key(
        Window(),
        AnomxHome(tmp_path / "home2").create_session(tmp_path, "openai", "gpt-5.5"),
        "\x1b[<0;7;23M",
        "",
        0,
        "",
        0.0,
    )

    assert action == SessionMouseAction("toggle_activity_item", 0, "process:proc123")
    assert running.input_text == ""
    assert "process:proc123" in app._expanded_activity_items


def test_completed_processes_do_not_render_in_session_activity(tmp_path):
    class Window:
        def __init__(self):
            self.writes = []

        def erase(self):
            pass

        def getmaxyx(self):
            return 32, 100

        def addnstr(self, y, x, text, n, attr=0):
            self.writes.append((y, x, text[:n], attr))

        def refresh(self):
            pass

    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    home.append_session_event(
        session.path,
        "process_event",
        {
            "process_id": "proc123",
            "command": "npm run build",
            "statement": "Building app",
            "status": "ended",
            "output": "done",
            "exit_code": 0,
        },
    )
    app = AnomxCliApp(home=home, cwd=repo, use_color=False)
    window = Window()

    app._draw_session(window, session, [], "", 0, 0)

    assert not any("Building app" in text for _, _, text, _ in window.writes)
    assert not any("Command npm run build" in text for _, _, text, _ in window.writes)


def test_worker_owned_command_activity_uses_owner_name(tmp_path):
    events = [
        {
            "type": "event_msg",
            "payload": {
                "type": "process_event",
                "process_id": "cmd123",
                "command": "npm run build",
                "statement": "Build assets",
                "status": "ended",
                "exit_code": 0,
                "source": "worker_command",
                "owner_name": "Engineer",
            },
        }
    ]
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), use_color=False)

    item = app._activity_items((), process_snapshots(events), events, frame=0)[0]

    assert item.title == "Engineer › Command Build assets"
    assert item.right_text == "Ready"


def test_confirm_mode_approval_rejection_only_feedbacks_model(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), repo, mode=AgentMode.CONFIRM)
    events: list[tuple[str, str]] = []

    output = runtime._execute_tool(
        "run_command",
        {"statement": "Checking Python", "command": "python3 -V"},
        RuntimeCallbacks(system_message=lambda role, message: events.append((role, message))),
    )

    payload = json.loads(output)
    assert payload["approved"] is False
    assert payload["output"] == "Command requires approval."
    assert events == []


def test_command_manager_classifies_allow_approve_forbidden(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    manager = CliToolManager(repo)

    assert manager.classify("pwd").safety == CommandSafety.ALLOW
    assert manager.classify("cat pyproject.toml").safety == CommandSafety.ALLOW
    assert manager.classify("grep TODO pyproject.toml").safety == CommandSafety.ALLOW
    assert manager.classify("find . -maxdepth 2").safety == CommandSafety.ALLOW
    assert manager.classify("find . -maxdepth 2 | sort").safety == CommandSafety.ALLOW
    assert manager.classify("git status; git log --oneline -5").safety == CommandSafety.ALLOW
    assert manager.classify("git status 2>/dev/null").safety == CommandSafety.ALLOW
    assert manager.classify("git status >/dev/null 2>&1").safety == CommandSafety.ALLOW
    assert manager.classify("git status 2>&1").safety == CommandSafety.ALLOW
    assert manager.classify("cat </dev/null").safety == CommandSafety.ALLOW
    assert manager.classify("which python3 2>/dev/null; echo ok").safety == CommandSafety.ALLOW
    assert (
        manager.classify('cat README.md 2>/dev/null || echo "No README.md"').safety
        == CommandSafety.ALLOW
    )
    curl_policy = manager.classify(
        'curl -s -o /dev/null -w "%{http_code}" http://localhost:3000/'
    )
    assert curl_policy.safety == CommandSafety.APPROVE
    assert "outside the trusted workspace" not in curl_policy.reason
    tool_check_policy = manager.classify(
        'which curl wget python3 python 2>/dev/null; echo "---"; '
        'curl --version 2>/dev/null | head -1; echo "---"; '
        "python3 --version 2>/dev/null"
    )
    assert tool_check_policy.safety == CommandSafety.APPROVE
    assert "outside the trusted workspace" not in tool_check_policy.reason
    assert manager.classify("find . -delete").safety == CommandSafety.APPROVE
    assert manager.classify("rm README.md").safety == CommandSafety.APPROVE
    assert manager.classify("cat > note.txt").safety == CommandSafety.APPROVE
    assert manager.classify("rg anomaly").safety == CommandSafety.ALLOW
    assert manager.classify(r'rg -n "/api/" src').safety == CommandSafety.ALLOW
    assert manager.classify("rg --pre sh anomaly").safety == CommandSafety.APPROVE
    assert manager.classify("sed -n 1,5p pyproject.toml").safety == CommandSafety.ALLOW
    assert (
        manager.classify(
            r"sed -n '/^class BaseModel/,/^class /p' src/anomx/agent/helpers/tool_manager.py "
            "| head -70"
        ).safety
        == CommandSafety.ALLOW
    )
    assert manager.classify("› sed -n 1,5p pyproject.toml").canonical_command == (
        "sed -n 1,5p pyproject.toml"
    )
    assert manager.classify("sed -n 1,5p /etc/passwd").safety == CommandSafety.APPROVE
    assert manager.classify("cat /etc/passwd").safety == CommandSafety.APPROVE
    assert manager.classify("reboot").safety == CommandSafety.APPROVE
    assert manager.classify("echo sudo").safety == CommandSafety.ALLOW
    assert (
        manager.classify(r'grep -rn "@\|mention\|file_picker" src').safety
        == CommandSafety.ALLOW
    )


def test_command_manager_always_allow_is_session_scoped(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("hello", encoding="utf-8")
    allowed: set[str] = set()
    manager = CliToolManager(repo, allowed)

    first = manager.run_command(
        "python -V",
        "I am checking Python",
        lambda _request: ApprovalChoice.ALWAYS_ALLOW,
    )
    second = manager.classify("python -V")
    different_parameters = manager.classify("python script.py")

    assert first.approved is True
    assert "cmd:python -V" in allowed
    assert "python -V" not in allowed
    assert second.safety == CommandSafety.ALLOW
    assert different_parameters.safety == CommandSafety.APPROVE


def test_command_manager_always_allow_trusts_command_with_saved_parameters(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "source.txt").write_text("hello", encoding="utf-8")
    allowed: set[str] = set()
    manager = CliToolManager(repo, allowed)

    first = manager.run_command(
        "cat source.txt > first.txt",
        "Writing file",
        lambda _request: ApprovalChoice.ALWAYS_ALLOW,
    )
    second = manager.classify("cat source.txt > second.txt")
    outside = manager.classify("cat ../secret.txt > second.txt")

    assert first.approved is True
    assert "cmd:cat >" in allowed
    assert "cat source.txt > first.txt" not in allowed
    assert second.safety == CommandSafety.ALLOW
    assert outside.safety == CommandSafety.FORBIDDEN


def test_command_manager_always_allow_trusts_compound_segment_parameters(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    allowed: set[str] = set()
    manager = CliToolManager(repo, allowed)

    first = manager.run_command(
        f"cd {repo} && grep hello README.md > first.txt",
        "Writing grep output",
        lambda request: (
            ApprovalChoice.ALWAYS_ALLOW
            if request.allowance_subject == "grep >"
            else ApprovalChoice.REJECT
        ),
    )
    second = manager.run_command(
        f"cd {repo} && grep hello README.md > second.txt",
        "Writing grep output again",
        None,
    )

    assert first.approved is True
    assert second.approved is True
    assert "cmd:grep >" in allowed
    assert (repo / "first.txt").read_text(encoding="utf-8") == "hello\n"
    assert (repo / "second.txt").read_text(encoding="utf-8") == "hello\n"


def test_command_manager_distinguishes_command_flags_in_allowances(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "plain.txt").write_text("plain", encoding="utf-8")
    (repo / "nested").mkdir()
    allowed: set[str] = set()
    manager = CliToolManager(repo, allowed)

    first = manager.run_command(
        "rm plain.txt",
        "Removing file",
        lambda _request: ApprovalChoice.ALWAYS_ALLOW,
    )
    recursive = manager.classify("rm -rf nested")

    assert first.approved is True
    assert "cmd:rm" in allowed
    assert "cmd:rm -rf" not in allowed
    assert recursive.safety == CommandSafety.APPROVE


def test_command_manager_approves_shell_compound_once(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    manager = CliToolManager(repo)
    requested_commands: list[str] = []

    def reject_compound(request: CommandApprovalRequest) -> ApprovalChoice:
        requested_commands.append(request.command)
        return ApprovalChoice.REJECT

    result = manager.run_command(
        "python -V && date",
        "Checking tools",
        reject_compound,
    )

    assert requested_commands == ["python -V && date"]
    assert result.approved is False
    assert result.command == "python -V && date"
    assert "The user does not allow you to do this" in result.output
    assert "Do not retry" in result.output


def test_command_manager_approves_heredoc_write_once(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    manager = CliToolManager(repo)
    requested_commands: list[str] = []
    command = (
        "cat > types.ts <<'EOF'\n"
        "export interface PollOption {\n"
        "  id: string\n"
        "  label: string | number\n"
        "  date: string\n"
        "}\n"
        "EOF"
    )

    result = manager.run_command(
        command,
        "Writing type definitions",
        lambda request: requested_commands.append(request.command) or ApprovalChoice.ALLOW,
    )

    assert requested_commands == [command]
    assert result.approved is True
    assert (repo / "types.ts").read_text(encoding="utf-8") == (
        "export interface PollOption {\n"
        "  id: string\n"
        "  label: string | number\n"
        "  date: string\n"
        "}\n"
    )


def test_command_manager_always_reject_blocks_command_family_for_session(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    rejected: set[str] = set()
    manager = CliToolManager(repo, session_rejected_commands=rejected)

    first = manager.run_command(
        "curl -s https://example.com",
        "Fetching URL",
        lambda _request: ApprovalChoice.ALWAYS_REJECT,
    )
    second = manager.classify("curl -s https://example.com/api")

    assert first.approved is False
    assert first.safety == CommandSafety.FORBIDDEN
    assert "The user does not allow you to do this" in first.output
    assert "cmd:curl -s" in rejected
    assert second.safety == CommandSafety.FORBIDDEN
    assert second.reason == "curl -s is blocked for this session by user policy."


def test_command_manager_modes_control_approval(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("hello", encoding="utf-8")

    confirm = CliToolManager(repo, mode=AgentMode.CONFIRM)
    confirm_read = confirm.run_command("cat README.md", "Reading README", None)
    confirm_execute = confirm.run_command(
        "python3 -V",
        "Checking Python",
        None,
    )

    assert confirm_read.approved is True
    assert confirm_read.output == "hello"
    assert confirm_execute.approved is False
    assert confirm_execute.safety == CommandSafety.APPROVE

    auto = CliToolManager(repo, mode=AgentMode.AUTO)
    auto_python = auto.run_command("python3 -V", "Checking Python", None)
    auto_unknown = auto.run_command("date", "Checking date", None)

    assert auto_python.approved is False
    assert auto_python.safety == CommandSafety.APPROVE
    assert auto_unknown.approved is False
    assert auto_unknown.safety == CommandSafety.APPROVE

    automatic_agent = agent_spec("automatic")
    assert automatic_agent.approval_choice_for_evaluation(
        CommandRiskEvaluation("low", "Reads tool version information.")
    ) == ApprovalChoice.ALLOW
    assert (
        automatic_agent.approval_choice_for_evaluation(
            CommandRiskEvaluation("medium", "Writes a generated file.")
        )
        is None
    )

    autonomous = CliToolManager(repo, mode=AgentMode.AUTONOMOUS)
    autonomous_unknown = autonomous.run_command("date", "Checking date", None)
    autonomous_shell = autonomous.run_command("echo $(pwd)", "Checking shell", None)
    autonomous_serious = autonomous.run_command("reboot", "Restarting host", None)

    assert autonomous_unknown.approved is True
    assert autonomous_unknown.safety == CommandSafety.ALLOW
    assert autonomous_shell.approved is True
    assert autonomous_shell.safety == CommandSafety.ALLOW
    assert autonomous_serious.approved is False
    assert autonomous_serious.safety == CommandSafety.FORBIDDEN
    assert autonomous_serious.blocked_by_mode is True
    assert "The user does not allow you to do this" in autonomous_serious.output
    assert "Do not retry" in autonomous_serious.output


def test_command_manager_handles_non_utf8_subprocess_output(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    manager = CliToolManager(repo)

    output = manager._execute_subprocess(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "sys.stdout.buffer.write(b'\\x93stdout'); "
                "sys.stderr.buffer.write(b'\\x94stderr')"
            ),
        ]
    )

    assert "\\x93stdout" in output
    assert "\\x94stderr" in output


def test_command_manager_abbreviates_middle_of_long_subprocess_output(
    tmp_path,
    monkeypatch,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(tool_manager_module, "MAX_COMMAND_OUTPUT_ROWS", 5)
    manager = CliToolManager(repo)

    output = manager._execute_subprocess(
        [
            sys.executable,
            "-c",
            "for row in range(1, 11): print(f'row {row}')",
        ]
    )

    assert output.splitlines() == [
        "row 1",
        "row 2",
        "row 3",
        "[... 5 More Rows omitted from the middle of this command output ...]",
        "row 9",
        "row 10",
    ]
