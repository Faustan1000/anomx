"""The slash-command registry is the single source for scope and availability."""

from __future__ import annotations

from anomx.agent.app import AnomxCliApp
from anomx.agent.store import AnomxHome
from anomx.agent.ui.constants import (
    COMMANDS,
    PLATFORM_COMMANDS,
    PROJECT_COMMANDS,
    RUNNING_SESSION_COMMANDS,
    SLASH_COMMANDS,
)


def _app(tmp_path):
    return AnomxCliApp(home=AnomxHome(tmp_path / "home"), cwd=tmp_path)


def test_derived_sets_follow_the_registry():
    assert [spec.command for spec in COMMANDS] == [item.command for item in SLASH_COMMANDS]
    assert "/rename" not in PROJECT_COMMANDS
    assert {"/new", "/config", "/model", "/effort", "/feedback", "/exit"} == PROJECT_COMMANDS
    assert {"/config", "/model", "/effort"} == RUNNING_SESSION_COMMANDS
    assert {"/feedback"} == PLATFORM_COMMANDS


def test_every_project_and_running_command_is_a_known_builtin():
    known = {item.command for item in SLASH_COMMANDS}
    assert known >= PROJECT_COMMANDS
    assert known >= RUNNING_SESSION_COMMANDS
    assert known >= PLATFORM_COMMANDS


def test_feedback_is_hidden_until_a_platform_is_connected(tmp_path, monkeypatch):
    app = _app(tmp_path)
    monkeypatch.setattr(app.home, "has_platform_connection", lambda: False)
    hidden = {spec.command for spec in app._command_specs()}
    hidden_project = {spec.command for spec in app._project_command_specs()}

    monkeypatch.setattr(app.home, "has_platform_connection", lambda: True)
    shown = {spec.command for spec in app._command_specs()}
    shown_project = {spec.command for spec in app._project_command_specs()}

    assert "/feedback" not in hidden and "/feedback" not in hidden_project
    assert "/feedback" in shown and "/feedback" in shown_project


def test_project_screen_offers_only_project_commands(tmp_path):
    app = _app(tmp_path)
    builtin = {item.command for item in SLASH_COMMANDS}

    project_builtin = {
        spec.command for spec in app._project_command_specs() if spec.command in builtin
    }
    session_builtin = {spec.command for spec in app._command_specs() if spec.command in builtin}

    assert "/rename" in session_builtin
    assert "/rename" not in project_builtin
    assert project_builtin <= session_builtin


def test_running_session_offers_only_commands_flagged_while_running(tmp_path):
    offered = {spec.command for spec in _app(tmp_path)._running_command_specs()}

    assert offered == RUNNING_SESSION_COMMANDS
