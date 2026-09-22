from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import cast

import pytest

import anomx.agent.file_reference_search as file_search_module
from anomx.agent.app import AnomxCliApp
from anomx.agent.file_reference_search import (
    FileReferenceSearchClient,
    FileReferenceSearchResult,
)
from anomx.agent.runtime_process import RuntimeProcessClient
from anomx.agent.store import AnomxHome, ProjectRecord
from anomx.agent.ui.models import CommandSpec, CursesWindow


def test_runtime_process_is_distinct_and_relays_events(tmp_path: Path) -> None:
    home = AnomxHome(tmp_path / "home")
    session = home.create_session(tmp_path, provider="missing", model="missing")
    home.append_session_event(session.path, "user_message", {"message": "hello"})
    statuses: list[str] = []
    runtime = RuntimeProcessClient(home, tmp_path, workspace_root=tmp_path)

    try:
        runtime.start()
        assert runtime.wait_until_ready(timeout=20)
        assert runtime.process_id is not None
        assert runtime.process_id != os.getpid()
        assert runtime.conversation_messages(session.path) == [
            {"role": "user", "content": "hello"}
        ]
        assert runtime.init_python_sandbox(status_callback=statuses.append)
        assert statuses == ["Preparing Python sandbox", "Python sandbox ready"]
        assert runtime.is_sandbox_active() is False
    finally:
        runtime.close()


def test_cli_import_does_not_eagerly_load_agent_or_components() -> None:
    package_root = Path(__file__).resolve().parents[1] / "src"
    env = {**os.environ, "PYTHONPATH": str(package_root)}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import anomx.cli; "
                "print('anomx.agent' in sys.modules, 'anomx.components' in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.stdout.strip() == "False False"


def test_cli_turn_runs_through_isolated_runtime(tmp_path: Path) -> None:
    home = AnomxHome(tmp_path / "home")
    config = home.load_config()
    config.update({"onboarding_complete": True, "provider": "missing", "model": "missing"})
    home.save_config(config)
    session = home.create_session(tmp_path, provider="missing", model="missing")
    home.append_session_event(session.path, "user_message", {"message": "hello"})
    app = AnomxCliApp(home=home, cwd=tmp_path, isolate_runtime=True)

    try:
        turn = app._start_session_turn(session)
        turn.worker.join(timeout=20)

        assert not turn.worker.is_alive()
        assert turn.result == {"response": "missing/missing backend is unavailable."}
        assert isinstance(turn.runtime, RuntimeProcessClient)
        assert turn.runtime.process_id != os.getpid()
    finally:
        app._shutdown_active_session_turns()
        app.runtime.shutdown()


def test_isolated_cli_enters_project_without_startup_screen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Window:
        def erase(self) -> None:
            return None

        def refresh(self) -> None:
            return None

    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    home.trust_repo(repo)
    project = home.save_project(repo, "Signals")
    config = home.load_config()
    config["onboarding_complete"] = True
    home.save_config(config)
    app = AnomxCliApp(home=home, cwd=repo, isolate_runtime=True)
    opened: list[str] = []

    monkeypatch.setattr(app, "_configure_terminal", lambda _window: None)
    monkeypatch.setattr(
        app,
        "_run_startup_loading",
        lambda _window: pytest.fail("startup loading screen must not run"),
    )
    monkeypatch.setattr(
        app,
        "_run_version_check",
        lambda _window: pytest.fail("startup update screen must not run"),
    )
    monkeypatch.setattr(app, "_cleanup_stale_project_sessions", lambda _project: None)
    monkeypatch.setattr(app, "_handle_sandbox_check", lambda _window, _project: True)

    def open_project(_window: object, current: ProjectRecord) -> int:
        opened.append(current.name)
        return 0

    monkeypatch.setattr(app, "_run_project", open_project)
    monkeypatch.setattr(app, "_shutdown_sandbox_containers", lambda: None)
    monkeypatch.setattr(app.runtime, "shutdown", lambda _path=None: None)

    assert app._run(cast(CursesWindow, Window())) == 0
    assert opened == [project.name]


def test_file_reference_search_runs_in_background_process_and_caches(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    (repo / "src" / "signals").mkdir(parents=True)
    (repo / "src" / "signals" / "target_detector.py").write_text("", encoding="utf-8")
    client = FileReferenceSearchClient(repo, tmp_path / "search-index.json")

    try:
        client.start()
        assert client.process_id is not None
        assert client.process_id != os.getpid()

        started_at = time.monotonic()
        client.suggestions("target")
        assert time.monotonic() - started_at < 0.1

        deadline = time.monotonic() + 10
        results: tuple[FileReferenceSearchResult, ...] = ()
        while time.monotonic() < deadline:
            results = client.suggestions("target")
            if results:
                break
            time.sleep(0.02)

        assert [result.label for result in results] == [
            "src/signals/target_detector.py"
        ]
        assert client.suggestions("target") == results
        assert (tmp_path / "search-index.json").exists()
    finally:
        client.close()


def test_slash_command_menu_token_works_after_existing_text(tmp_path: Path) -> None:
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), cwd=tmp_path)
    prompt = "Please inspect this /mo"
    token = app._active_command_token(prompt, len(prompt))

    assert token == (20, 23, "/mo")
    assert token is not None
    suggestions = app._filtered_commands(token[2])
    assert [suggestion.command for suggestion in suggestions] == ["/model"]

    updated, cursor = app._remove_command_token(
        prompt,
        token,
        [],
    )
    assert updated == "Please inspect this"
    assert cursor == len(updated)

    attached_prompt = "Please inspect this/"
    attached_token = app._active_command_token(attached_prompt, len(attached_prompt))
    assert attached_token == (19, 20, "/")
    assert attached_token is not None
    attached, attached_cursor = app._remove_command_token(
        attached_prompt,
        attached_token,
        [],
    )
    assert attached == "Please inspect this"
    assert attached_cursor == len(attached)
    assert app._active_command_token("Read @src/", len("Read @src/")) is None

    session = app.home.create_session(tmp_path, provider="missing", model="missing")
    running_result = app._handle_running_key(
        cast(CursesWindow, object()),
        session,
        "\n",
        prompt,
        len(prompt),
        "",
        0.0,
        [CommandSpec("/model", "Choose model")],
        0,
        [],
    )
    assert running_result.input_text == "Please inspect this"
    assert running_result.command == "/model"


def test_empty_file_results_show_searching_until_worker_finishes(tmp_path: Path) -> None:
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), cwd=tmp_path)

    searching = app._file_reference_bottom_panel([], 0, active=True, searching=True)
    finished = app._file_reference_bottom_panel([], 0, active=True, searching=False)

    assert searching is not None
    assert searching.subtitle == "Searching…"
    assert finished is not None
    assert finished.subtitle == "No matches found"


def test_background_file_index_does_not_drop_deep_results_after_5000_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    filenames = [f"entry-{index:05d}.txt" for index in range(5_100)]
    filenames.append("flow-workspace.js")
    monkeypatch.setattr(
        "anomx.agent.file_reference_search.os.walk",
        lambda _root: [(str(tmp_path), [], filenames)],
    )
    index = file_search_module._WorkspaceFileIndex(
        tmp_path,
        tmp_path / "index.json",
    )

    assert "flow-workspace.js" in index._scan()
