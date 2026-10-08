"""The /cowork panel driven with key presses against a simulated model."""

from __future__ import annotations

import curses
import json
import subprocess
from pathlib import Path

import pytest

from anomx.agent.app import AnomxCliApp
from anomx.agent.cowork.agent import CoworkRuntime
from anomx.agent.runtime import AgentRuntime
from anomx.agent.store import AnomxHome
from anomx.agent.ui.constants import PROJECT_COMMANDS, SLASH_COMMANDS

SOURCE = "def load():\n    return {'a': 1}\n\n\ndef show(x):\n    return x['a']\n"
ENTER, ESC, DOWN = "\n", "\x1b", curses.KEY_DOWN


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout


class Keys:
    """Window that plays a list of key presses and then reports 'no input'."""

    def __init__(self, keys):
        self._keys = list(keys)

    def get_wch(self):
        if not self._keys:
            raise curses.error("no input")
        return self._keys.pop(0)

    def nodelay(self, flag):
        pass

    def getmaxyx(self):
        return 30, 100


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    repo = root / "proj"
    (repo / "src").mkdir(parents=True)
    git(repo, "init", "-b", "main")
    git(repo, "config", "user.name", "Dev")
    git(repo, "config", "user.email", "dev@example.com")
    (repo / "src" / "data.py").write_text(SOURCE, encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-m", "initial")
    (root / ".margo").mkdir()
    vault = {"./proj/src/data.py": [{"id": "n-1", "line": 0, "title": "make dataclass"}]}
    (root / ".margo" / "local.notes.json").write_text(json.dumps(vault), encoding="utf-8")
    return root


class Screen:
    """Captures what the panel draws and which messages it shows."""

    def __init__(self, app: AnomxCliApp) -> None:
        self.overlays: list[dict] = []
        self.messages: list[tuple[str, str]] = []
        app._draw_overlay = self._draw  # type: ignore[method-assign]
        app._message = lambda _w, title, text: self.messages.append((title, text))  # type: ignore[method-assign]

    def _draw(self, _stdscr, **kwargs):
        self.overlays.append(kwargs)
        return (0, 0, 80, 20)

    def last_choices(self) -> list[tuple[str, str]]:
        choices = next(o["choices"] for o in reversed(self.overlays) if o.get("choices"))
        return [(c.label, c.detail) for c in choices]


@pytest.fixture
def app(tmp_path: Path, workspace: Path) -> AnomxCliApp:
    return AnomxCliApp(home=AnomxHome(tmp_path / "home"), cwd=workspace / "proj" / "src")


@pytest.fixture
def fake_agent(monkeypatch):
    def backend_response(self, session_path, callbacks=None, **_kwargs):
        self._execute_tool(
            "edit_file",
            {"statement": "s", "path": "src/data.py", "old_text": "x['a']", "new_text": "x.a"},
            callbacks,
            session_path,
        )
        return "Switched to attribute access."

    monkeypatch.setattr(AgentRuntime, "backend_response", backend_response)


def _panel(app: AnomxCliApp, workspace: Path):
    return app._cowork_panels[workspace]


def _start_and_finish(app: AnomxCliApp, workspace: Path) -> None:
    app._run_cowork_panel(Keys([ENTER, ENTER, ESC]))  # open note, "Start agent", close
    _panel(app, workspace).wait("n-1")


def test_command_is_registered_for_session_and_project():
    command = next(item for item in SLASH_COMMANDS if item.command == "/cowork")

    assert command.in_project and not command.while_running
    assert "/cowork" in PROJECT_COMMANDS


def test_handle_command_opens_the_panel(app, monkeypatch):
    opened: list[object] = []
    monkeypatch.setattr(app, "_run_cowork_panel", lambda stdscr: opened.append(stdscr))

    assert app._handle_command(Keys([]), "/cowork", object()) is None

    assert len(opened) == 1


def test_without_a_margo_vault_the_panel_explains_what_to_do(tmp_path):
    lonely = AnomxCliApp(home=AnomxHome(tmp_path / "home"), cwd=tmp_path / "nowhere")
    screen = Screen(lonely)

    lonely._run_cowork_panel(Keys([]))

    assert "No Margo notes found" in screen.messages[0][1]


def test_panel_lists_notes_with_their_status(app):
    screen = Screen(app)

    app._run_cowork_panel(Keys([ESC]))

    assert screen.last_choices()[0] == ("proj/src/data.py:1  make dataclass", "[new]")
    assert screen.last_choices()[-1][0] == "Follow my notes in the editor: off"
    assert screen.overlays[0]["subtitle"] == "1 notes"


def test_start_runs_the_agent_and_the_row_turns_ready(app, workspace, fake_agent):
    screen = Screen(app)

    _start_and_finish(app, workspace)
    app._run_cowork_panel(Keys([ESC]))

    label, detail = screen.last_choices()[0]
    assert label == "proj/src/data.py:1  make dataclass"
    assert detail.startswith("[ready]") and "attribute access" in detail


def test_diff_viewer_shows_the_agents_change(app, workspace, fake_agent):
    screen = Screen(app)
    _start_and_finish(app, workspace)

    # open note -> "Show changes" (second action) -> close viewer -> close menu
    app._run_cowork_panel(Keys([ENTER, DOWN, ENTER, ESC, ESC]))

    viewer = next(o for o in screen.overlays if o["title"].startswith("Changes"))
    assert "+    return x.a" in viewer["body_lines"]


def test_follow_switches_the_tree_and_back_returns(app, workspace, fake_agent):
    screen = Screen(app)
    _start_and_finish(app, workspace)
    data = workspace / "proj" / "src" / "data.py"

    # open note -> "Show in editor" (third action) -> message -> close panel
    app._run_cowork_panel(Keys([ENTER, DOWN, DOWN, ENTER, ESC]))
    assert "x.a" in data.read_text("utf-8")
    assert any("agent's version" in text for _title, text in screen.messages)

    app._run_cowork_panel(Keys([ESC]))
    assert screen.last_choices()[0][0] == "Back to main"  # offered while following

    app._run_cowork_panel(Keys([ENTER, ESC]))  # choose "Back to main"
    assert data.read_text("utf-8") == SOURCE
    assert git(workspace / "proj", "symbolic-ref", "--short", "HEAD").strip() == "main"


def test_merge_needs_confirmation_and_then_updates_the_branch(app, workspace, fake_agent):
    screen = Screen(app)
    _start_and_finish(app, workspace)
    data = workspace / "proj" / "src" / "data.py"

    # open note -> "Merge into main" (fourth action) -> confirm menu: Cancel
    app._run_cowork_panel(Keys([ENTER, DOWN, DOWN, DOWN, ENTER, ENTER, ESC]))
    assert data.read_text("utf-8") == SOURCE  # cancelled: nothing merged

    # open note -> Merge -> confirm "Merge into main" (second entry)
    app._run_cowork_panel(Keys([ENTER, DOWN, DOWN, DOWN, ENTER, DOWN, ENTER, ESC]))

    assert "x.a" in data.read_text("utf-8")
    assert ("Cowork", "Merged into main.") in screen.messages
    assert any("Merge into main?" in (o.get("subtitle") or "") for o in screen.overlays)
    assert "Changed: src/data.py" in next(
        o["subtitle"] for o in screen.overlays if "Merge into main?" in (o.get("subtitle") or "")
    )


def test_discard_asks_first_and_frees_the_note(app, workspace, fake_agent):
    screen = Screen(app)
    _start_and_finish(app, workspace)

    # open note -> actions: chat, diff, follow, merge, run again, discard (index 5) -> confirm
    app._run_cowork_panel(Keys([ENTER, *[DOWN] * 5, ENTER, DOWN, ENTER, ESC]))

    app._run_cowork_panel(Keys([ESC]))
    assert screen.last_choices()[0][1] == "[new]"


def test_start_error_is_shown_instead_of_crashing(app, workspace, fake_agent):
    screen = Screen(app)
    (workspace / "proj" / "src" / "data.py").write_text(SOURCE + "# wip\n", encoding="utf-8")

    app._run_cowork_panel(Keys([ENTER, ENTER, ESC]))

    assert screen.messages and screen.messages[0][0] == "Cannot start"
    assert "uncommitted" in screen.messages[0][1]


# -- chat per note --------------------------------------------------------------------


def test_discuss_first_returns_the_chat_session_for_the_screen_to_switch_to(app, workspace):
    Screen(app)

    # open note -> "Discuss first" (second action)
    record = app._run_cowork_panel(Keys([ENTER, DOWN, ENTER]))

    assert record is not None and record.title == "make dataclass"
    panel = _panel(app, workspace)
    task = next(row.task for row in panel.rows() if row.note_id == "n-1")
    assert Path(record.cwd) == Path(task.worktree).resolve()
    assert task.session_path == str(record.path)


def test_the_slash_command_hands_the_chat_to_the_session_switcher(app):
    Screen(app)

    result = app._handle_command(Keys([ENTER, DOWN, ENTER]), "/cowork", object())

    assert result is not None and result.title == "make dataclass"


def test_only_chats_of_notes_get_the_cowork_runtime(app, workspace):
    Screen(app)
    chat = app._run_cowork_panel(Keys([ENTER, DOWN, ENTER]))
    ordinary = app.home.create_session(workspace / "proj", "openai", "gpt-5.5")

    assert isinstance(app._cowork_runtime_for(chat), CoworkRuntime)
    assert app._cowork_runtime_for(ordinary) is None


def test_a_chat_turn_runs_in_the_worktree_and_is_committed(app, workspace, monkeypatch):
    Screen(app)
    chat = app._run_cowork_panel(Keys([ENTER, DOWN, ENTER]))
    seen: dict[str, object] = {}

    def backend_response(self, session_path, callbacks=None, **_kwargs):
        seen["cwd"] = self.cwd
        self._execute_tool(
            "edit_file",
            {"statement": "s", "path": "src/data.py", "old_text": "x['a']", "new_text": "x.a"},
            callbacks,
            session_path,
        )
        return "Done."

    monkeypatch.setattr(AgentRuntime, "backend_response", backend_response)

    turn = app._start_session_turn(chat)
    turn.worker.join(10)

    assert isinstance(turn.runtime, CoworkRuntime)
    row = next(r for r in _panel(app, workspace).rows() if r.note_id == "n-1")
    assert seen["cwd"] == Path(row.task.worktree).resolve()
    assert row.status == "ready" and "x.a" in "\n".join(_panel(app, workspace).diff_lines(row))
    assert (workspace / "proj" / "src" / "data.py").read_text("utf-8") == SOURCE  # user tree intact


def test_open_chat_resumes_the_same_session_and_is_refused_while_the_agent_runs(
    app, workspace, fake_agent
):
    Screen(app)
    first = app._run_cowork_panel(Keys([ENTER, DOWN, ENTER]))
    panel = _panel(app, workspace)
    row = next(r for r in panel.rows() if r.note_id == "n-1")

    again, problem = panel.open_chat(row, discuss=False)

    assert problem is None and again is not None and again.path == first.path
    assert [a.value for a in panel.actions(row)][:2] == ["start", "chat"]


# -- following the clicked note -------------------------------------------------------


def _put_cursor(app, workspace: Path, line: int, *, dirty: bool = False) -> None:
    from datetime import UTC, datetime

    target = app.home.root / "editor-context.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "schema": 1,
                "updatedAt": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "workspaceFolders": [str(workspace)],
                "activeFile": str(workspace / "proj" / "src" / "data.py"),
                "isDirty": dirty,
                "isActiveEditor": True,
                "cursor": {"line": line, "character": 1},
                "selection": {"startLine": line, "endLine": line},
                "visibleRange": {"startLine": 1, "endLine": 50},
            }
        ),
        encoding="utf-8",
    )


def _enable_follow(app, workspace: Path, monkeypatch):
    from anomx.agent import app as app_module
    from anomx.agent.ui.models import SessionViewState

    monkeypatch.setattr(app_module, "COWORK_POLL_SECONDS", 0.0)
    screen = Screen(app)
    _put_cursor(app, workspace, 1)
    app._run_cowork_panel(Keys([*[DOWN] * 1, ENTER, ESC]))  # toggle is the last entry
    assert app._cowork_follower is not None
    return screen, SessionViewState()


def _step(app, session, view):
    app._next_cowork_poll = 0.0
    return app._cowork_follow_step(Keys([]), session, view)


def test_toggle_needs_the_extension_and_can_be_turned_off_again(app, workspace):
    screen = Screen(app)

    app._run_cowork_panel(Keys([DOWN, ENTER, ESC]))  # no editor-context file yet
    assert app._cowork_follower is None
    assert "Editor Context extension" in screen.messages[-1][1]

    _put_cursor(app, workspace, 1)
    app._run_cowork_panel(Keys([DOWN, ENTER, ESC]))
    assert app._cowork_follower is not None
    assert screen.last_choices()[-1][0] == "Follow my notes in the editor: on"

    app._run_cowork_panel(Keys([DOWN, ENTER, ESC]))
    assert app._cowork_follower is None


def test_clicking_a_note_switches_to_its_chat_after_two_polls(app, workspace, monkeypatch):
    _screen, view = _enable_follow(app, workspace, monkeypatch)
    ordinary = app.home.create_session(workspace / "proj", "openai", "gpt-5.5")

    assert _step(app, ordinary, view) is ordinary  # first sighting only
    chat = _step(app, ordinary, view)

    assert chat.path != ordinary.path and chat.title == "make dataclass"
    assert Path(chat.cwd).name.startswith("n-1")
    assert view.notice.startswith("Cowork · make dataclass")
    assert _step(app, chat, view) is chat  # same note: nothing to do


def test_a_problem_is_shown_and_the_current_chat_stays(app, workspace, monkeypatch):
    _screen, view = _enable_follow(app, workspace, monkeypatch)
    (workspace / "proj" / "src" / "data.py").write_text(SOURCE + "# wip\n", encoding="utf-8")
    ordinary = app.home.create_session(workspace / "proj", "openai", "gpt-5.5")

    _step(app, ordinary, view)
    result = _step(app, ordinary, view)

    assert result is ordinary
    assert "uncommitted" in view.notice


def test_without_a_follower_nothing_happens(app, workspace):
    from anomx.agent.ui.models import SessionViewState

    ordinary = app.home.create_session(workspace / "proj", "openai", "gpt-5.5")
    view = SessionViewState()

    assert _step(app, ordinary, view) is ordinary and view.notice == ""


def test_the_session_loop_switches_chat_when_a_note_is_clicked(app, workspace, monkeypatch):
    class Stop(Exception):
        pass

    _screen, _view = _enable_follow(app, workspace, monkeypatch)
    ordinary = app.home.create_session(workspace / "proj", "openai", "gpt-5.5")
    shown: list[str] = []

    def fake_draw(_stdscr, session, *_args, **_kwargs):
        shown.append(session.title)
        if session.title == "make dataclass":
            raise Stop

        class Viewport:
            scroll = 0

        return Viewport()

    monkeypatch.setattr(app, "_draw_session", fake_draw)

    with pytest.raises(Stop):
        app._run_session(Keys(["a", "b", "c"]), ordinary)  # keys only wake the loop up

    assert shown[0] != "make dataclass" and shown[-1] == "make dataclass"


def test_finished_turns_of_other_chats_are_wrapped_up_in_the_background(
    app, workspace, monkeypatch
):
    Screen(app)
    chat = app._run_cowork_panel(Keys([ENTER, DOWN, ENTER]))
    monkeypatch.setattr(AgentRuntime, "backend_response", lambda self, *a, **k: "Done.")
    turn = app._start_session_turn(chat)
    turn.worker.join(10)
    elsewhere = app.home.create_session(workspace / "proj", "openai", "gpt-5.5")

    app._drain_background_session_turns(Keys([]), chat)  # the chat is on screen: untouched
    assert turn.completed is False
    app._drain_background_session_turns(Keys([]), elsewhere)

    assert turn.completed is True and not app._active_session_turns


def test_notices_expire_on_their_own():
    from anomx.agent.ui.models import SessionViewState

    view = SessionViewState()
    view.show_notice("hello", now=100.0, seconds=6.0)

    view.expire_notice(105.0)
    assert view.notice == "hello"
    view.expire_notice(107.0)
    assert view.notice == ""
