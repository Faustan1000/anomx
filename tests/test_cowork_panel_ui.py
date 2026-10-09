"""The /cowork panel driven with key presses against a simulated model."""

from __future__ import annotations

import curses
import json
import subprocess
from pathlib import Path

import pytest
from cowork_notes import note_entry, write_notes

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

    write_notes(root, note_entry("n-1", "proj/src/data.py", 1, "make dataclass"))
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


@pytest.fixture(autouse=True)
def _quiet_backend(monkeypatch):
    """Opening a chat now sends the note; never let a test reach a real model."""

    monkeypatch.setattr(AgentRuntime, "backend_response", lambda self, *a, **k: "ok")


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


def test_without_notes_the_panel_explains_what_to_do(tmp_path):
    lonely = AnomxCliApp(home=AnomxHome(tmp_path / "home"), cwd=tmp_path / "nowhere")
    screen = Screen(lonely)

    lonely._run_cowork_panel(Keys([]))

    assert "No notes found" in screen.messages[0][1]


def test_panel_lists_notes_with_their_status(app):
    screen = Screen(app)

    app._run_cowork_panel(Keys([ESC]))

    assert screen.last_choices()[0] == ("proj/src/data.py:1  make dataclass", "[new]")
    assert screen.last_choices()[-2][0] == "Follow my notes in the editor: off"
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

    # open note -> "Show changes" -> Esc: viewer -> note menu -> list -> close (one level each)
    app._run_cowork_panel(Keys([ENTER, DOWN, ENTER, ESC, ESC, ESC]))

    viewer = next(o for o in screen.overlays if o["title"].startswith("Changes"))
    assert "+    return x.a" in viewer["body_lines"]


def test_follow_switches_the_tree_and_back_returns(app, workspace, fake_agent):
    screen = Screen(app)
    _start_and_finish(app, workspace)
    data = workspace / "proj" / "src" / "data.py"

    # open note -> "Show in editor" (fourth action) -> message -> close panel
    app._run_cowork_panel(Keys([ENTER, DOWN, DOWN, DOWN, ENTER, ESC, ESC]))
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

    # open note -> "Merge into main" (fifth action) -> confirm menu: Cancel
    app._run_cowork_panel(Keys([ENTER, *[DOWN] * 4, ENTER, ENTER, ESC]))
    assert data.read_text("utf-8") == SOURCE  # cancelled: nothing merged

    # open note -> Merge -> confirm "Merge into main" (second entry)
    app._run_cowork_panel(Keys([ENTER, *[DOWN] * 4, ENTER, DOWN, ENTER, ESC]))

    # staged for review: the files are in the working tree, the merge waits for the commit
    assert "x.a" in data.read_text("utf-8")
    assert git(workspace / "proj", "rev-parse", "-q", "--verify", "MERGE_HEAD").strip()
    assert "src/data.py" in git(workspace / "proj", "diff", "--cached", "--name-only")
    assert any("Merge in progress" in text for _title, text in screen.messages)
    question = next(
        o["subtitle"] for o in screen.overlays if "for review?" in (o.get("subtitle") or "")
    )
    assert "Stage the changes in main for review?" in question
    assert "Changed: src/data.py" in question


def test_discard_asks_first_and_frees_the_note(app, workspace, fake_agent):
    screen = Screen(app)
    _start_and_finish(app, workspace)

    # actions: chat, diff, compare, follow, merge, merge now, run again, discard (index 7)
    app._run_cowork_panel(Keys([ENTER, *[DOWN] * 7, ENTER, DOWN, ENTER, ESC]))

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

    app._run_cowork_panel(Keys([ENTER, DOWN, ENTER]))  # opens the chat and sends the note
    (turn,) = app._active_session_turns.values()
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
    assert "anomx --install-vscode" in screen.messages[-1][1]

    _put_cursor(app, workspace, 1)
    app._run_cowork_panel(Keys([DOWN, ENTER, ESC]))
    assert app._cowork_follower is not None
    assert screen.last_choices()[-2][0] == "Follow my notes in the editor: on"

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


# -- the note must actually be sent ---------------------------------------------------


@pytest.fixture
def agent_calls(monkeypatch):
    calls: list[str] = []

    def backend_response(self, session_path, callbacks=None, **_kwargs):
        calls.append(str(session_path))
        return "Here are the options."

    monkeypatch.setattr(AgentRuntime, "backend_response", backend_response)
    return calls


def test_opening_a_chat_from_the_panel_sends_the_note_right_away(
    app, workspace, agent_calls
):
    Screen(app)

    chat = app._run_cowork_panel(Keys([ENTER, DOWN, ENTER]))  # "Discuss first"

    (turn,) = app._active_session_turns.values()
    turn.worker.join(10)
    assert agent_calls == [str(chat.path)]


def test_reopening_an_answered_chat_does_not_send_anything_again(app, workspace, agent_calls):
    Screen(app)
    chat = app._run_cowork_panel(Keys([ENTER, DOWN, ENTER]))
    (turn,) = app._active_session_turns.values()
    turn.worker.join(10)
    app._complete_session_turn(Keys([]), turn, render_final=False)  # stores the answer

    again = app._run_cowork_panel(Keys([ENTER, ENTER]))  # "Open chat" is the first action

    assert again is not None and again.path == chat.path
    assert agent_calls == [str(chat.path)]  # still just the first turn
    assert not app._active_session_turns


def test_an_empty_enter_sends_the_unanswered_note(app, workspace, agent_calls, monkeypatch):
    class Stop(Exception):
        pass

    panel = app._cowork_workspace()
    row = next(r for r in panel.rows() if r.note_id == "n-1")
    chat, problem = panel.open_chat(row, discuss=True)  # as the follow mode does: not sent
    assert problem is None and chat is not None and not agent_calls

    def fake_draw(_stdscr, session, *_args, **_kwargs):
        if app._active_session_turns:
            raise Stop

        class Viewport:
            scroll = 0

        return Viewport()

    monkeypatch.setattr(app, "_draw_session", fake_draw)

    with pytest.raises(Stop):
        app._run_session(Keys(["\n"]), chat)

    (turn,) = app._active_session_turns.values()
    turn.worker.join(10)
    assert agent_calls == [str(chat.path)]


def test_a_normal_session_is_not_touched_by_an_empty_enter(app, workspace, agent_calls):
    ordinary = app.home.create_session(workspace / "proj", "openai", "gpt-5.5")

    assert app._cowork_send_pending(ordinary) is False
    assert not app._active_session_turns and not agent_calls


def test_the_follow_hint_tells_how_to_send_a_fresh_note(app, workspace, monkeypatch):
    _screen, view = _enable_follow(app, workspace, monkeypatch)
    ordinary = app.home.create_session(workspace / "proj", "openai", "gpt-5.5")

    _step(app, ordinary, view)
    _step(app, ordinary, view)

    assert "press Enter on an empty prompt" in view.notice


# -- side-by-side comparison ----------------------------------------------------------


@pytest.fixture
def ready_task(app, workspace, monkeypatch):
    """A finished task whose agent edited one file and added another."""

    def backend_response(self, session_path, callbacks=None, **_kwargs):
        edits = [
            ("src/data.py", "x['a']", "x.a"),
            ("src/new_module.py", "", "VALUE = 1\n"),
        ]
        for path, old, new in edits:
            self._execute_tool(
                "edit_file",
                {"statement": "s", "path": path, "old_text": old, "new_text": new},
                callbacks,
                session_path,
            )
        return "Done."

    monkeypatch.setattr(AgentRuntime, "backend_response", backend_response)
    Screen(app)
    _start_and_finish(app, workspace)
    return _panel(app, workspace)


def test_comparison_files_pair_the_original_with_the_agents_version(ready_task):
    row = next(r for r in ready_task.rows() if r.note_id == "n-1")

    files = {item.file: item for item in ready_task.comparison_files(row)}

    assert set(files) == {"src/data.py", "src/new_module.py"}
    changed = files["src/data.py"]
    assert changed.status == "M" and changed.label == "changed  src/data.py"
    assert changed.original.read_text("utf-8") == SOURCE  # what the repo had
    assert "x.a" in changed.agent.read_text("utf-8")  # the agent's real file
    assert Path(row.task.worktree) in changed.agent.parents
    added = files["src/new_module.py"]
    assert added.status == "A" and added.original.read_text("utf-8") == ""
    assert added.agent.read_text("utf-8") == "VALUE = 1\n"
    # the user's repository is untouched by preparing the comparison
    assert (workspace_path(ready_task) / "proj" / "src" / "data.py").read_text("utf-8") == SOURCE


def _fake_vscode(monkeypatch, *, code="/usr/bin/code", returncode=0):
    """Replace only the cowork view's view of ``shutil``/``subprocess`` (git keeps working)."""

    import subprocess as real_subprocess
    from types import SimpleNamespace

    from anomx.agent.ui.views import cowork as cowork_view

    calls: list[list[str]] = []

    def run(command, **_kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=returncode)

    monkeypatch.setattr(cowork_view, "shutil", SimpleNamespace(which=lambda _name: code))
    monkeypatch.setattr(
        cowork_view,
        "subprocess",
        SimpleNamespace(run=run, SubprocessError=real_subprocess.SubprocessError),
    )
    monkeypatch.setenv("VSCODE_IPC_HOOK_CLI", "/tmp/hook.sock")
    return calls


def workspace_path(panel) -> Path:
    return panel.notes_workspace


def test_compare_opens_vscodes_diff_with_original_left_and_agent_right(
    app, workspace, ready_task, monkeypatch
):
    calls = _fake_vscode(monkeypatch)
    screen = Screen(app)

    # open note -> "Compare in VS Code" (third action) -> "Open all 2 comparisons"
    app._run_cowork_panel(Keys([ENTER, DOWN, DOWN, ENTER, ENTER, ESC, ESC]))

    assert len(calls) == 2
    for command in calls:
        assert command[:3] == ["/usr/bin/code", "--reuse-window", "--diff"]
        assert "/original/" in command[3] and "/worktrees/" in command[4]
    assert any("Opened 2 comparison(s)" in text for _title, text in screen.messages)


def test_compare_one_file_from_the_picker(app, workspace, ready_task, monkeypatch):
    calls = _fake_vscode(monkeypatch)
    Screen(app)

    # picker entries: all, then the files; take the first file
    app._run_cowork_panel(Keys([ENTER, DOWN, DOWN, ENTER, DOWN, ENTER, ESC, ESC]))

    assert len(calls) == 1 and calls[0][3].endswith("/original/src/data.py")


def test_compare_explains_how_to_continue_when_vscode_cannot_be_steered(
    app, workspace, ready_task, monkeypatch
):
    _fake_vscode(monkeypatch, code=None)
    screen = Screen(app)

    app._run_cowork_panel(Keys([ENTER, DOWN, DOWN, ENTER, ESC, ESC]))

    title, text = screen.messages[-1]
    assert title == "Cannot open VS Code" and "code --diff" in text and "/original/" in text


def test_compare_reports_a_failed_vscode_call(app, workspace, ready_task, monkeypatch):
    _fake_vscode(monkeypatch, returncode=1)
    screen = Screen(app)

    app._run_cowork_panel(Keys([ENTER, DOWN, DOWN, ENTER, ENTER, ESC, ESC]))

    assert screen.messages[-1][0] == "Cannot open VS Code"


# -- Esc goes up exactly one level ----------------------------------------------------


def test_closing_the_diff_viewer_returns_to_the_note_menu_not_the_list(
    app, workspace, ready_task
):
    screen = Screen(app)

    app._run_cowork_panel(Keys([ENTER, DOWN, ENTER, ESC, ESC, ESC]))

    titles = [o["title"] for o in screen.overlays]
    viewer = next(i for i, title in enumerate(titles) if title.startswith("Changes"))
    after_viewer = screen.overlays[viewer + 1]
    assert after_viewer["title"] == "Cowork"
    assert "proj/src/data.py:1" in after_viewer["subtitle"]  # the note menu, not the list
    assert [c.label for c in after_viewer["choices"]][:2] == ["Open chat", "Show changes"]


def test_the_selection_in_the_note_menu_is_kept_after_a_viewing_action(
    app, workspace, ready_task
):
    screen = Screen(app)

    app._run_cowork_panel(Keys([ENTER, DOWN, ENTER, ESC, ESC, ESC]))

    note_menus = [o for o in screen.overlays if o.get("choices") and "Show changes" in [
        c.label for c in o["choices"]
    ]]
    assert note_menus[0]["selected"] == 0 and note_menus[-1]["selected"] == 1


def test_esc_in_a_note_chat_leads_back_to_the_panel_not_the_project(app, workspace, monkeypatch):
    Screen(app)
    chat = app._run_cowork_panel(Keys([ENTER, DOWN, ENTER]))
    ordinary = app.home.create_session(workspace / "proj", "openai", "gpt-5.5")

    assert app._is_cowork_chat(chat) and not app._is_cowork_chat(ordinary)
    assert app._session_exit_target(chat) == "cowork"
    assert app._session_exit_target(ordinary) == "project"

    class Viewport:
        scroll = 0
        max_scroll = 0

    monkeypatch.setattr(app, "_draw_session", lambda *a, **k: Viewport())
    assert app._run_session(Keys([ESC]), chat) == "cowork"
    assert app._run_session(Keys([ESC]), ordinary) == "project"


def test_opening_a_chat_from_the_project_comes_back_through_the_panel(app, workspace, monkeypatch):
    Screen(app)
    chat = app._run_cowork_panel(Keys([ENTER, DOWN, ENTER]))
    other = app.home.create_session(workspace / "proj", "openai", "gpt-5.5")
    sessions = iter(["cowork", "cowork", "project"])
    opened: list[str] = []
    panels = iter([other, None])  # first the panel opens another chat, then the user closes it

    def fake_run_session(_stdscr, session):
        opened.append(session.title)
        return next(sessions)

    monkeypatch.setattr(app, "_run_session", fake_run_session)
    monkeypatch.setattr(app, "_run_cowork_panel", lambda _stdscr: next(panels))

    result = app._open_project_session(Keys([]), chat)

    # chat -> Esc -> panel -> another chat -> Esc -> panel -> Esc -> back at the project list
    assert result is None
    assert opened == [chat.title, other.title]


def test_merge_now_commits_immediately_and_cleans_up(app, workspace, ready_task):
    screen = Screen(app)
    repo = workspace / "proj"

    # actions: chat, diff, compare, follow, merge, merge now (index 5) -> confirm "Merge into main"
    app._run_cowork_panel(Keys([ENTER, *[DOWN] * 5, ENTER, DOWN, ENTER, ESC]))

    assert ("Cowork", "Merged into main.") in screen.messages
    assert not (repo / ".git" / "MERGE_HEAD").exists()
    assert "cowork" in git(repo, "log", "-1", "--format=%s").lower()
    assert "x.a" in (repo / "src" / "data.py").read_text("utf-8")


def test_a_pending_merge_offers_to_abort_it_from_the_panel(app, workspace, ready_task):
    screen = Screen(app)
    repo = workspace / "proj"
    app._run_cowork_panel(Keys([ENTER, *[DOWN] * 4, ENTER, DOWN, ENTER, ESC]))  # stage for review
    assert (repo / ".git" / "MERGE_HEAD").exists()

    # the note now shows [merging]; its menu has "Show changes" and "Abort the pending merge"
    app._run_cowork_panel(Keys([ENTER, DOWN, ENTER, ESC]))

    assert not (repo / ".git" / "MERGE_HEAD").exists()
    assert any("pending merge was aborted" in text for _title, text in screen.messages)
    assert (repo / "src" / "data.py").read_text("utf-8") == SOURCE
