"""The /cowork panel driven with key presses against a simulated model."""

from __future__ import annotations

import curses
import json
import subprocess
from pathlib import Path

import pytest

from anomx.agent.app import AnomxCliApp
from anomx.agent.cowork.agent import CoworkRuntime
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

    monkeypatch.setattr(CoworkRuntime, "backend_response", backend_response)


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

    assert screen.last_choices() == [("proj/src/data.py:1  make dataclass", "[new]")]
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

    # open note -> "Show changes" (first action) -> close viewer -> close menu
    app._run_cowork_panel(Keys([ENTER, ENTER, ESC, ESC]))

    viewer = next(o for o in screen.overlays if o["title"].startswith("Changes"))
    assert "+    return x.a" in viewer["body_lines"]


def test_follow_switches_the_tree_and_back_returns(app, workspace, fake_agent):
    screen = Screen(app)
    _start_and_finish(app, workspace)
    data = workspace / "proj" / "src" / "data.py"

    # open note -> "Show in editor" (second action) -> message -> close panel
    app._run_cowork_panel(Keys([ENTER, DOWN, ENTER, ESC]))
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

    # open note -> "Merge into main" (third action) -> confirm menu: Cancel
    app._run_cowork_panel(Keys([ENTER, DOWN, DOWN, ENTER, ENTER, ESC]))
    assert data.read_text("utf-8") == SOURCE  # cancelled: nothing merged

    # open note -> Merge -> confirm "Merge into main" (second entry)
    app._run_cowork_panel(Keys([ENTER, DOWN, DOWN, ENTER, DOWN, ENTER, ESC]))

    assert "x.a" in data.read_text("utf-8")
    assert ("Cowork", "Merged into main.") in screen.messages
    assert any("Merge into main?" in (o.get("subtitle") or "") for o in screen.overlays)
    assert "Changed: src/data.py" in next(
        o["subtitle"] for o in screen.overlays if "Merge into main?" in (o.get("subtitle") or "")
    )


def test_discard_asks_first_and_frees_the_note(app, workspace, fake_agent):
    screen = Screen(app)
    _start_and_finish(app, workspace)

    # open note -> actions: diff, follow, merge, run again, discard (index 4) -> confirm
    app._run_cowork_panel(Keys([ENTER, DOWN, DOWN, DOWN, DOWN, ENTER, DOWN, ENTER, ESC]))

    app._run_cowork_panel(Keys([ESC]))
    assert screen.last_choices()[0][1] == "[new]"


def test_start_error_is_shown_instead_of_crashing(app, workspace, fake_agent):
    screen = Screen(app)
    (workspace / "proj" / "src" / "data.py").write_text(SOURCE + "# wip\n", encoding="utf-8")

    app._run_cowork_panel(Keys([ENTER, ENTER, ESC]))

    assert screen.messages and screen.messages[0][0] == "Cannot start"
    assert "uncommitted" in screen.messages[0][1]
