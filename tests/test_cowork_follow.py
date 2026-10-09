"""Following the note the user clicks: debounced activation and working-tree sync."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest
from cowork_notes import note_entry, write_notes

from anomx.agent.cowork import CoworkFollower, CoworkWorkspace
from anomx.agent.runtime import AgentRuntime
from anomx.agent.store import AnomxHome

SOURCE = (
    "def load():\n"  # line 1  <- note a (stored line 0)
    "    return {'a': 1}\n"
    "\n"
    "\n"
    "def show(x):\n"  # line 5  <- note b (stored line 4)
    "    return x['a']\n"
)


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    repo = root / "proj"
    (repo / "src").mkdir(parents=True)
    git(repo, "init", "-b", "main")
    git(repo, "config", "user.name", "Dev")
    git(repo, "config", "user.email", "dev@example.com")
    (repo / "src" / "data.py").write_text(SOURCE, encoding="utf-8")
    (repo / "src" / "other.py").write_text("x = 1\n", encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-m", "initial")
    write_notes(
        root,
        note_entry("a", "proj/src/data.py", 1, "note a"),
        note_entry("b", "proj/src/data.py", 5, "note b"),
    )
    return root


@pytest.fixture
def panel(tmp_path: Path, workspace: Path) -> CoworkWorkspace:
    found = CoworkWorkspace.locate(AnomxHome(tmp_path / "home"), workspace)
    assert found is not None
    return found


@pytest.fixture
def context_file(tmp_path: Path) -> Path:
    return tmp_path / "editor-context.json"


def put_cursor(
    path: Path,
    workspace: Path,
    line: int,
    *,
    file: str = "proj/src/data.py",
    dirty: bool = False,
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": 1,
                "updatedAt": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "workspaceFolders": [str(workspace)],
                "activeFile": str(workspace / file),
                "isDirty": dirty,
                "isActiveEditor": True,
                "cursor": {"line": line, "character": 1},
                "selection": {"startLine": line, "endLine": line},
                "visibleRange": {"startLine": 1, "endLine": 50},
            }
        ),
        encoding="utf-8",
    )


# -- follower -------------------------------------------------------------------------


def test_a_note_becomes_active_only_after_the_cursor_rests_on_it(panel, workspace, context_file):
    follower = CoworkFollower(panel, context_file)
    put_cursor(context_file, workspace, 1)

    assert follower.poll() is None  # first sighting: could be arrowing past
    activation = follower.poll()

    assert activation is not None and activation.note.id == "a"
    assert follower.poll() is None  # already active


def test_passing_over_a_note_does_not_activate_it(panel, workspace, context_file):
    follower = CoworkFollower(panel, context_file)
    put_cursor(context_file, workspace, 1)
    assert follower.poll() is None
    put_cursor(context_file, workspace, 3)  # moved on to a line without a note

    assert follower.poll() is None
    put_cursor(context_file, workspace, 1)
    assert follower.poll() is None  # the count starts again


def test_the_active_note_is_sticky_until_another_note_is_selected(panel, workspace, context_file):
    follower = CoworkFollower(panel, context_file)
    put_cursor(context_file, workspace, 1)
    follower.poll()
    assert follower.poll().note.id == "a"

    put_cursor(context_file, workspace, 3)  # cursor leaves the note: topic stays
    assert follower.poll() is None and follower.active_note_id == "a"

    put_cursor(context_file, workspace, 5)
    follower.poll()
    assert follower.poll().note.id == "b"


def test_other_files_missing_context_and_reset(panel, workspace, context_file):
    follower = CoworkFollower(panel, context_file)
    assert follower.poll() is None  # no context file yet

    put_cursor(context_file, workspace, 1, file="proj/src/other.py")
    assert follower.poll() is None and follower.poll() is None

    put_cursor(context_file, workspace, 1)
    follower.poll()
    assert follower.poll() is not None
    follower.reset()
    follower.poll()
    assert follower.poll().note.id == "a"  # reactivates after a reset


def test_the_dirty_flag_of_the_editor_is_passed_on(panel, workspace, context_file):
    follower = CoworkFollower(panel, context_file)
    put_cursor(context_file, workspace, 1, dirty=True)
    follower.poll()

    assert follower.poll().editor_dirty is True


# -- working tree sync ----------------------------------------------------------------


@pytest.fixture
def ready(panel, monkeypatch):
    """Notes a and b both have a finished task; only note a changed something."""

    def backend_response(self, session_path, callbacks=None, **_kwargs):
        prompt = next(
            e["payload"]["message"]
            for e in self.home.read_session_events(session_path)
            if e["type"] == "event_msg" and e["payload"].get("type") == "user_message"
        )
        if "note a" in prompt.splitlines():  # the note text is a line of its own
            self._execute_tool(
                "edit_file",
                {"statement": "s", "path": "src/data.py", "old_text": "x['a']", "new_text": "x.a"},
                callbacks,
                session_path,
            )
        return "ok"

    monkeypatch.setattr(AgentRuntime, "backend_response", backend_response)
    for note_id in ("a", "b"):
        row = next(r for r in panel.rows() if r.note_id == note_id)
        assert panel.start(row) is None
        panel.wait(note_id)
    return panel


def _row(panel: CoworkWorkspace, note_id: str):
    return next(r for r in panel.rows() if r.note_id == note_id)


def test_sync_shows_the_task_with_changes_and_returns_for_a_note_without(ready, workspace):
    data = workspace / "proj" / "src" / "data.py"

    assert ready.sync_tree(_row(ready, "a"), editor_dirty=False) == ""
    assert "x.a" in data.read_text("utf-8")
    assert ready.following().note_id == "a"
    assert ready.sync_tree(_row(ready, "a"), editor_dirty=False) == ""  # nothing more to do

    assert ready.sync_tree(_row(ready, "b"), editor_dirty=False) == ""  # b changed nothing
    assert data.read_text("utf-8") == SOURCE
    assert ready.following() is None
    assert git(workspace / "proj", "symbolic-ref", "--short", "HEAD").strip() == "main"


def test_sync_never_switches_over_unsaved_editor_text(ready, workspace):
    data = workspace / "proj" / "src" / "data.py"

    notice = ready.sync_tree(_row(ready, "a"), editor_dirty=True)

    assert "Unsaved changes" in notice
    assert data.read_text("utf-8") == SOURCE and ready.following() is None


def test_sync_reports_uncommitted_files_instead_of_overwriting_them(ready, workspace):
    data = workspace / "proj" / "src" / "data.py"
    data.write_text(SOURCE + "# my work\n", encoding="utf-8")

    notice = ready.sync_tree(_row(ready, "a"), editor_dirty=False)

    assert "uncommitted" in notice
    assert data.read_text("utf-8") == SOURCE + "# my work\n"


def test_returning_is_also_blocked_by_unsaved_editor_text(ready, workspace):
    ready.sync_tree(_row(ready, "a"), editor_dirty=False)

    notice = ready.sync_tree(_row(ready, "b"), editor_dirty=True)

    assert "switched back" in notice and ready.following() is not None
