"""The cowork panel's view model: rows, offered actions and background runs."""

from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path

import pytest

from anomx.agent.cowork import CoworkWorkspace, TaskStatus
from anomx.agent.runtime import AgentRuntime, RuntimeCallbacks
from anomx.agent.store import AnomxHome

SOURCE = "def load():\n    return {'a': 1}\n\n\ndef show(x):\n    return x['a']\n"


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
    (repo / "plain.txt").write_text("x\n", encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-m", "initial")
    (root / "loose.py").write_text("print(1)\n", encoding="utf-8")  # not in any repository
    (root / ".margo").mkdir()
    vault = {
        "./proj/src/data.py": [{"id": "n-1", "line": 0, "title": "make dataclass", "text": ""}],
        "./loose.py": [{"id": "n-2", "line": 0, "title": "outside a repo", "text": ""}],
        "./proj/plain.txt": [{"id": "n-3", "line": 0, "title": "resolved", "status": "resolved"}],
    }
    (root / ".margo" / "local.notes.json").write_text(json.dumps(vault), encoding="utf-8")
    return root


@pytest.fixture
def panel(tmp_path: Path, workspace: Path) -> CoworkWorkspace:
    found = CoworkWorkspace.locate(AnomxHome(tmp_path / "home"), workspace / "proj" / "src")
    assert found is not None
    return found


def _row(panel: CoworkWorkspace, note_id: str):
    return next(row for row in panel.rows() if row.note_id == note_id)


def _fake_agent(monkeypatch, *, edit=True, release: threading.Event | None = None):
    def backend_response(self, session_path, callbacks=None, **_kwargs):
        active = callbacks or RuntimeCallbacks()
        if active.status:
            active.status("Reading data.py")
        if release is not None:
            release.wait(10)
        if edit:
            self._execute_tool(
                "edit_file",
                {"statement": "s", "path": "src/data.py", "old_text": "x['a']", "new_text": "x.a"},
                active,
                session_path,
            )
        return "Switched to attribute access."

    monkeypatch.setattr(AgentRuntime, "backend_response", backend_response)


def test_locate_needs_a_margo_vault(tmp_path):
    assert CoworkWorkspace.locate(AnomxHome(tmp_path / "home"), tmp_path / "nowhere") is None


def test_rows_list_open_notes_and_mark_files_outside_a_repository(panel):
    rows = {row.note_id: row for row in panel.rows()}

    assert set(rows) == {"n-1", "n-2"}  # the resolved note is hidden
    assert rows["n-1"].status == "new" and rows["n-1"].label == "proj/src/data.py:1  make dataclass"
    assert rows["n-2"].status == "no-repo" and panel.actions(rows["n-2"]) == []


def test_a_new_note_offers_only_start(panel):
    actions = panel.actions(_row(panel, "n-1"))

    assert [a.value for a in actions] == ["start"]


def test_start_runs_in_the_background_and_ends_ready_with_actions(panel, monkeypatch):
    release = threading.Event()
    _fake_agent(monkeypatch, release=release)

    assert panel.start(_row(panel, "n-1")) is None
    for _ in range(100):  # the first progress line arrives once the agent is up
        running = _row(panel, "n-1")
        if running.detail == "Reading data.py":
            break
        threading.Event().wait(0.05)
    assert running.running and running.status == "running"
    assert running.detail == "Reading data.py"  # live progress from the agent
    assert [a.value for a in panel.actions(running)] == ["stop"]

    release.set()
    panel.wait("n-1")
    ready = _row(panel, "n-1")
    assert ready.status == "ready" and not ready.running
    assert "attribute access" in ready.detail
    assert [a.value for a in panel.actions(ready)] == [
        "diff",
        "follow",
        "merge",
        "start",
        "discard",
    ]
    assert panel.actions(ready)[2].label == "Merge into main"
    assert any("+    return x.a" in line for line in panel.diff_lines(ready))


def test_ready_without_changes_offers_no_merge(panel, monkeypatch):
    _fake_agent(monkeypatch, edit=False)
    panel.start(_row(panel, "n-1"))
    panel.wait("n-1")

    assert [a.value for a in panel.actions(_row(panel, "n-1"))] == ["start", "discard"]


def test_preparation_errors_are_returned_not_raised(panel, workspace):
    (workspace / "proj" / "src" / "data.py").write_text(SOURCE + "# wip\n", encoding="utf-8")

    message = panel.start(_row(panel, "n-1"))

    assert message is not None and "uncommitted" in message
    assert _row(panel, "n-1").status == "new"  # nothing was created


def test_a_crashing_agent_is_shown_as_failed_with_the_reason(panel, monkeypatch):
    def explode(self, *_a, **_k):
        raise RuntimeError("model unreachable")

    monkeypatch.setattr(AgentRuntime, "backend_response", explode)
    panel.start(_row(panel, "n-1"))
    panel.wait("n-1")

    failed = _row(panel, "n-1")
    assert failed.status == "failed" and "model unreachable" in failed.detail
    assert "start" in [a.value for a in panel.actions(failed)]  # run again


def test_stop_asks_the_running_agent_to_end(panel, monkeypatch):
    started = threading.Event()

    def waits_for_abort(self, *_a, **_k):
        started.set()
        while not self._turn_aborted():
            threading.Event().wait(0.01)
        return ""

    monkeypatch.setattr(AgentRuntime, "backend_response", waits_for_abort)
    panel.start(_row(panel, "n-1"))
    started.wait(5)

    assert panel.stop(_row(panel, "n-1")) is True
    panel.wait("n-1")
    assert _row(panel, "n-1").status == "failed"


def test_follow_show_return_and_merge(panel, workspace, monkeypatch):
    _fake_agent(monkeypatch)
    panel.start(_row(panel, "n-1"))
    panel.wait("n-1")
    ready = _row(panel, "n-1")
    data = workspace / "proj" / "src" / "data.py"

    assert panel.following() is None
    assert panel.follow(ready).ok
    assert "x.a" in data.read_text("utf-8")
    followed = panel.following()
    assert followed is not None and followed.note_id == "n-1"
    assert panel.return_to_base(ready).ok and panel.following() is None
    assert data.read_text("utf-8") == SOURCE

    merged = panel.merge(ready)

    assert merged.ok and "x.a" in data.read_text("utf-8")
    after = _row(panel, "n-1")
    assert after.status == "merged" and panel.actions(after) == []


def test_discard_makes_the_note_startable_again(panel, monkeypatch):
    _fake_agent(monkeypatch)
    panel.start(_row(panel, "n-1"))
    panel.wait("n-1")

    panel.discard(_row(panel, "n-1"))

    assert _row(panel, "n-1").status == "new"
    assert panel.start(_row(panel, "n-1")) is None
    panel.wait("n-1")
    assert _row(panel, "n-1").status == "ready"


def test_a_task_whose_note_was_resolved_stays_visible_as_orphan(panel, workspace, monkeypatch):
    _fake_agent(monkeypatch)
    panel.start(_row(panel, "n-1"))
    panel.wait("n-1")
    vault = workspace / ".margo" / "local.notes.json"
    data = json.loads(vault.read_text(encoding="utf-8"))
    data["./proj/src/data.py"][0]["status"] = "resolved"
    vault.write_text(json.dumps(data), encoding="utf-8")

    orphan = _row(panel, "n-1")

    assert orphan.status == "orphan" and orphan.note is None and orphan.task is not None
    assert orphan.task.status is TaskStatus.READY
    assert "discard" in [a.value for a in panel.actions(orphan)]
