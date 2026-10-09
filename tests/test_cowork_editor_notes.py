"""Notes from the editor: replies file, locations in answers, questions and dispatching."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from anomx.agent.cowork import AgentReplies, CoworkWorkspace, find_locations
from anomx.agent.cowork.agent import CoworkRuntime, question_tools
from anomx.agent.runtime import AgentRuntime
from anomx.agent.store import AnomxHome
from cowork_notes import note_entry, read_agent_file, write_notes

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
    git(repo, "add", "-A")
    git(repo, "commit", "-m", "initial")
    return root


@pytest.fixture
def panel(tmp_path: Path, workspace: Path) -> CoworkWorkspace:
    return CoworkWorkspace(AnomxHome(tmp_path / "home"), workspace)


def _user_messages(home: AnomxHome, session_path: Path) -> list[str]:
    return [
        e["payload"]["message"]
        for e in home.read_session_events(session_path)
        if e["type"] == "event_msg" and e["payload"].get("type") == "user_message"
    ]


def _fake_agent(monkeypatch, reply="Done."):
    seen: list[dict[str, object]] = []

    def backend_response(self, session_path, callbacks=None, **_kwargs):
        seen.append(
            {
                "cwd": str(self.cwd),
                "question": getattr(self, "question", False),
                "messages": _user_messages(self.home, session_path),
            }
        )
        return reply if isinstance(reply, str) else reply(self, len(seen))

    monkeypatch.setattr(AgentRuntime, "backend_response", backend_response)
    return seen


def _wait(panel: CoworkWorkspace, *note_ids: str) -> None:
    for note_id in note_ids:
        panel.wait(note_id)


# -- locations in answers ---------------------------------------------------------------


def test_locations_keep_only_places_that_exist(workspace):
    repo = workspace / "proj"
    answer = (
        "It is built in `src/data.py:5-6` and loaded at src/data.py:1. "
        "Also look at src/gone.py:3, src/data.py:99 and ../outside.py:1. Repeat: src/data.py:1"
    )

    found = find_locations(answer, repo, workspace)

    assert [(item.file, item.line, item.end_line) for item in found] == [
        ("proj/src/data.py", 5, 6),
        ("proj/src/data.py", 1, None),
    ]


def test_locations_accept_workspace_relative_paths(workspace):
    found = find_locations("see proj/src/data.py:2", workspace / "proj", workspace)

    assert [(item.file, item.line) for item in found] == [("proj/src/data.py", 2)]


# -- the replies file ---------------------------------------------------------------------


def test_replies_file_records_pickup_answer_and_status(workspace):
    replies = AgentReplies(workspace)

    replies.pick_up("n-1", "m-1")
    assert replies.status("n-1") == "working" and replies.picked_up("n-1") == {"m-1"}

    replies.add_answer("n-1", "The answer", after_message="m-1")
    entry = read_agent_file(workspace)["n-1"]
    assert entry["status"] == "ready" and entry["pickedUp"] == ["m-1"]
    assert entry["messages"][0]["text"] == "The answer"
    assert entry["messages"][0]["afterMessage"] == "m-1"

    replies.add_answer("n-1", "boom", failed=True)
    assert read_agent_file(workspace)["n-1"]["status"] == "failed"
    assert read_agent_file(workspace)["n-1"]["detail"] == "boom"
    assert (workspace / ".anomx" / "agent.json").stat().st_mode & 0o777 == 0o600


def test_replies_survive_a_corrupt_file(workspace):
    path = workspace / ".anomx" / "agent.json"
    path.parent.mkdir()
    path.write_text("{broken", encoding="utf-8")

    replies = AgentReplies(workspace)

    assert replies.status("x") == "idle"
    replies.pick_up("x", "m")
    assert json.loads(path.read_text(encoding="utf-8"))["notes"]["x"]["pickedUp"] == ["m"]


# -- questions ------------------------------------------------------------------------------


def test_a_question_agent_cannot_edit():
    assert "edit_file" not in {tool.name for tool in question_tools()}


def test_a_question_is_answered_read_only_with_clickable_locations(panel, workspace, monkeypatch):
    seen = _fake_agent(monkeypatch, "The dict is read in `src/data.py:5-6`.")
    write_notes(
        workspace,
        note_entry(
            "q-1",
            None,
            None,
            "where is the dict used?",
            kind="question",
            context_file=str(workspace / "proj" / "src" / "data.py"),
        ),
    )

    assert panel.dispatch() == 1
    _wait(panel, "q-1")

    entry = read_agent_file(workspace)["q-1"]
    assert entry["status"] == "ready" and entry["pickedUp"] == ["m-q-1"]
    assert entry["messages"][0]["text"].startswith("The dict is read")
    assert entry["messages"][0]["locations"] == [
        {"file": "proj/src/data.py", "line": 5, "endLine": 6}
    ]
    assert seen[0]["question"] is True
    assert seen[0]["cwd"] == str((workspace / "proj").resolve())  # reads the real repository
    assert (
        "data.py" in seen[0]["messages"][0] and "where is the dict used?" in seen[0]["messages"][0]
    )
    assert git(workspace / "proj", "branch", "--list").split() == ["*", "main"]  # no task branch
    assert panel.dispatch() == 0  # answered messages are not run again


def test_a_follow_up_question_continues_the_same_conversation(panel, workspace, monkeypatch):
    seen = _fake_agent(monkeypatch, "Answer.")
    file = str(workspace / "proj" / "src" / "data.py")
    write_notes(
        workspace, note_entry("q", None, None, "first?", kind="question", context_file=file)
    )
    panel.dispatch()
    _wait(panel, "q")

    write_notes(
        workspace,
        note_entry(
            "q",
            None,
            None,
            "first?",
            kind="question",
            context_file=file,
            extra_messages=("and then?",),
        ),
    )
    assert panel.dispatch() == 1
    _wait(panel, "q")

    assert len(seen[1]["messages"]) == 2 and seen[1]["messages"][1] == "and then?"
    assert len(read_agent_file(workspace)["q"]["messages"]) == 2


def test_a_question_without_a_repository_says_so(tmp_path, monkeypatch):
    lonely = tmp_path / "lonely"
    lonely.mkdir()
    panel = CoworkWorkspace(AnomxHome(tmp_path / "home"), lonely)
    _fake_agent(monkeypatch)
    write_notes(lonely, note_entry("q", None, None, "where?", kind="question"))

    assert panel.dispatch() == 0

    entry = read_agent_file(lonely)["q"]
    assert entry["status"] == "failed" and "which repository" in entry["messages"][0]["text"]
    assert panel.dispatch() == 0  # reported once, not on every poll


# -- notes ----------------------------------------------------------------------------------


def test_a_comment_starts_an_agent_in_its_own_worktree_and_answers_in_the_thread(
    panel, workspace, monkeypatch
):
    seen = _fake_agent(monkeypatch, "Changed `src/data.py:1`.")
    write_notes(
        workspace, note_entry("n-1", "proj/src/data.py", 2, "make it a dataclass", end_line=3)
    )

    assert panel.dispatch() == 1
    _wait(panel, "n-1")

    entry = read_agent_file(workspace)["n-1"]
    assert entry["status"] == "ready" and entry["pickedUp"] == ["m-n-1"]
    assert entry["messages"][0]["afterMessage"] == "m-n-1"
    assert "/cowork/" in str(seen[0]["cwd"]) and seen[0]["question"] is False
    prompt = seen[0]["messages"][0]
    assert "Lines: 2-3" in prompt and "make it a dataclass" in prompt


def test_a_reply_in_the_thread_continues_the_task_conversation(panel, workspace, monkeypatch):
    seen = _fake_agent(monkeypatch, "Ok.")
    write_notes(workspace, note_entry("n-1", "proj/src/data.py", 2, "make it a dataclass"))
    panel.dispatch()
    _wait(panel, "n-1")

    write_notes(
        workspace,
        note_entry(
            "n-1", "proj/src/data.py", 2, "make it a dataclass", extra_messages=("also rename it",)
        ),
    )
    assert panel.dispatch() == 1
    _wait(panel, "n-1")

    assert seen[1]["messages"][-1] == "also rename it"
    assert len(read_agent_file(workspace)["n-1"]["messages"]) == 2
    assert read_agent_file(workspace)["n-1"]["status"] == "ready"


def test_a_note_in_a_file_outside_any_repository_is_rejected_once(panel, workspace, monkeypatch):
    _fake_agent(monkeypatch)
    (workspace / "loose.py").write_text("x = 1\n", encoding="utf-8")
    write_notes(workspace, note_entry("n-2", "loose.py", 1, "change x"))

    assert panel.dispatch() == 0

    entry = read_agent_file(workspace)["n-2"]
    assert entry["status"] == "failed" and "not inside a git repository" in entry["detail"]
    assert panel.dispatch() == 0


def test_resolved_notes_are_ignored(panel, workspace, monkeypatch):
    _fake_agent(monkeypatch)
    write_notes(workspace, note_entry("n-1", "proj/src/data.py", 2, "x", status="resolved"))

    assert panel.dispatch() == 0
    assert read_agent_file(workspace) == {}


def test_a_chat_opened_in_the_terminal_is_not_started_again_by_the_editor(
    panel, workspace, monkeypatch
):
    seen = _fake_agent(monkeypatch)
    write_notes(workspace, note_entry("n-1", "proj/src/data.py", 2, "discuss this"))
    row = next(item for item in panel.rows() if item.note_id == "n-1")

    record, problem = panel.open_chat(row, discuss=True)

    assert problem is None and record is not None
    assert panel.dispatch() == 0 and seen == []
    assert read_agent_file(workspace)["n-1"]["status"] == "idle"


def test_a_merge_the_user_aborts_is_reported_back_to_the_editor(panel, workspace, monkeypatch):
    _fake_agent(monkeypatch)
    write_notes(workspace, note_entry("n-1", "proj/src/data.py", 2, "x"))
    panel.dispatch()
    _wait(panel, "n-1")
    service = panel._service(workspace / "proj")  # noqa: SLF001
    task = service.tasks.get("n-1")
    assert task is not None
    worktree = Path(task.worktree)
    (worktree / "src" / "data.py").write_text("x = 2\n", encoding="utf-8")
    git(worktree, "commit", "-am", "agent change")  # an unmerged branch, like a real result
    assert service.merge("n-1").pending  # staged in the user's repository, not committed

    panel.dispatch()
    assert read_agent_file(workspace)["n-1"]["status"] == "merging"

    service.abort_pending_merge("n-1")
    panel.dispatch()
    assert read_agent_file(workspace)["n-1"]["status"] == "ready"


def test_question_runtime_describes_a_read_only_repository(tmp_path):
    runtime = CoworkRuntime(AnomxHome(tmp_path / "home"), tmp_path, question=True)

    instructions = runtime._instructions()  # noqa: SLF001

    assert "Only read" in instructions and "edit_file" not in instructions
