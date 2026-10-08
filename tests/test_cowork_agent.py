"""Cowork agent: the worktree-scoped edit tool, the lean runtime and the task service."""

from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

from anomx.agent.cowork import (
    CoworkService,
    GitError,
    TaskStatus,
    build_task_prompt,
    find_repo_root,
    note_context,
    read_margo_notes,
)
from anomx.agent.cowork.agent import CoworkRuntime, EditError, apply_edit
from anomx.agent.exceptions import BackendFailure
from anomx.agent.runtime import AgentRuntime, RuntimeCallbacks
from anomx.agent.store import AnomxHome

SOURCE = (
    "def load():\n"
    "    return {'name': 'a', 'size': 1}\n"
    "\n"
    "\n"
    "def show(item):\n"
    "    return item['name']\n"
)


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout


# -- apply_edit -----------------------------------------------------------------------


def test_edit_replaces_a_unique_match_and_reports_the_line(tmp_path):
    (tmp_path / "a.py").write_text(SOURCE, encoding="utf-8")

    result = apply_edit(tmp_path, "a.py", "item['name']", "item.name")

    assert (result.action, result.replacements, result.first_changed_line) == ("edited", 1, 6)
    assert "item.name" in (tmp_path / "a.py").read_text(encoding="utf-8")
    assert [p.name for p in tmp_path.iterdir()] == ["a.py"]  # no temp files left


def test_edit_requires_a_unique_match_unless_replace_all(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\nx = 1\n", encoding="utf-8")

    with pytest.raises(EditError, match="matches 2 places"):
        apply_edit(tmp_path, "a.py", "x = 1", "x = 2")
    with pytest.raises(EditError, match="not found"):
        apply_edit(tmp_path, "a.py", "y = 1", "y = 2")

    assert apply_edit(tmp_path, "a.py", "x = 1", "x = 2", replace_all=True).replacements == 2
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "x = 2\nx = 2\n"


def test_edit_creates_new_files_but_never_overwrites_with_an_empty_old_text(tmp_path):
    result = apply_edit(tmp_path, "pkg/new.py", "", "VALUE = 1\n")

    assert result.action == "created"
    assert (tmp_path / "pkg" / "new.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    with pytest.raises(EditError, match="already exists"):
        apply_edit(tmp_path, "pkg/new.py", "", "other")


def test_edit_cannot_leave_the_worktree(tmp_path):
    root = tmp_path / "wt"
    root.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("secret", encoding="utf-8")
    (root / "link.txt").symlink_to(outside)

    for path in ("../secret.txt", str(outside), "link.txt"):
        with pytest.raises(EditError, match="outside the task worktree"):
            apply_edit(root, path, "secret", "changed")
    assert outside.read_text(encoding="utf-8") == "secret"


def test_edit_refuses_git_internals(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("[core]\n", encoding="utf-8")

    with pytest.raises(EditError, match=r"\.git"):
        apply_edit(tmp_path, ".git/config", "[core]", "[evil]")


def test_edit_preserves_crlf_line_endings_and_file_mode(tmp_path):
    path = tmp_path / "script.sh"
    path.write_bytes(b"echo one\r\necho two\r\n")
    path.chmod(0o755)

    apply_edit(tmp_path, "script.sh", "echo one\necho two", "echo 1\necho 2")

    assert path.read_bytes() == b"echo 1\r\necho 2\r\n"
    assert stat.S_IMODE(path.stat().st_mode) == 0o755


def test_edit_refuses_binary_and_missing_files(tmp_path):
    (tmp_path / "blob.bin").write_bytes(b"\xff\xfe\x00")

    with pytest.raises(EditError, match="UTF-8"):
        apply_edit(tmp_path, "blob.bin", "x", "y")
    with pytest.raises(EditError, match="does not exist"):
        apply_edit(tmp_path, "nope.py", "x", "y")


# -- lean runtime ---------------------------------------------------------------------


@pytest.fixture
def home(tmp_path: Path) -> AnomxHome:
    return AnomxHome(tmp_path / "home")


def test_runtime_exposes_only_read_search_and_edit_tools(home, tmp_path):
    worktree = tmp_path / "wt"
    worktree.mkdir()
    runtime = CoworkRuntime(home, worktree)

    names = {tool.name for tool in runtime._available_tools()}

    assert names == {"read", "list", "glob", "grep", "edit_file"}
    assert not runtime.agent_spec.can_spawn_subagents and not runtime.agent_spec.can_use_web
    runtime.set_agent("main")  # cannot be swapped for a general agent
    assert runtime.agent_spec.name == "Cowork"


def test_runtime_prompt_is_lean_and_names_the_worktree(home, tmp_path):
    worktree = tmp_path / "wt"
    worktree.mkdir()
    prompt = CoworkRuntime(home, worktree)._instructions(None)

    assert "cowork agent" in prompt and "edit_file" in prompt
    assert str(worktree.resolve()) in prompt
    assert "skill" not in prompt.lower() and "memor" not in prompt.lower()
    assert len(prompt) < 4_000


def test_edit_tool_runs_through_the_runtime_inside_the_worktree(home, tmp_path):
    worktree = tmp_path / "wt"
    worktree.mkdir()
    (worktree / "a.py").write_text(SOURCE, encoding="utf-8")
    runtime = CoworkRuntime(home, worktree)
    callbacks = RuntimeCallbacks()
    edit = {"statement": "edit", "path": "a.py", "old_text": "'size': 1", "new_text": "'size': 2"}

    ok = json.loads(runtime._execute_tool("edit_file", edit, callbacks))
    outside = json.loads(
        runtime._execute_tool("edit_file", {**edit, "path": "../x.py", "old_text": ""}, callbacks)
    )
    shell = json.loads(runtime._execute_tool("run_cli_command", {"command": "ls"}, callbacks))

    assert ok["action"] == "edited" and "'size': 2" in (worktree / "a.py").read_text("utf-8")
    assert "outside the task worktree" in outside["error"]
    assert "Unknown tool" in shell["error"]  # no shell for cowork agents


# -- service --------------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """VS Code workspace folder holding one repository (like app_project/anomx)."""

    root = tmp_path / "workspace"
    repo = root / "proj"
    (repo / "src").mkdir(parents=True)
    git(repo, "init", "-b", "main")
    git(repo, "config", "user.name", "Dev")
    git(repo, "config", "user.email", "dev@example.com")
    (repo / "src" / "data.py").write_text(SOURCE, encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-m", "initial")
    (root / ".margo").mkdir()
    vault = {
        "./proj/src/data.py": [
            {"id": "n-1", "line": 0, "title": "make the dict a dataclass", "text": "and callers"}
        ]
    }
    (root / ".margo" / "local.notes.json").write_text(json.dumps(vault), encoding="utf-8")
    return root


@pytest.fixture
def service(home: AnomxHome, workspace: Path) -> CoworkService:
    return CoworkService(home, workspace / "proj")


def _fake_agent(monkeypatch, *, edits=(), reply="Changed the dict; callers updated."):
    seen: dict[str, str] = {}

    def backend_response(self, session_path, callbacks=None, **_kwargs):
        seen["prompt"] = next(
            e["payload"]["message"]
            for e in self.home.read_session_events(session_path)
            if e["type"] == "event_msg" and e["payload"].get("type") == "user_message"
        )
        seen["cwd"] = str(self.cwd)
        for old, new in edits:
            self._execute_tool(
                "edit_file",
                {"statement": "s", "path": "src/data.py", "old_text": old, "new_text": new},
                callbacks or RuntimeCallbacks(),
                session_path,
            )
        return reply

    monkeypatch.setattr(AgentRuntime, "backend_response", backend_response)
    return seen


def _note(workspace: Path):
    (note,) = read_margo_notes(workspace)
    return note


def test_prepare_creates_branch_worktree_and_anchor(service, workspace):
    task = service.prepare(_note(workspace), workspace)

    assert task.status is TaskStatus.QUEUED
    assert (task.file, task.line, task.base_branch) == ("src/data.py", 1, "main")
    assert task.anchor_text == "def load():"
    assert Path(task.worktree, "src", "data.py").is_file()
    assert task.branch == "anomx/cowork/n-1"
    assert service.git.current_branch() == "main" and service.git.is_clean()


def test_prepare_refuses_files_with_uncommitted_changes_and_duplicate_tasks(service, workspace):
    target = workspace / "proj" / "src" / "data.py"
    target.write_text(SOURCE + "# wip\n", encoding="utf-8")

    with pytest.raises(GitError, match="uncommitted"):
        service.prepare(_note(workspace), workspace)

    git(workspace / "proj", "checkout", "--", "src/data.py")
    service.prepare(_note(workspace), workspace)
    with pytest.raises(GitError, match="already has an active task"):
        service.prepare(_note(workspace), workspace)


def test_run_lets_the_agent_edit_its_worktree_and_commits_the_result(
    service, workspace, monkeypatch
):
    seen = _fake_agent(monkeypatch, edits=[("item['name']", "item.name")])
    task = service.prepare(_note(workspace), workspace)

    result = service.run(task.note_id)

    assert result.committed and not result.error
    assert result.task.status is TaskStatus.READY
    assert result.summary == "Changed the dict; callers updated."
    assert seen["cwd"] == str(Path(task.worktree).resolve())
    assert "make the dict a dataclass" in seen["prompt"] and "and callers" in seen["prompt"]
    assert "src/data.py" in seen["prompt"] and "def load():" in seen["prompt"]
    assert service.changed_files(task.note_id) == ["src/data.py"]
    assert "+    return item.name" in service.diff(task.note_id)
    # the user's tree is untouched while the agent works in its worktree
    assert (workspace / "proj" / "src" / "data.py").read_text("utf-8") == SOURCE
    assert git(Path(task.worktree), "log", "-1", "--format=%s").strip().startswith("cowork:")


def test_run_without_changes_is_ready_but_not_mergeable(service, workspace, monkeypatch):
    _fake_agent(monkeypatch, reply="No change needed: the value is already a dataclass.")
    task = service.prepare(_note(workspace), workspace)

    result = service.run(task.note_id)

    assert result.task.status is TaskStatus.READY and not result.committed
    merged = service.merge(task.note_id)
    assert not merged.ok and "no changes" in merged.reason


def test_backend_failure_marks_the_task_failed_and_it_can_run_again(
    service, workspace, monkeypatch
):
    monkeypatch.setattr(
        AgentRuntime, "backend_response", lambda self, *a, **k: BackendFailure("model down")
    )
    task = service.prepare(_note(workspace), workspace)

    failed = service.run(task.note_id)

    assert failed.task.status is TaskStatus.FAILED and failed.error == "model down"
    _fake_agent(monkeypatch, edits=[("item['name']", "item.name")])
    assert service.run(task.note_id).task.status is TaskStatus.READY


def test_an_exception_or_abort_is_recorded_instead_of_escaping(service, workspace, monkeypatch):
    def explode(self, *_a, **_k):
        raise RuntimeError("network lost")

    monkeypatch.setattr(AgentRuntime, "backend_response", explode)
    task = service.prepare(_note(workspace), workspace)
    assert "network lost" in service.run(task.note_id).error

    def aborted(self, *_a, **_k):
        self.abort_current_turn()
        return ""

    monkeypatch.setattr(AgentRuntime, "backend_response", aborted)
    result = service.run(task.note_id)
    assert result.task.status is TaskStatus.FAILED and "Stopped" in result.error


def test_only_queued_or_failed_tasks_can_run(service, workspace, monkeypatch):
    _fake_agent(monkeypatch, edits=[("item['name']", "item.name")])
    task = service.prepare(_note(workspace), workspace)
    service.run(task.note_id)

    with pytest.raises(GitError, match="only queued or failed"):
        service.run(task.note_id)
    with pytest.raises(GitError, match="No cowork task"):
        service.run("missing")


def test_merge_after_approval_updates_the_users_branch_and_cleans_up(
    service, workspace, monkeypatch
):
    _fake_agent(monkeypatch, edits=[("item['name']", "item.name")])
    task = service.prepare(_note(workspace), workspace)
    service.run(task.note_id)
    assert service.follow(task.note_id).ok  # the user looks at the agent's version
    assert "item.name" in (workspace / "proj" / "src" / "data.py").read_text("utf-8")

    result = service.merge(task.note_id, review=False)

    assert result.ok
    assert service.git.current_branch() == "main"
    assert "item.name" in (workspace / "proj" / "src" / "data.py").read_text("utf-8")
    assert service.tasks.get(task.note_id).status is TaskStatus.MERGED
    assert not Path(task.worktree).exists() and not service.git.branch_exists(task.branch)
    assert not service.merge(task.note_id).ok  # a merged task cannot be merged twice


def test_discard_while_following_returns_to_the_base_branch(service, workspace, monkeypatch):
    _fake_agent(monkeypatch, edits=[("item['name']", "item.name")])
    task = service.prepare(_note(workspace), workspace)
    service.run(task.note_id)
    service.follow(task.note_id)

    service.discard(task.note_id)

    assert service.git.current_branch() == "main"
    assert (workspace / "proj" / "src" / "data.py").read_text("utf-8") == SOURCE
    assert service.tasks.get(task.note_id).status is TaskStatus.DISCARDED
    assert not service.git.branch_exists(task.branch)
    assert service.prepare(_note(workspace), workspace).status is TaskStatus.QUEUED  # reusable


def test_task_state_survives_a_new_service_instance(home, service, workspace, monkeypatch):
    _fake_agent(monkeypatch, edits=[("item['name']", "item.name")])
    task = service.prepare(_note(workspace), workspace)
    service.run(task.note_id)

    again = CoworkService(home, workspace / "proj")

    restored = again.tasks.get(task.note_id)
    assert restored is not None and restored.status is TaskStatus.READY
    assert again.state_dir == service.state_dir


def test_prompt_contains_only_the_note_and_nearby_code(workspace):
    note = _note(workspace)
    context = note_context(workspace, note)
    repo = find_repo_root(workspace / note.file)
    assert repo == workspace / "proj"

    class _Task:
        file, line, request, anchor_kind = "src/data.py", 1, note.request, "source"

    prompt = build_task_prompt(_Task(), context)  # type: ignore[arg-type]

    assert prompt.startswith("The user left this note")
    assert "make the dict a dataclass" in prompt and "    2      return {" in prompt
    assert "Make the change the note asks for" in prompt
    assert os.sep + "workspace" not in prompt  # no absolute paths leak into the prompt


# -- chat per note --------------------------------------------------------------------


def _messages(home: AnomxHome, session_path: Path) -> list[str]:
    return [
        event["payload"]["message"]
        for event in home.read_session_events(session_path)
        if event["type"] == "event_msg" and event["payload"].get("type") == "user_message"
    ]


def test_open_chat_creates_a_session_in_the_worktree_titled_after_the_note(
    service, home, workspace
):
    task, record = service.open_chat(_note(workspace), workspace, discuss=True)

    assert Path(record.cwd) == Path(task.worktree).resolve()
    assert record.title == "make the dict a dataclass"
    (first,) = _messages(home, record.path)
    assert "talk this through first" in first and "Do not edit any file" in first
    assert "make the dict a dataclass" in first and "def load():" in first
    assert service.note_for_session(record.path) == "n-1"
    assert service.note_for_session(Path("/somewhere/else.jsonl")) is None


def test_open_chat_reuses_the_session_instead_of_creating_another(service, workspace):
    _task, first = service.open_chat(_note(workspace), workspace, discuss=True)

    _task, again = service.open_chat(_note(workspace), workspace)

    assert again.path == first.path


def test_chat_turns_are_committed_on_top_of_each_other(service, home, workspace, monkeypatch):
    task, record = service.open_chat(_note(workspace), workspace)
    replies = iter(
        [
            ([("item['name']", "item.name")], "First: attribute access."),
            ([], "Option B would be a TypedDict; no edit."),
            ([("'size': 1", "'size': 2")], "Second: bumped the size."),
        ]
    )

    def turn(self, session_path, callbacks=None, **_kwargs):
        edits, reply = next(replies)
        for old, new in edits:
            self._execute_tool(
                "edit_file",
                {"statement": "s", "path": "src/data.py", "old_text": old, "new_text": new},
                callbacks or RuntimeCallbacks(),
                session_path,
            )
        return reply

    monkeypatch.setattr(AgentRuntime, "backend_response", turn)
    runtime = service.runtime_for("n-1")

    runtime.backend_response(record.path)
    assert service.git.commits_ahead(task.branch, task.base_commit) == 1
    runtime.backend_response(record.path)  # a discussion turn changes nothing
    assert service.git.commits_ahead(task.branch, task.base_commit) == 1
    assert service.tasks.get("n-1").summary.startswith("Option B")
    runtime.backend_response(record.path)

    assert service.git.commits_ahead(task.branch, task.base_commit) == 2
    assert service.tasks.get("n-1").status is TaskStatus.READY
    diff = service.diff("n-1")
    assert "+    return item.name" in diff and "'size': 2" in diff


def test_a_failed_chat_turn_marks_the_task_failed(service, workspace, monkeypatch):
    task, record = service.open_chat(_note(workspace), workspace)
    monkeypatch.setattr(
        AgentRuntime, "backend_response", lambda self, *a, **k: BackendFailure("rate limited")
    )

    service.runtime_for("n-1").backend_response(record.path)

    stored = service.tasks.get(task.note_id)
    assert stored.status is TaskStatus.FAILED and stored.summary == "rate limited"


def test_starting_the_agent_continues_an_existing_chat(service, home, workspace, monkeypatch):
    _task, record = service.open_chat(_note(workspace), workspace, discuss=True)
    seen: dict[str, str] = {}

    def backend_response(self, session_path, callbacks=None, **_kwargs):
        seen["session"] = str(session_path)
        self._execute_tool(
            "edit_file",
            {
                "statement": "s",
                "path": "src/data.py",
                "old_text": "item['name']",
                "new_text": "item.name",
            },
            callbacks or RuntimeCallbacks(),
            session_path,
        )
        return "Done."

    monkeypatch.setattr(AgentRuntime, "backend_response", backend_response)

    result = service.run("n-1")

    assert seen["session"] == str(record.path)  # same conversation, no second session
    assert _messages(home, record.path)[-1] == "Please make the change now."
    assert result.committed and result.task.status is TaskStatus.READY


def test_the_cowork_runtime_never_starts_a_sandbox(home, tmp_path):
    worktree = tmp_path / "wt"
    worktree.mkdir()

    assert CoworkRuntime(home, worktree).init_sandbox() is False


# -- unanswered notes -----------------------------------------------------------------


def _event(kind: str) -> dict:
    return {"type": "event_msg", "payload": {"type": kind, "message": "x"}}


def test_awaiting_reply_follows_the_last_user_and_agent_messages():
    from anomx.agent.cowork.runner import awaiting_reply

    meta = {"type": "session_meta", "payload": {}}
    assert awaiting_reply([]) is False
    assert awaiting_reply([meta, _event("user_message")]) is True
    assert awaiting_reply([meta, _event("user_message"), _event("agent_message")]) is False
    assert awaiting_reply([_event("user_message"), _event("agent_message"), _event("user_message")])
    assert awaiting_reply([_event("user_message"), _event("work_message")]) is True


def test_a_fresh_chat_awaits_a_reply_until_the_agent_has_answered(service, home, workspace):
    _task, record = service.open_chat(_note(workspace), workspace, discuss=True)

    assert service.awaiting_reply(record.path) is True

    home.append_session_event(record.path, "agent_message", {"message": "Here are options."})
    assert service.awaiting_reply(record.path) is False


# -- notes on files git does not track ------------------------------------------------


@pytest.fixture
def generated_note(workspace):
    """A note sitting on build output that is ignored by git (like dist/bundle.mjs)."""

    repo = workspace / "proj"
    (repo / ".gitignore").write_text("dist/\n", encoding="utf-8")
    git(repo, "add", ".gitignore")
    git(repo, "commit", "-m", "ignore dist")
    (repo / "dist").mkdir()
    (repo / "dist" / "bundle.mjs").write_text("var a=1;" * 50, encoding="utf-8")
    vault = {
        "./proj/dist/bundle.mjs": [
            {"id": "g-1", "line": 0, "title": "show languages as a dropdown", "text": ""}
        ]
    }
    (workspace / ".margo" / "local.notes.json").write_text(json.dumps(vault), encoding="utf-8")
    return next(n for n in read_margo_notes(workspace) if n.id == "g-1")


def test_a_note_on_ignored_build_output_can_still_start(service, workspace, generated_note):
    task = service.prepare(generated_note, workspace)

    assert task.anchor_kind == "generated" and task.anchor_text == ""
    assert task.file == "dist/bundle.mjs"
    assert not Path(task.worktree, "dist").exists()  # the agent cannot see build output


def test_the_prompt_for_generated_files_points_the_agent_at_the_sources(
    service, home, workspace, generated_note
):
    task, record = service.open_chat(generated_note, workspace)

    (prompt,) = _messages(home, record.path)
    assert "git does not track" in prompt and "dist/bundle.mjs" in prompt
    assert "find the source files" in prompt and "show languages as a dropdown" in prompt
    assert "var a=1;" not in prompt  # no minified junk in the context
    assert "Code around line" not in prompt
    _task, discussion = service.open_chat(generated_note, workspace, discuss=True)
    assert discussion.path == record.path  # same chat; discuss only matters at creation


def test_discussing_a_generated_note_tells_the_agent_not_to_edit(
    service, home, workspace, generated_note
):
    _task, record = service.open_chat(generated_note, workspace, discuss=True)

    (prompt,) = _messages(home, record.path)
    assert "Do not edit any file until they ask" in prompt


def test_uncommitted_changes_are_reported_as_such(service, workspace):
    (workspace / "proj" / "src" / "data.py").write_text(SOURCE + "# wip\n", encoding="utf-8")

    with pytest.raises(GitError) as raised:
        service.prepare(_note(workspace), workspace)

    assert "has uncommitted changes" in str(raised.value)
    assert "not tracked" not in str(raised.value)


# -- merges that wait for the user's commit -------------------------------------------


def _ready_task(service, workspace, monkeypatch):
    _fake_agent(monkeypatch, edits=[("item['name']", "item.name")])
    task = service.prepare(_note(workspace), workspace)
    service.run(task.note_id)
    return task


def test_the_default_merge_waits_for_the_users_commit(service, workspace, monkeypatch):
    task = _ready_task(service, workspace, monkeypatch)
    repo = workspace / "proj"

    result = service.merge(task.note_id)

    assert result.ok and result.pending
    assert service.tasks.get(task.note_id).status is TaskStatus.MERGING
    assert service.refresh_merge_state(task.note_id) is TaskStatus.MERGING
    assert Path(task.worktree).exists()  # the task keeps its worktree until it is committed
    assert "src/data.py" in git(repo, "diff", "--cached", "--name-only")

    git(repo, "commit", "--no-edit")  # the user commits in Source Control

    assert service.refresh_merge_state(task.note_id) is TaskStatus.MERGED
    assert not Path(task.worktree).exists() and not service.git.branch_exists(task.branch)


def test_aborting_the_merge_in_vscode_returns_the_task_to_ready(service, workspace, monkeypatch):
    task = _ready_task(service, workspace, monkeypatch)
    service.merge(task.note_id)

    git(workspace / "proj", "merge", "--abort")  # "Abort merge" in Source Control

    assert service.refresh_merge_state(task.note_id) is TaskStatus.READY
    assert service.merge(task.note_id).pending  # and it can be staged again


def test_aborting_from_anomx_unstages_everything(service, workspace, monkeypatch):
    task = _ready_task(service, workspace, monkeypatch)
    service.merge(task.note_id)

    service.abort_pending_merge(task.note_id)

    assert service.tasks.get(task.note_id).status is TaskStatus.READY
    assert not service.git.merge_in_progress() and service.git.is_clean()
    assert (workspace / "proj" / "src" / "data.py").read_text("utf-8") == SOURCE


def test_discarding_a_task_with_a_staged_merge_undoes_the_merge(service, workspace, monkeypatch):
    task = _ready_task(service, workspace, monkeypatch)
    service.merge(task.note_id)

    service.discard(task.note_id)

    assert not service.git.merge_in_progress()
    assert (workspace / "proj" / "src" / "data.py").read_text("utf-8") == SOURCE
    assert service.tasks.get(task.note_id).status is TaskStatus.DISCARDED


def test_the_panel_reports_a_staged_merge_and_notices_the_commit(
    service, home, workspace, monkeypatch
):
    from anomx.agent.cowork import CoworkWorkspace

    task = _ready_task(service, workspace, monkeypatch)
    panel = CoworkWorkspace.locate(home, workspace)
    row = next(r for r in panel.rows() if r.note_id == task.note_id)
    assert panel.merge(row).pending

    merging = next(r for r in panel.rows() if r.note_id == task.note_id)
    assert merging.status == "merging" and "Source Control" in merging.detail
    assert [a.value for a in panel.actions(merging)] == ["diff", "abort_merge"]

    git(workspace / "proj", "commit", "--no-edit")

    assert next(r for r in panel.rows() if r.note_id == task.note_id).status == "merged"
