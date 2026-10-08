"""Cowork core: Margo vault reader, task state and git worktree handling."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from anomx.agent.cowork import (
    CoworkGit,
    CoworkTask,
    GitError,
    TaskStatus,
    TaskStore,
    find_margo_workspace,
    find_repo_root,
    normalize_note_path,
    note_context,
    read_margo_notes,
)
from anomx.agent.cowork.git import branch_for

AUTH = (
    "def get_token():\n    return {'user': 'a'}\n\n\n"
    "def authenticate(token):\n    return token['user']\n"
)


def run(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    run(root, "init", "-b", "main")
    run(root, "config", "user.name", "Dev")
    run(root, "config", "user.email", "dev@example.com")
    (root / "src" / "auth.py").write_text(AUTH, encoding="utf-8")
    run(root, "add", "-A")
    run(root, "commit", "-m", "initial")
    return root


# -- Margo reader ---------------------------------------------------------------------


def test_reads_both_vaults_and_hides_resolved_notes(tmp_path):
    (tmp_path / ".margo").mkdir()
    local = {
        "src/auth.py": [
            {"id": "1-a", "line": 0, "title": "Make dict a dataclass", "text": "also callers"},
            {"id": "2-b", "line": 4, "title": "done", "text": "", "status": "resolved"},
        ]
    }
    team = {"src/other.py": [{"id": "3-c", "line": 2, "title": "Shared", "vault": "team"}]}
    (tmp_path / ".margo" / "local.notes.json").write_text(json.dumps(local), encoding="utf-8")
    (tmp_path / ".notes.json").write_text(json.dumps(team), encoding="utf-8")

    notes = read_margo_notes(tmp_path)

    assert [(n.id, n.file, n.vault) for n in notes] == [
        ("1-a", "src/auth.py", "local"),
        ("3-c", "src/other.py", "team"),
    ]
    assert notes[0].line_number == 1  # Margo stores the first line as 0
    assert notes[0].request == "Make dict a dataclass\n\nalso callers"
    assert len(read_margo_notes(tmp_path, include_resolved=True)) == 3


def test_reader_skips_malformed_files_and_entries(tmp_path):
    (tmp_path / ".notes.json").write_text("{not json", encoding="utf-8")
    (tmp_path / ".margo").mkdir()
    broken = {
        "a.py": [
            {"id": "ok", "line": 1, "title": "fine"},
            {"id": "", "line": 1},
            {"id": "neg", "line": -3},
            {"id": "bool", "line": True},
            "string",
        ],
        "b.py": "not a list",
    }
    (tmp_path / ".margo" / "local.notes.json").write_text(json.dumps(broken), encoding="utf-8")

    assert [n.id for n in read_margo_notes(tmp_path)] == ["ok"]
    assert read_margo_notes(tmp_path / "missing") == []


def test_note_context_numbers_lines_and_records_anchor_text(repo):
    (repo / ".margo").mkdir()
    vault = {"src/auth.py": [{"id": "n1", "line": 1, "title": "x"}]}
    (repo / ".margo" / "local.notes.json").write_text(json.dumps(vault), encoding="utf-8")
    note = read_margo_notes(repo)[0]

    context = note_context(repo, note, radius=1)

    assert context.file_exists
    assert context.anchor_text == "    return {'user': 'a'}"
    assert context.first_line == 1
    assert "    2      return {'user': 'a'}" in context.snippet


def test_note_context_for_missing_file(tmp_path):
    (tmp_path / ".notes.json").write_text(
        json.dumps({"gone.py": [{"id": "n", "line": 0, "title": "x"}]}), encoding="utf-8"
    )
    note = read_margo_notes(tmp_path)[0]

    assert note_context(tmp_path, note).file_exists is False


# -- task store -----------------------------------------------------------------------


def _task(note_id: str = "1-a") -> CoworkTask:
    return CoworkTask(
        note_id=note_id,
        file="src/auth.py",
        line=2,
        title="t",
        request="r",
        branch=branch_for(note_id),
        worktree="/tmp/w",
        base_branch="main",
        base_commit="abc",
    )


def test_task_store_roundtrip_update_and_remove(tmp_path):
    store = TaskStore(tmp_path / "cowork" / "tasks.json")
    store.save(_task("1-a"))
    store.save(_task("2-b"))

    store.update("1-a", status=TaskStatus.READY, summary="converted")
    task = store.get("1-a")

    assert task is not None and task.status is TaskStatus.READY and task.summary == "converted"
    assert [t.note_id for t in store.list()] == ["1-a", "2-b"]
    assert task.is_active
    store.remove("1-a")
    assert store.get("1-a") is None
    assert store.update("missing", summary="x") is None


def test_task_store_tolerates_corrupt_file_and_unknown_fields(tmp_path):
    path = tmp_path / "tasks.json"
    path.write_text("{broken", encoding="utf-8")
    assert TaskStore(path).list() == []

    good = {"version": 9, "tasks": {"n": {**_task("n").__dict__, "status": "ready", "future": 1}}}
    path.write_text(json.dumps(good), encoding="utf-8")

    (task,) = TaskStore(path).list()
    assert task.status is TaskStatus.READY


# -- git ------------------------------------------------------------------------------


def test_branch_names_are_valid_refs():
    assert branch_for("1759933000000-abc def/ghi") == "anomx/cowork/1759933000000-abc-def-ghi"


def test_task_runs_in_its_own_worktree_without_touching_the_users_tree(repo, tmp_path):
    git = CoworkGit(repo)
    workspace = git.create_task("n1", tmp_path / "wt" / "n1")

    (workspace.path / "src" / "auth.py").write_text("changed\n", encoding="utf-8")
    commit = git.commit_all(workspace.path, "Use dataclass")

    assert commit and git.commits_ahead(workspace.branch, "main") == 1
    assert git.changed_files(workspace.branch, "main") == ["src/auth.py"]
    assert (repo / "src" / "auth.py").read_text(encoding="utf-8") == AUTH
    assert git.current_branch() == "main" and git.is_clean()
    assert git.commit_all(workspace.path, "nothing to commit") is None


def test_two_tasks_work_in_parallel_without_interfering(repo, tmp_path):
    git = CoworkGit(repo)
    first = git.create_task("n1", tmp_path / "wt" / "n1")
    second = git.create_task("n2", tmp_path / "wt" / "n2")
    (first.path / "src" / "auth.py").write_text("one\n", encoding="utf-8")
    (second.path / "src" / "auth.py").write_text("two\n", encoding="utf-8")
    git.commit_all(first.path, "one")
    git.commit_all(second.path, "two")

    assert (first.path / "src" / "auth.py").read_text(encoding="utf-8") == "one\n"
    assert (second.path / "src" / "auth.py").read_text(encoding="utf-8") == "two\n"


def test_follow_shows_the_task_state_and_return_restores_the_branch(repo, tmp_path):
    git = CoworkGit(repo)
    workspace = git.create_task("n1", tmp_path / "wt" / "n1")
    (workspace.path / "src" / "auth.py").write_text("agent version\n", encoding="utf-8")
    git.commit_all(workspace.path, "agent")

    assert git.follow(workspace.branch).ok
    assert git.current_branch() is None  # detached, no branch is moved
    assert (repo / "src" / "auth.py").read_text(encoding="utf-8") == "agent version\n"

    assert git.return_to("main").ok
    assert git.current_branch() == "main"
    assert (repo / "src" / "auth.py").read_text(encoding="utf-8") == AUTH


def test_follow_is_refused_when_the_user_has_uncommitted_changes(repo, tmp_path):
    git = CoworkGit(repo)
    workspace = git.create_task("n1", tmp_path / "wt" / "n1")
    (repo / "src" / "auth.py").write_text("my unsaved work\n", encoding="utf-8")

    result = git.follow(workspace.branch)

    assert not result.ok and "uncommitted" in result.reason
    assert git.current_branch() == "main"
    assert (repo / "src" / "auth.py").read_text(encoding="utf-8") == "my unsaved work\n"


def test_follow_ignores_untracked_files(repo, tmp_path):
    git = CoworkGit(repo)
    workspace = git.create_task("n1", tmp_path / "wt" / "n1")
    (repo / "scratch.txt").write_text("notes", encoding="utf-8")

    assert git.follow(workspace.branch).ok
    assert (repo / "scratch.txt").exists()


def test_merge_brings_the_task_into_the_base_branch(repo, tmp_path):
    git = CoworkGit(repo)
    workspace = git.create_task("n1", tmp_path / "wt" / "n1")
    (workspace.path / "src" / "auth.py").write_text("merged\n", encoding="utf-8")
    git.commit_all(workspace.path, "agent change")
    git.follow(workspace.branch)  # the user was looking at the task

    result = git.merge(workspace.branch, "main", "Merge cowork task n1")

    assert result.ok
    assert git.current_branch() == "main"
    assert (repo / "src" / "auth.py").read_text(encoding="utf-8") == "merged\n"
    assert "Merge cowork task n1" in run(repo, "log", "-1", "--format=%s")
    git.remove_task(workspace.branch, workspace.path)
    assert not git.branch_exists(workspace.branch) and not workspace.path.exists()


def test_conflicting_merge_is_aborted_and_reported(repo, tmp_path):
    git = CoworkGit(repo)
    workspace = git.create_task("n1", tmp_path / "wt" / "n1")
    (workspace.path / "src" / "auth.py").write_text("agent\n", encoding="utf-8")
    git.commit_all(workspace.path, "agent change")
    (repo / "src" / "auth.py").write_text("user\n", encoding="utf-8")
    run(repo, "commit", "-am", "user change on main")

    result = git.merge(workspace.branch, "main", "Merge")

    assert not result.ok and result.conflicts == ("src/auth.py",)
    assert git.is_clean() and git.current_branch() == "main"
    assert (repo / "src" / "auth.py").read_text(encoding="utf-8") == "user\n"
    assert git.branch_exists(workspace.branch)  # the work is kept for the user


def test_merge_is_refused_with_uncommitted_changes(repo, tmp_path):
    git = CoworkGit(repo)
    workspace = git.create_task("n1", tmp_path / "wt" / "n1")
    (workspace.path / "src" / "auth.py").write_text("agent\n", encoding="utf-8")
    git.commit_all(workspace.path, "agent")
    (repo / "src" / "auth.py").write_text("dirty\n", encoding="utf-8")

    result = git.merge(workspace.branch, "main", "Merge")

    assert not result.ok and "overwritten" in result.reason  # git refuses: the file overlaps


def test_discarding_an_unmerged_task_needs_force(repo, tmp_path):
    git = CoworkGit(repo)
    workspace = git.create_task("n1", tmp_path / "wt" / "n1")
    (workspace.path / "src" / "auth.py").write_text("unmerged\n", encoding="utf-8")
    git.commit_all(workspace.path, "agent")

    git.remove_task(workspace.branch, workspace.path)  # safe delete refuses unmerged work
    assert git.branch_exists(workspace.branch)

    git.remove_task(workspace.branch, workspace.path, force=True)
    assert not git.branch_exists(workspace.branch)


def test_a_detached_head_cannot_host_tasks(repo, tmp_path):
    run(repo, "switch", "--detach", "HEAD")

    with pytest.raises(GitError, match="Check out a branch"):
        CoworkGit(repo).create_task("n1", tmp_path / "wt" / "n1")


# -- real-world layout: workspace folder above one or more repositories ----------------

# Verbatim shape of a vault written by Margo 0.1.0 (keys carry a leading "./").
REAL_VAULT = {
    "./anomx/examples/baseline_detection.py": [
        {
            "id": "1791449926580-x93ibwz6i",
            "line": 2,
            "title": "this is a comment",
            "text": "",
            "timestamp": 1791449926580,
            "author": "ultrasat",
            "priority": "medium",
            "category": "note",
            "status": "open",
            "vault": "local",
        }
    ]
}


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A VS Code workspace folder that contains a git repository (like app_project/anomx)."""

    root = tmp_path / "workspace"
    project = root / "anomx"
    (project / "examples").mkdir(parents=True)
    run(project, "init", "-b", "main")
    run(project, "config", "user.name", "Dev")
    run(project, "config", "user.email", "dev@example.com")
    source = '"""Doc."""\n\nfrom anomx import (\n    A,\n)\n'
    (project / "examples" / "baseline_detection.py").write_text(source, encoding="utf-8")
    run(project, "add", "-A")
    run(project, "commit", "-m", "initial")
    (root / ".margo").mkdir()
    (root / ".margo" / "local.notes.json").write_text(json.dumps(REAL_VAULT), encoding="utf-8")
    return root


def test_vault_in_a_parent_folder_is_found_from_inside_the_repo(workspace):
    assert find_margo_workspace(workspace / "anomx" / "examples") == workspace
    assert find_margo_workspace(workspace.parent.parent / "nowhere") is None


def test_real_vault_keys_lose_the_leading_dot_slash(workspace):
    (note,) = read_margo_notes(workspace)

    assert note.file == "anomx/examples/baseline_detection.py"
    assert normalize_note_path("././a\\b.py") == "a/b.py"
    assert note.line_number == 3


def test_note_is_resolved_to_the_repository_of_its_own_file(workspace):
    (note,) = read_margo_notes(workspace)
    target = workspace / note.file

    repo_root = find_repo_root(target)

    assert repo_root == workspace / "anomx"
    assert target.relative_to(repo_root).as_posix() == "examples/baseline_detection.py"
    context = note_context(workspace, note, radius=0)
    assert context.anchor_text == "from anomx import ("
    assert find_repo_root(workspace.parent) is None  # not inside any repository


def test_missing_files_still_resolve_to_a_repository(workspace):
    assert find_repo_root(workspace / "anomx" / "deleted" / "gone.py") == workspace / "anomx"


def test_file_state_distinguishes_committed_modified_ignored_and_untracked(repo):
    (repo / ".gitignore").write_text("dist/\n", encoding="utf-8")
    (repo / "dist").mkdir()
    (repo / "dist" / "bundle.js").write_text("built\n", encoding="utf-8")
    (repo / "new.py").write_text("x = 1\n", encoding="utf-8")
    run(repo, "add", ".gitignore")
    run(repo, "commit", "-m", "ignore dist")
    git = CoworkGit(repo)
    (repo / "src" / "edited.py").write_text("a\n", encoding="utf-8")
    run(repo, "add", "src/edited.py")
    run(repo, "commit", "-m", "add edited")
    (repo / "src" / "edited.py").write_text("a changed\n", encoding="utf-8")

    assert git.file_state("src/auth.py") == "committed"
    assert git.file_state("src/edited.py") == "modified"
    assert git.file_state("dist/bundle.js") == "ignored"
    assert git.file_state("new.py") == "untracked"
    assert git.is_file_committed("src/auth.py") and not git.is_file_committed("dist/bundle.js")


def test_tasks_saved_before_anchor_kind_existed_load_as_source(tmp_path):
    path = tmp_path / "tasks.json"
    task = _task("n")
    data = {"version": 1, "tasks": {"n": {**task.__dict__, "status": "queued"}}}
    data["tasks"]["n"].pop("anchor_kind", None)
    path.write_text(json.dumps(data), encoding="utf-8")

    assert TaskStore(path).get("n").anchor_kind == "source"


def test_changed_entries_and_file_at_describe_added_modified_and_deleted(repo, tmp_path):
    git = CoworkGit(repo)
    (repo / "src" / "gone.py").write_text("old\n", encoding="utf-8")
    run(repo, "add", "-A")
    run(repo, "commit", "-m", "add gone")
    workspace = git.create_task("n1", tmp_path / "wt" / "n1")
    (workspace.path / "src" / "auth.py").write_text("changed\n", encoding="utf-8")
    (workspace.path / "src" / "added.py").write_text("new\n", encoding="utf-8")
    (workspace.path / "src" / "gone.py").unlink()
    git.commit_all(workspace.path, "agent")

    entries = dict((path, status) for status, path in git.changed_entries(workspace.branch, "main"))

    assert entries == {"src/auth.py": "M", "src/added.py": "A", "src/gone.py": "D"}
    assert git.file_at(workspace.base_commit, "src/gone.py") == b"old\n"
    assert git.file_at(workspace.base_commit, "src/added.py") is None


# -- cleanliness, pending merges ------------------------------------------------------


def _agent_change(git: CoworkGit, tmp_path: Path, name: str = "n1", text: str = "agent\n"):
    workspace = git.create_task(name, tmp_path / "wt" / name)
    (workspace.path / "src" / "auth.py").write_text(text, encoding="utf-8")
    git.commit_all(workspace.path, "agent change")
    return workspace


def test_a_modified_submodule_does_not_make_the_tree_unclean(repo, tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    run(sub, "init", "-b", "main")
    run(sub, "config", "user.name", "Dev")
    run(sub, "config", "user.email", "dev@example.com")
    (sub / "f.txt").write_text("1\n", encoding="utf-8")
    run(sub, "add", "-A")
    run(sub, "commit", "-m", "sub init")
    run(repo, "-c", "protocol.file.allow=always", "submodule", "add", str(sub), "vendor/sub")
    run(repo, "commit", "-m", "add submodule")
    (sub / "f.txt").write_text("2\n", encoding="utf-8")
    run(sub, "commit", "-am", "sub moves on")
    run(repo / "vendor" / "sub", "-c", "protocol.file.allow=always", "pull", "-q", "origin", "main")
    assert " M vendor/sub" in run(repo, "status", "--porcelain", "--untracked-files=no")

    assert CoworkGit(repo).is_clean()  # git status is not clean, but nothing a merge touches


def test_unrelated_uncommitted_work_does_not_block_a_merge(repo, tmp_path):
    git = CoworkGit(repo)
    workspace = _agent_change(git, tmp_path)
    (repo / "notes.txt").write_text("mine\n", encoding="utf-8")
    run(repo, "add", "notes.txt")
    run(repo, "commit", "-m", "tracked file")
    (repo / "notes.txt").write_text("my unsaved edit\n", encoding="utf-8")

    result = git.merge(workspace.branch, "main", "Merge")

    assert result.ok
    assert (repo / "notes.txt").read_text(encoding="utf-8") == "my unsaved edit\n"


def test_pending_merge_stages_the_changes_without_committing(repo, tmp_path):
    git = CoworkGit(repo)
    workspace = _agent_change(git, tmp_path)
    head_before = git.head()

    result = git.merge_pending(workspace.branch, "main", "Merge cowork task: x")

    assert result.ok and result.pending
    assert git.merge_in_progress() and git.head() == head_before  # nothing committed yet
    assert run(repo, "diff", "--cached", "--name-only").split() == ["src/auth.py"]
    assert (repo / "src" / "auth.py").read_text(encoding="utf-8") == "agent\n"
    assert "Merge cowork task: x" in (repo / ".git" / "MERGE_MSG").read_text(encoding="utf-8")
    assert not git.is_ancestor(workspace.branch, "main")  # not merged until the user commits

    run(repo, "commit", "--no-edit")  # what the user does in Source Control

    assert git.is_ancestor(workspace.branch, "main") and not git.merge_in_progress()


def test_a_pending_merge_can_be_aborted(repo, tmp_path):
    git = CoworkGit(repo)
    workspace = _agent_change(git, tmp_path)
    git.merge_pending(workspace.branch, "main", "Merge")

    git.abort_merge()

    assert not git.merge_in_progress() and git.is_clean() and not git.has_staged_changes()
    assert (repo / "src" / "auth.py").read_text(encoding="utf-8") == AUTH


def test_pending_merge_is_refused_with_staged_changes_or_a_running_merge(repo, tmp_path):
    git = CoworkGit(repo)
    first = _agent_change(git, tmp_path, "n1", "one\n")
    second = _agent_change(git, tmp_path, "n2", "two\n")
    (repo / "extra.txt").write_text("staged work\n", encoding="utf-8")
    run(repo, "add", "extra.txt")

    staged = git.merge_pending(first.branch, "main", "Merge")
    assert not staged.ok and "staged changes" in staged.reason

    run(repo, "reset", "-q", "extra.txt")
    (repo / "extra.txt").unlink()
    assert git.merge_pending(first.branch, "main", "Merge").pending
    again = git.merge_pending(second.branch, "main", "Merge")
    assert not again.ok and "already in progress" in again.reason


def test_a_conflicting_pending_merge_is_aborted_cleanly(repo, tmp_path):
    git = CoworkGit(repo)
    workspace = _agent_change(git, tmp_path)
    (repo / "src" / "auth.py").write_text("user\n", encoding="utf-8")
    run(repo, "commit", "-am", "user change on main")

    result = git.merge_pending(workspace.branch, "main", "Merge")

    assert not result.ok and result.conflicts == ("src/auth.py",)
    assert not git.merge_in_progress() and git.is_clean()
