"""Run one cowork task: prepare a worktree for a note, let the agent work, commit, merge.

The service owns everything about tasks of one repository. It never touches Margo's files
and never changes the user's working tree except through ``follow`` and ``merge``, which
refuse to run over uncommitted work (see ``git.py``).
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anomx.agent.base.agents import AgentKind
from anomx.agent.cowork.agent import CoworkRuntime
from anomx.agent.cowork.git import CoworkGit, FollowResult, GitError, MergeResult, branch_for
from anomx.agent.cowork.margo import MARGO_LINE_BASE, MargoNote, NoteContext, note_context
from anomx.agent.cowork.tasks import CoworkTask, TaskStatus, TaskStore
from anomx.agent.helpers.mode import AgentMode
from anomx.agent.runtime import RuntimeCallbacks
from anomx.agent.store import AnomxHome, SessionRecord

MAX_SUMMARY_CHARS = 2_000
RUNNABLE = {TaskStatus.QUEUED, TaskStatus.FAILED}


@dataclass(frozen=True)
class RunResult:
    """Outcome of one agent run."""

    task: CoworkTask
    committed: bool
    summary: str
    error: str = ""


def awaiting_reply(events: Sequence[Mapping[str, Any]]) -> bool:
    """Whether the latest user message of a transcript has not been answered yet.

    A fresh cowork chat contains only the note; nothing is sent to the model until a turn
    is started, so this is what tells the screens that the note still has to be sent.
    """

    state = False
    for event in events:
        payload = event.get("payload")
        if event.get("type") != "event_msg" or not isinstance(payload, dict):
            continue
        kind = payload.get("type")
        if kind == "user_message":
            state = True
        elif kind == "agent_message":
            state = False
    return state


def build_task_prompt(task: CoworkTask, context: NoteContext, *, discuss: bool = False) -> str:
    """The only thing the agent is told: this note and the code around it.

    With ``discuss`` the user wants to talk the note through first, so the agent must not
    edit anything until asked.
    """

    if task.anchor_kind == "generated":
        lines = [
            "The user left this note in their editor on a file that git does not track "
            f"({task.file}). That is generated output (for example a build result) or a new "
            "file, so it is not in your worktree and must not be edited.",
            "",
            "Note:",
            task.request,
            "",
            "Treat the note as an instruction for the project: find the source files that "
            "produce what the note is about and change those.",
        ]
        if discuss:
            lines.append(
                "The user wants to talk this through first. Do not edit any file until they "
                "ask you to make a change; name the options and ask what they prefer."
            )
        return "\n".join(lines)
    lines = [
        "The user left this note in their editor:",
        "",
        f"File: {task.file}",
        f"Line: {task.line}",
        "Note:",
        task.request,
        "",
    ]
    if context.snippet:
        lines += [
            f"Code around line {task.line} (line numbers on the left):",
            context.snippet,
            "",
        ]
    if discuss:
        lines.append(
            "The user wants to talk this through first. Do not edit any file until they ask "
            "you to make a change. Answer concisely, name the options and their trade-offs "
            "and ask what they prefer."
        )
    else:
        lines.append("Make the change the note asks for, then give your short summary.")
    return "\n".join(lines)


class CoworkService:
    """Tasks, worktrees and agent runs for one git repository."""

    def __init__(self, home: AnomxHome, repo_root: Path) -> None:
        self.home = home
        self.repo_root = repo_root.resolve()
        self.git = CoworkGit(self.repo_root)
        key = hashlib.sha256(str(self.repo_root).encode("utf-8")).hexdigest()[:12]
        self.state_dir = home.root / "cowork" / f"{self.repo_root.name}-{key}"
        self.tasks = TaskStore(self.state_dir / "tasks.json")
        self.worktrees = self.state_dir / "worktrees"
        self._running: dict[str, CoworkRuntime] = {}
        self._outcomes: dict[str, tuple[str | None, str]] = {}
        self._lock = threading.Lock()

    # -- preparing --------------------------------------------------------------------

    def prepare(self, note: MargoNote, workspace: Path) -> CoworkTask:
        """Create the branch and worktree for a note (no agent runs yet)."""

        existing = self.tasks.get(note.id)
        if existing is not None and existing.is_active:
            raise GitError(f"The note already has an active task ({existing.status.value}).")
        absolute = (workspace / note.file).resolve()
        try:
            relative = absolute.relative_to(self.repo_root).as_posix()
        except ValueError:
            raise GitError(f"{note.file} is not inside {self.repo_root}.") from None
        state = self.git.file_state(relative)
        if state == "modified":
            raise GitError(
                f"{relative} has uncommitted changes. The agent works from the committed "
                "version, so commit or stash it first."
            )
        # Ignored or untracked files (for example dist/ output) are not part of the agent's
        # checkout; the note then acts as an instruction for the project.
        generated = state in {"ignored", "untracked"}
        context = note_context(workspace, note)
        workspace_info = self.git.create_task(note.id, self.worktrees / _slug(note.id))
        task = CoworkTask(
            note_id=note.id,
            file=relative,
            line=note.line_number,
            title=note.title,
            request=note.request,
            branch=workspace_info.branch,
            worktree=str(workspace_info.path),
            base_branch=workspace_info.base_branch,
            base_commit=workspace_info.base_commit,
            anchor_text="" if generated else context.anchor_text,
            anchor_kind="generated" if generated else "source",
        )
        return self.tasks.save(task)

    # -- running ----------------------------------------------------------------------

    def run(self, note_id: str, callbacks: RuntimeCallbacks | None = None) -> RunResult:
        """Run the agent for a prepared task and commit what it did. Blocks until done.

        When the task already has a chat (``open_chat``), the agent continues that
        conversation; otherwise a new session with the note as its first message is made.
        """

        task = self._require(note_id)
        if task.status not in RUNNABLE:
            raise GitError(f"Task is {task.status.value}; only queued or failed tasks can run.")
        self.tasks.update(note_id, status=TaskStatus.RUNNING, summary="")

        session_path = self._existing_session(task)
        if session_path is None:
            session_path = self._create_session(task, discuss=False).path
        else:
            self.home.append_session_event(
                session_path, "user_message", {"message": "Please make the change now."}
            )
        runtime = self.runtime_for(note_id)
        with self._lock:
            self._running[note_id] = runtime
        self._outcomes.pop(note_id, None)

        try:
            runtime.backend_response(session_path, callbacks or RuntimeCallbacks())
        except Exception as exc:  # noqa: BLE001 - the task must end in a recorded state
            error = f"{type(exc).__name__}: {exc}"
            self._finish(note_id, "", aborted=False, failed=True, error=error)
        finally:
            with self._lock:
                self._running.pop(note_id, None)
        updated = self._require(note_id)
        commit, error = self._outcomes.pop(note_id, (None, ""))
        return RunResult(updated, commit is not None, updated.summary if not error else "", error)

    # -- chat -------------------------------------------------------------------------

    def open_chat(
        self, note: MargoNote, workspace: Path, *, discuss: bool = False
    ) -> tuple[CoworkTask, SessionRecord]:
        """Task and chat session of a note, created on first use.

        The chat is an ordinary anomx session whose working directory is the task's
        worktree; its first message is the note. ``discuss`` tells the agent to talk first
        instead of editing.
        """

        task = self.tasks.get(note.id)
        if task is None or not task.is_active:
            task = self.prepare(note, workspace)
        existing = self._existing_session(task)
        if existing is not None:
            record = self.home.session_record(existing)
            if record is not None:
                return task, record
        record = self._create_session(task, discuss=discuss)
        return self._require(note.id), record

    def runtime_for(self, note_id: str) -> CoworkRuntime:
        """A runtime for the task's worktree that commits after every model turn."""

        task = self._require(note_id)
        runtime = CoworkRuntime(self.home, Path(task.worktree), mode=AgentMode.STANDARD)

        def after_turn(response: str, aborted: bool, failed: bool) -> None:
            error = ""
            if failed:
                error = response or "The model request failed."
            elif aborted:
                error = "Stopped before the agent finished."
            self._finish(note_id, response, aborted=aborted, failed=bool(error), error=error)

        runtime.after_turn = after_turn
        return runtime

    def awaiting_reply(self, session_path: Path) -> bool:
        return awaiting_reply(self.home.read_session_events(session_path))

    def note_for_session(self, session_path: Path) -> str | None:
        """Note whose chat is this session, if any."""

        for task in self.tasks.list():
            if task.session_path and Path(task.session_path) == session_path:
                return task.note_id
        return None

    def _existing_session(self, task: CoworkTask) -> Path | None:
        if not task.session_path:
            return None
        path = Path(task.session_path)
        return path if path.exists() else None

    def _create_session(self, task: CoworkTask, *, discuss: bool) -> SessionRecord:
        worktree = Path(task.worktree)
        config = self.home.load_config()
        session = self.home.create_session(
            worktree,
            str(config.get("provider", "")),
            str(config.get("model", "")),
            agent_kind=AgentKind.MAIN,  # the screen shows the regular agent; the runtime differs
        )
        self.home.update_session_title(session.path, task.title or "Cowork note")
        context = note_context(worktree, _note_view(task))
        self.home.append_session_event(
            session.path,
            "user_message",
            {"message": build_task_prompt(task, context, discuss=discuss)},
        )
        self.tasks.update(task.note_id, session_path=str(session.path))
        return self.home.session_record(session.path) or session

    def _finish(
        self, note_id: str, response: str, *, aborted: bool, failed: bool, error: str
    ) -> None:
        """Commit what a turn produced and record the outcome (also used for chat turns)."""

        del aborted
        task = self.tasks.get(note_id)
        if task is None:
            return
        summary = response.strip()[:MAX_SUMMARY_CHARS]
        try:
            commit = self.git.commit_all(
                Path(task.worktree), _commit_message(task, summary or error)
            )
        except GitError as exc:
            commit, error, failed = None, error or str(exc), True
        self._outcomes[note_id] = (commit, error if failed else "")
        status = TaskStatus.FAILED if failed else TaskStatus.READY
        self.tasks.update(note_id, status=status, summary=error if failed else summary)

    def abort(self, note_id: str) -> bool:
        """Ask a running agent to stop; ``run`` then records the task as failed."""

        with self._lock:
            runtime = self._running.get(note_id)
        if runtime is None:
            return False
        runtime.abort_current_turn()
        return True

    def is_running(self, note_id: str) -> bool:
        with self._lock:
            return note_id in self._running

    # -- reviewing --------------------------------------------------------------------

    def diff(self, note_id: str) -> str:
        task = self._require(note_id)
        return self.git.diff(task.branch, task.base_commit)

    def changed_files(self, note_id: str) -> list[str]:
        task = self._require(note_id)
        return self.git.changed_files(task.branch, task.base_commit)

    def follow(self, note_id: str) -> FollowResult:
        """Show the task's current state in the user's working tree."""

        return self.git.follow(self._require(note_id).branch)

    def return_to_base(self, note_id: str) -> FollowResult:
        return self.git.return_to(self._require(note_id).base_branch)

    def merge(self, note_id: str) -> MergeResult:
        """Merge an approved task into its base branch and clean it up."""

        task = self._require(note_id)
        if task.status is not TaskStatus.READY:
            return MergeResult(False, reason=f"Task is {task.status.value}, not ready.")
        if self.git.commits_ahead(task.branch, task.base_commit) == 0:
            return MergeResult(False, reason="The agent made no changes; nothing to merge.")
        result = self.git.merge(task.branch, task.base_branch, f"Merge cowork task: {task.title}")
        if result.ok:
            self.git.remove_task(task.branch, Path(task.worktree))
            self.tasks.update(note_id, status=TaskStatus.MERGED)
        return result

    def discard(self, note_id: str) -> None:
        """Throw a task away, including unmerged work."""

        task = self._require(note_id)
        self.abort(note_id)
        if self.git.current_branch() is None:
            # The user may be looking at this very task; step back before deleting it.
            self.git.return_to(task.base_branch)
        self.git.remove_task(task.branch, Path(task.worktree), force=True)
        self.tasks.update(note_id, status=TaskStatus.DISCARDED)

    def _require(self, note_id: str) -> CoworkTask:
        task = self.tasks.get(note_id)
        if task is None:
            raise GitError(f"No cowork task for note {note_id}.")
        return task


def _slug(note_id: str) -> str:
    return branch_for(note_id).removeprefix("anomx/cowork/")


def _note_view(task: CoworkTask) -> MargoNote:
    """The task's note as seen from inside its worktree (file path relative to it)."""

    return MargoNote(
        id=task.note_id,
        file=task.file,
        line=task.line - 1 + MARGO_LINE_BASE,
        title=task.title,
        text="",
    )


def _commit_message(task: CoworkTask, summary: str) -> str:
    body = summary.strip() or "No summary."
    return f"cowork: {task.title}\n\n{body}\n\nNote: {task.note_id}"
